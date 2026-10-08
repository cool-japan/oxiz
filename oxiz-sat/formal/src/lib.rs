//! `oxiz-sat-formal`: machine-checked obligations for `oxiz-sat`'s literal
//! encoding.
//!
//! `oxiz-sat` is the CDCL core of OxiZ. Its `Lit` packs a variable index and a
//! sign into a single `u32` as `index << 1 | sign`, and its `to_dimacs` /
//! `from_dimacs` carry that packing across the DIMACS boundary. Both encodings
//! are total only under a bound, and until the fix recorded in this working
//! tree neither the bound nor its violation was visible anywhere: an index at
//! or above `2^31` silently produced a literal denoting a *different*
//! variable, and the DIMACS conversion returned a wrong number rather than
//! failing. This package states nine such properties as `#[harness]` entry
//! points over the **public** API of `oxiz-sat`, and records the verdict
//! `cargo formal check` actually returned for each of them. Since wave D2 of
//! cargo-formal's Phase 3 it also restates each of the nine as a
//! `#[requires]`/`#[ensures]` contract on a `spec_*` wrapper in [`spec`], with
//! a `#[proof_for_contract]` harness beside it, so the same property is
//! measured once at L1 (an `assert` inside a harness) and once at L2 (an
//! `ensures` on a contracted function). The contract is on the **wrapper**;
//! nothing in `oxiz-sat` is annotated.
//!
//! Nothing here is a copy of `oxiz-sat`: it is an ordinary path dependency
//! (`..`), and `cargo-formal` lowers its reachable bodies into this crate's
//! verification conditions through the `dep-crates` key of
//! `[package.metadata.formal]`. Every target is reachable from outside the
//! crate.
//!
//! # Self-host
//!
//! `cargo-formal` calls OxiZ as its SMT backend, so `oxiz-sat` is part of the
//! verifier's own trusted computing base, and the package sets
//! `self-host = true` in `[package.metadata.formal]`. The report says so.
//!
//! That setting also decides what counts as evidence. `claim-requires`
//! defaults to `lrat, oxilean-verify, external-replay` here, and the LRAT
//! certificates `cargo formal check --evidence lrat` attaches are produced by
//! crates.io `oxiz-sat` 0.3.3 and checked by `oxiz-proof` 0.3.3 -- the
//! published copy of the crate under test and its sibling -- so cargo-formal
//! discounts them: they corroborate nothing for a `self-host` package.
//! `cargo formal replay --all --solver z3` is what does, and `README.md`
//! records what z3 4.15.4 answered.
//!
//! Two versions of one file are involved, and keeping them apart is the whole
//! point: the harnesses verify the `oxiz-sat/src/literal.rs` **of this working
//! tree**, reached by relative path, while the solver that decides them is the
//! crates.io OxiZ pin `=0.3.3`, whose embedded SAT core still carries the
//! **unfixed** copy of that same file. `README.md` states this as a numbered
//! fact, with what it does and does not license.
//!
//! # The three builds
//!
//! This crate is exercised in exactly the three ways `oxiformal`'s own docs
//! describe, and each proves something different:
//!
//! 1. **`cargo build`** (plain, stable). Harnesses vanish entirely (neither
//!    `#[cfg(formal)]` nor `#[cfg(all(test, oxiformal_runtime_checks))]`
//!    applies), so this only checks that the package and `oxiz-sat` type-check
//!    on stable. `cargo test` additionally runs `harness::plain_tests` and
//!    `spec::plain_tests` (13 ordinary tests, measured), which hold a
//!    concrete witness for every `refuted` row plus the boundary facts no
//!    harness reaches -- ordinary Rust tests, no solver involved.
//! 2. **`RUSTFLAGS="--cfg oxiformal_runtime_checks" cargo test`**. Every
//!    harness becomes a `#[test]` that loops [`oxiformal::rt::iterations`]
//!    (256 by default) times through a random draw. A harness whose
//!    counterexample is *dense* under uniform random inputs is additionally
//!    marked `#[should_panic]`, because `oxiformal::rt::run_harness`
//!    propagates any panic that is not an `AssumeViolation`. A harness whose
//!    counterexample is a single narrow point in a huge domain is left
//!    unmarked: it is genuinely `refuted` at L1, but a fair-coin fuzzer
//!    essentially never finds the witness in 256 draws, so the test passes --
//!    which is not evidence of correctness. Exactly one `#[harness]` here is
//!    dense and exactly one is narrow; both are documented as such. A
//!    `#[proof_for_contract]` harness is different: its body runs **once**,
//!    un-wrapped, so a single draw is never a sample. The one whose witness
//!    would be a coin flip (`spec::spec_lit_pos_unbounded_contract`) is
//!    `#[ignore]`d in this build and says why; measured, this build runs
//!    17 tests and ignores 1.
//! 3. **`cargo +nightly-2026-06-20 check` with `--cfg formal
//!    -Zcrate-attr=feature(register_tool) -Zcrate-attr=register_tool(formal_tool)`**
//!    and a separate `--target-dir`, exactly the flags the `cargo-formal`
//!    driver build uses. This only type-checks the `#[cfg(formal)]` copy of
//!    every harness; it does not run the driver or the solver, so it says
//!    nothing about which harnesses are `proved` or `refuted`.
//!
//! The verdicts in [`harness`] and in `EXPECTED.toml` come from a separate,
//! real `cargo formal check` run and are **measurements, not predictions**.
//! `README.md` names the run, the command and the solver version.
//!
//! # Reading a verdict
//!
//! `proved` means the solver could not find an input that violates the
//! property within the harness's bound (and, for a harness with no explicit
//! `assert(..)`, that the code under test never traps for any input the
//! harness allows). `refuted` means a concrete counterexample exists; the doc
//! comment names it and `plain_tests` runs it. `unknown` means the answer is
//! not known -- either the `unwind` bound was hit before the property could be
//! decided, or the solver returned a model that cargo-formal's mandatory model
//! check refused (`solver-model-rejected`; see `README.md`).
//! `unsupported`/`unverifiable` means the encoder could not reduce the harness
//! to a verification condition at all, and the doc comment says which
//! construct stopped it.
//!
//! A `debug_assert!` with no format arguments lowers to
//! `core::panicking::panic`, so its property key is `panic`, not `assert`.
//! Most of this package's interesting rows carry that key -- they are the
//! checks the upstream fix added, and every `refuted` row measured here is
//! one of them. In the 2026-09-15 measurement there are three: the `panic`
//! row of `harness::lit_pos_roundtrip_unbounded_harness` and the `panic` rows
//! of its two contract twins `spec::spec_lit_pos_unbounded_contract` and
//! `spec::spec_lit_from_dimacs_nonzero_contract`. `harness::dimacs_negation_harness`'s
//! `panic` row was `refuted` in the 2026-09-08/09 and 2026-09-14 runs and is
//! `unknown` (`solver-model-rejected`) in this one, on unchanged source and
//! pins; `README.md` and `EXPECTED.toml` explain why that is the model check
//! working rather than a fact changing.

#![forbid(unsafe_code)]

pub mod harness;
pub mod spec;
