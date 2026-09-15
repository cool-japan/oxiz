# `oxiz-sat-formal`

Machine-checked obligations for `oxiz-sat`'s literal encoding, written with
[`cargo-formal`](https://github.com/cool-japan/cargo-formal) and `oxiformal`.

`Lit` is the smallest and most-used value in the whole solver: a variable index
and a sign packed into one `u32` as `index << 1 | sign`, with
`from_dimacs`/`to_dimacs` carrying that packing across the DIMACS boundary. It
is also the value with the least room for a mistake to be noticed — a literal
that silently denotes the *wrong variable* does not crash, it answers. This
package states nine properties of that encoding as `#[harness]` entry points
over the **public** API of `oxiz-sat`, and records the verdict the solver
actually returned for each of them. Since wave D2 of cargo-formal Phase 3 each
of the nine is also stated as a `#[requires]`/`#[ensures]` **contract on a
wrapper function** (`src/spec.rs`, `spec_*`) with a `#[proof_for_contract]`
harness beside it, so the same property is measured once at L1 and once at L2.
The contract is on the wrapper: nothing in `oxiz-sat` is annotated.

It is a standalone package with its own `[workspace]`: it is not a member of
the `oxiz` root workspace, it is never published (`publish = false`), and
nothing outside this directory is touched by building it.

## What is verified

Nothing here is a copy of `oxiz-sat`. The crate is an ordinary path dependency
(`..`), and `cargo-formal` lowers its reachable bodies into the verification
conditions through `[package.metadata.formal] dep-crates`. The run below
lowered **19 dependency bodies** (all 19 reachable) and 0 monomorphic
instances.

| Target | File | Harnesses |
|---|---|---|
| `Var::new`, `Var::MAX_INDEX`, `Lit::pos`/`neg`/`var`/`code`/`from_code`/`index`/`negate`/`is_pos`/`is_neg`/`sign` | `oxiz-sat/src/literal.rs` | 5 |
| `Lit::from_dimacs`, `Lit::to_dimacs`, `Lit::try_to_dimacs` | `oxiz-sat/src/literal.rs` | 3 |
| `LBool::from_bool`/`is_defined`/`is_true`/`is_false`/`negate` | `oxiz-sat/src/literal.rs` | 1 |
| the nine `spec_*` wrappers (`spec_lit_pos`, `spec_lit_neg`, `spec_lit_pos_unbounded`, `spec_lit_negate`, `spec_lit_from_code`, `spec_lit_from_dimacs`, `spec_lit_from_dimacs_nonzero`, `spec_lit_try_to_dimacs`, `spec_lbool_from_bool`), each calling the same API as its `#[harness]` twin | `formal/src/spec.rs` | 9 (`#[proof_for_contract]`) |

`Lit`'s field is private and `literal` is a private module, so everything is
reached through the crate-root re-export (`oxiz-sat/src/lib.rs:227`) — which is
exactly the surface a caller has.

## Self-host

`cargo-formal` calls OxiZ as its SMT backend, and `oxiz-sat` is OxiZ's CDCL
core. So this package verifies part of its own verifier's trusted computing
base, and sets `self-host = true` in `[package.metadata.formal]`; the report
prints `self_host: true`.

Three facts keep that claim honest, and they are worth stating separately:

1. **Two versions of one file are involved.** The harnesses verify
   `oxiz-sat/src/literal.rs` *of this working tree* — `oxiz-sat` 0.3.4,
   unpublished, reached by relative path. The solver that decides them is
   crates.io OxiZ `=0.3.3`: that is the pin `cargo-formal` builds against, it
   is what the report's `pins` field says (`"oxiz": "0.3.3"`) and what every
   item's evidence records, and its embedded SAT core still carries the
   **unfixed** copy of that same file. No 0.3.4 solver decides anything here —
   the code under test and the copy inside the solver deciding it are the same
   source file at two different versions.
2. **That is a circularity of trust, not of logic.** A verification condition
   is a formula; a wrong answer from the solver is a wrong answer whether or
   not the solver's own code is the subject. What the arrangement buys is
   pointed: the two `refuted` rows below are the bugs the working tree fixed,
   and the `unknown` rows below are caused by a *different* bug of the same
   working tree (see the next section). Both are visible from the same run.
3. **It is not a proof of the fixed solver.** Nothing here re-verifies OxiZ
   0.3.3's answers. cargo-formal's mandatory model check is what stands between
   a bad model and a fabricated counterexample, and it is why the affected rows
   below are `unknown` rather than `refuted`.
4. **The certificate is self-produced, and the tool says so.** With
   `--evidence lrat` a proved condition can be bit-blasted by cargo-formal's
   own encoder, solved a second time by crates.io `oxiz-sat` 0.3.3 with LRAT
   tracing, and the proof checked by `oxiz-proof` 0.3.3 — the *published* copy
   of the very crate this package verifies, and its sibling. In the 2026-09-15
   run **15 of the 115 proved conditions** got such a certificate;
   `report.json` records each as `evidence: lrat`, `checker:
   "oxiz-proof 0.3.3"`, `proof: vc/NNNN.lrat`, and `cargo formal replay --all
   --check-lrat` re-verified all 15 against their `.cnf` with both recorded
   SHA-256s matching. Because the package is `self-host`, cargo-formal **does
   not count them**: the run prints `self-host discount: 15 proved item(s)
   carry evidence from this prover's own release train, which corroborates
   nothing here`, and `claim-requires` (which defaults to `lrat,
   oxilean-verify, external-replay` for a self-host package) stays unmet at
   `115 of 115 proved`. What does count is `cargo formal replay --all --solver
   z3`: z3 4.15.4, an independently implemented solver run as a subprocess on
   the byte-identical `.smt2`, **agreed with all 115 proved conditions**
   (`unsat`), agreed weakly with the three refuted ones (`sat`), and decided
   all 12 recorded `unknown`s — 11 `unsat`, 1 `sat`. That moved the report's
   claim row from `115 of 115 proved` unmet to **`34 of 115`**: 81 proved rows
   now carry `external-replay`. The remaining 34 are the ones that carry an
   over-approximation hole (`over-approximated: core::fmt`, from the
   `debug_assert!`/`panic!` message machinery of the DIMACS paths), and an item
   with an undischarged assumption earns no external evidence. That agreement
   says nothing about the encoder — the script is inside the trusted base
   either way — but it is the one piece of evidence here that does not come
   from OxiZ.

## The three builds

All commands are run **from this directory**; `target/` and `Cargo.lock` here
are git-ignored (load-bearing in this repository: the root `.gitignore` anchors
its `Cargo.lock` rule to the root).

```sh
# 1. plain, stable: type-checks the package and runs `harness::plain_tests`
#    (13 ordinary tests over `harness::plain_tests` and `spec::plain_tests`,
#    including a concrete run of every counterexample the solver reported).
cargo build
cargo test

# 2. randomized execution: every harness becomes a #[test] that draws random
#    inputs. A #[harness] whose counterexample is dense under uniform draws is
#    additionally marked #[should_panic]. A #[proof_for_contract] body runs
#    ONCE, un-wrapped, so a single draw is never a sample: the one whose
#    witness would be a coin flip is #[ignore]d and says why.
RUSTFLAGS="--cfg oxiformal_runtime_checks" cargo test

# 3. the driver's own type-check of the #[cfg(formal)] copy of each harness.
RUSTFLAGS="--cfg formal -Zcrate-attr=feature(register_tool) -Zcrate-attr=register_tool(formal_tool)" \
  cargo +nightly-2026-06-20 check --target-dir target/formal-check
```

Measured 2026-09-15, after `src/spec.rs` was added (the 2026-09-08/09 and
2026-09-14 runs are the same table with 10 and 9 tests):

| build | result |
|---|---|
| `cargo build` | exit 0, 0 warnings |
| `cargo test` | 13 passed, 0 failed |
| `RUSTFLAGS="--cfg oxiformal_runtime_checks" cargo test` | 17 passed, 0 failed, **1 ignored** — the one `#[should_panic]` harness observed to panic, and `spec_lit_pos_unbounded_contract` ignored on purpose |
| `RUSTFLAGS="--cfg oxiformal_runtime_checks" cargo clippy --all-targets -- -D warnings` | exit 0 — the only stable build that compiles the contract closures |
| nightly `--cfg formal` check | exit 0, 0 warnings |
| `cargo clippy --all-targets -- -D warnings` | exit 0 |
| `cargo fmt --check` | exit 0 |

`oxiz-sat` is taken with `default-features = false, features = ["std"]`. The
`std` feature is **mandatory, not a choice**: the crate's `no_std` path does not
compile (18 errors, measured by this phase's wave-0 probe on 2026-09-08), which
is a finding about `oxiz-sat` and not about this package.

## Running the verifier

```sh
# `oxiformal` is not on crates.io yet, so both the CLI and the driver come from
# the cargo-formal checkout beside this repository. FORMAL_DRIVER must point at
# the *release* driver binary.
FORMAL_DRIVER=../../../cargo-formal/driver/target/release/formal-driver \
  cargo formal check --evidence lrat    # L1 + L2 verdicts, plus a self-produced LRAT per certifiable proved condition
cargo formal replay --all --solver z3   # the independent evidence: z3 4.15.4 on the same scripts
cargo formal replay --all --check-lrat  # re-check every certificate and its recorded hashes
cargo formal status
```

`cargo formal check` exits non-zero while any obligation is refuted, which is
the intended state here: **three** of them are refuted on purpose — one in
`#[harness]` form (`lit_pos_roundtrip_unbounded_harness`) and two in contract
form (`spec_lit_pos_unbounded_contract`, `spec_lit_from_dimacs_nonzero_contract`).
Both `replay --all` invocations exit 0.

One more cross-check was run and is *not* what `EXPECTED.toml` records
(`autoharness` stays at its default `false`): `cargo formal check --autoharness`
gives every contracted `pub fn` a generated plan of its own, keyed by the
wrapper's path rather than by the hand-written harness's. Of the 43 rows the
nine hand-written `spec_*_contract` harnesses raise, **40 are reproduced verdict
for verdict** by the generated plans. The three that differ are: the two
`Undef` `assert`s, which live in the hand-written harness body and have no
generated counterpart; `spec_lit_from_dimacs#panic`, `proved` under the
generated plan and `solver-model-rejected` under the hand-written one; and
`spec_lit_from_dimacs_nonzero#panic`, `refuted` under the hand-written one and
`solver-model-rejected` under the generated plan. Both differences are on the
same `debug_assert!` at `../src/literal.rs:89` and are the OxiZ 0.3.3 model
weakness moving, not the harness narrowing hiding anything.

## Measured verdicts

**Measured, not predicted.** Every row is from a real `cargo formal check` run
with the release CLI (`cargo-formal` 0.1.0), the release driver with
dependency-body and monomorphic-instance lowering, OxiZ 0.3.3 and rustc
`nightly-2026-06-20`. `EXPECTED.toml` is the machine-readable mirror of this
table, and each harness's doc comment carries the same verdict.

Re-measured 2026-09-14 with rebuilt release binaries on the same pins
(`cargo-formal` 0.1.0, OxiZ 0.3.3, rustc `nightly-2026-06-20`): every verdict
was identical — the same 18 rows, 15 proved / 2 refuted / 1 unknown, the
same two counterexamples, the same layer counters (`bmc` 65 proved / 2 refuted
/ 3 unknown over 70 obligations), `solver-model-rejected: 3`, and the same
`cargo formal check` exit 1.

**Measured 2026-09-15 (cargo-formal Phase 3, wave D2)** with a release CLI and
a release driver built from cargo-formal commit `0621fc0` (0 uncommitted
files), OxiZ 0.3.3, rustc `nightly-2026-06-20`, `--jobs 4`, `--evidence lrat`,
a fresh `--target-dir`, and the run's own `options` row reading `autoharness
off   modular on   vacuity check on`. That run is the one the table below and
`EXPECTED.toml` record. It solved **130 obligations over 18 harnesses** (the
nine `#[harness]` entry points and the nine `spec_*_contract` proof harnesses)
in 27 s and 44 s wall clock from cold target dirs (two runs, on a machine
carrying other work), 8.4 s and 8.6 s of summed per-item solver time, with
**byte-identical verdict tables**.

Two things moved against the 2026-09-14 table, and only one of them is about
this package:

* `dimacs_negation_harness`'s `panic` row is **`unknown`
  (`solver-model-rejected`) instead of `refuted`**, on unchanged source and
  unchanged pins. It reproduced in all four 2026-09-15 runs, two of them taken
  before `src/spec.rs` existed. The witness is still real: z3 4.15.4 decides
  the same script `sat`, `plain_tests` panics on `i32::MIN`, and the contract
  twin `spec_lit_from_dimacs_nonzero_contract` records the refutation with that
  exact counterexample in the same run. OxiZ 0.3.3 returned a model
  cargo-formal's mandatory model check refused, which is the check doing its
  job — see the `solver-model-rejected` section below.
* Nothing else. The `--prev` diff against the pre-`spec.rs` run of the same day
  reports 0 verdict flips on the 70 pre-existing obligations; every other
  difference is an added `spec_*` row or the new `proved vacuously` clause on a
  message.

| # | Harness | Property | Verdict |
|---|---|---|---|
| 1 | `lit_pos_roundtrip_bounded_harness` | `assert` (5 sites) | **proved** |
| 1 | " | `panic` (the two `debug_assert!`s) | **proved** |
| 2 | `lit_neg_roundtrip_bounded_harness` | `assert` (5 sites) | **proved** |
| 2 | " | `panic` | **proved** |
| 3 | `lit_pos_roundtrip_unbounded_harness` | `panic` | **refuted** — `index = 4294967295` |
| 3 | " | `assert` | **proved** |
| 3 | " | `shift-overflow` (2 sites) | **proved** |
| 4 | `lit_negate_involution_harness` | `assert` (4 sites) | **proved** |
| 5 | `lit_code_roundtrip_harness` | `assert` (3 sites) | **proved** |
| 6 | `dimacs_roundtrip_harness` | `assert` (3 sites) | **proved** |
| 6 | " | `panic` (2 sites) | **proved** |
| 6 | " | `neg-overflow` | **proved** |
| 7 | `dimacs_negation_harness` | `panic` (2 sites) | **refuted** — `dimacs = -2147483648` |
| 7 | " | `neg-overflow` | **proved** |
| 8 | `try_to_dimacs_is_total_harness` | `assert` (5 sites) | **proved** |
| 8 | " | `panic` | **proved** |
| 8 | " | `neg-overflow` | **proved** |
| 9 | `lbool_negate_involution_harness` | `assert` (8 sites) | **unknown** — 5 proved, 3 solver-model-rejected |
| 10 | `spec_lit_pos_contract` | `ensures` | **proved** |
| 10 | " | `ensures-2` (the fused packing/polarity clause) | **unknown** — solver-model-rejected |
| 10 | " | `panic`, `shift-overflow` (3 sites) | **proved** |
| 11 | `spec_lit_neg_contract` | `ensures`, `ensures-2`, `ensures-3` | **proved** |
| 11 | " | `panic`, `shift-overflow` (4 sites) | **proved** |
| 12 | `spec_lit_pos_unbounded_contract` | `panic` | **refuted** — `index = 4294967295` |
| 12 | " | `ensures`, `shift-overflow` (2 sites) | **proved** |
| 13 | `spec_lit_negate_contract` | `ensures`…`ensures-4`, `shift-overflow` | **proved** |
| 14 | `spec_lit_from_code_contract` | `ensures`, `shift-overflow` (2 sites) | **proved** |
| 15 | `spec_lit_from_dimacs_contract` | `ensures`, `ensures-2`, `ensures-3`, `neg-overflow` | **proved** |
| 15 | " | `panic` (2 sites) | **unknown** — 1 solver-model-rejected, 1 proved |
| 16 | `spec_lit_from_dimacs_nonzero_contract` | `panic` (2 sites) | **refuted** — `dimacs = -2147483648` |
| 16 | " | `ensures`, `neg-overflow` | **proved** |
| 17 | `spec_lit_try_to_dimacs_contract` | `ensures`, `ensures-2` | **unknown** — both solver-model-rejected |
| 17 | " | `panic`, `neg-overflow` | **proved** |
| 18 | `spec_lbool_from_bool_contract` | `ensures` (the fused `is_defined`/`is_true` clause) | **unknown** — solver-model-rejected |
| 18 | " | `ensures-2` | **proved** |
| 18 | " | `ensures-3`, `ensures-4`, `ensures-5` (the three negation identities) | **unknown** — all solver-model-rejected |
| 18 | " | `assert` (2 sites, the `Undef` facts) | **proved** |

55 property rows over 18 harnesses: **42 proved / 3 refuted / 10 unknown**
(the nine `spec_*` harnesses contribute 37 of them: 27 proved / 2 refuted /
8 unknown). Nothing is `unsupported`, `unverifiable` or `timeout`: every
harness encodes, and every obligation that is not decided is not decided by
the *solver* — z3 4.15.4 decides 11 of the 12 undecided ones `unsat` and the
twelfth `sat`.

### Layer counters

Including the incidental MIR-inserted checks the table above does not
enumerate:

| Layer | Counters |
|---|---|
| `hygiene` | PASS — 0 errors, 0 warnings, 0 notes, 3 files scanned, 0 `unsafe` sites |
| `bmc` | **100 proved / 3 refuted / 0 timeout / 5 unknown / 0 unsupported / 0 unverifiable** |
| `contract` | **15 proved / 0 refuted / 7 unknown** over the 22 `ensures` obligations of the nine `spec_*` wrappers |
| evidence | `reproduction` 130 on every obligation; `lrat` 15 (checker `oxiz-proof 0.3.3`, **discounted** — self-host); `external-replay` 83 (z3 4.15.4) after `replay --all --solver z3` |
| claim | `claim-requires` defaults to `lrat, oxilean-verify, external-replay`; `115 of 115 proved` unmet before replay, **`34 of 115` after** |
| vacuity | `proved vacuously: 11 of 115 checked` — all 11 are panic-*message* sites (`core/src/panic.rs:62:9`, `core/src/macros/mod.rs:290:13`) that no execution reaches, six of them pre-existing `#[harness]` rows; no `ensures` obligation is vacuous |
| holes | 37, every one of them `assume / over-approximated: core::fmt` from the `debug_assert!`/`panic!` message machinery of the DIMACS paths. **No `modular` or `trusted` hole**: no wrapper calls another wrapper, so contract substitution never fires |
| `theorem` | not run |
| `audit` | `unverifiable` histogram: `async: 66` (dependency bodies, not harnesses); 0 trusted items, 0 uncovered `unsafe` |
| coverage | 18 harnesses; `annotated/public` **33.3 %** — the run counts 27 public functions, the nine `spec_*` wrappers plus the 18 `__formal_check_*`/`__formal_replace_*` siblings the contract macro generates, and the 9 annotated ones are the wrappers |
| lowering | `dependency bodies 19 lowered (19 reachable)`; 0 monomorphic instances |

Cost, for the record: the whole run is **about 10 s** wall clock from a cold
`--target-dir` (21.6 s on the first, colder run of the session), of which
`cargo check` with the driver attached — extraction, dependency-body lowering
included — is 9.0 s (19.4 s cold) and the 70 solver calls are the remainder.
The driver logged `lowering dependency bodies from [oxiz_sat] (budget 20000 per
crate)` and lowered 19; **no `FORMAL_DEP_BUDGET` warning appeared**, and at 19
of 20 000 the bound is not close. `oxiz-sat` is a large crate (63 `pub use` lines at its
crate root), so this is the measurement that matters:
dependency-body lowering is driven by reachability from the harnesses, and
nine harnesses over one `u32` type pull in nineteen bodies, not the crate.

The 2026-09-14 re-measurement, also from a cold `--target-dir`, took **21 s**
wall clock, of which the driver-attached `cargo check` is 19.54 s; the 70
solver calls sum to 2.0 s of per-item solver time (the report's `time_ms`,
summed, not elapsed). It lowered the same 19 bodies (19 reachable), and again
**no `FORMAL_DEP_BUDGET` warning appeared**.

The 2026-09-15 wave-D2 run with `--evidence lrat` solves 130 obligations
instead of 70 and additionally asks a vacuity question about 115 of them and
attempts a bit-level certificate for every candidate: **27 s and 44 s** wall
clock from cold target dirs (two runs, on a machine carrying other work), 8.4 s
and 8.6 s of summed per-item solver time, still 19 dependency bodies (19
reachable) and
still no `FORMAL_DEP_BUDGET` warning. `replay --all --solver z3` over all 130
conditions takes 2 s; `replay --all --check-lrat` re-checks the 15
certificates in about the same. Certificates are the exception rather than the
rule here: 115 of the 130 conditions get none, and the run says why on each —
`no certificate: assertion N of this condition folded to the constant 'false',
so the clause set does not describe it`, plus the 34 items whose
`core::fmt` over-approximation hole disqualifies them.

### `solver-model-rejected`: 12

cargo-formal is pinned to OxiZ **0.3.3**, whose Boolean-structure-over-
bit-vector queries can return a model that does not satisfy the formula
(upstream item U-Z10). cargo-formal runs a **mandatory model check** on every
reported counterexample, so such an answer becomes `unknown` and never a
`refuted` with a fabricated witness.

The 2026-09-15 run rejects twelve models, four in `#[harness]` form and eight
in contract form:

| where | what |
|---|---|
| `lbool_negate_involution_harness#assert` ×3 | the three negation identities — the same three as in every earlier run |
| `dimacs_negation_harness#panic` | **new in this run**; this row was `refuted` in 2026-09-08/09 and 2026-09-14 |
| `spec_lbool_from_bool_contract#ensures-3/-4/-5` | the contract form of the same three identities: the contract form does **not** move them |
| `spec_lbool_from_bool_contract#ensures` | the fused `is_defined() && is_true() == raw` clause (both halves are `proved` separately as `assert`s in the harness twin) |
| `spec_lit_try_to_dimacs_contract#ensures`, `#ensures-2` | an `Option<i32>` postcondition — Boolean structure over an enum |
| `spec_lit_pos_contract#ensures-2` | the fused `code == index << 1 && is_pos() && !is_neg() && sign()` clause |
| `spec_lit_from_dimacs_contract#panic` | the `debug_assert!` at `../src/literal.rs:89`, the same site as the `#[harness]` row above it |

Two independent checks say every one of these is a solver limit and not a
property failure. **z3 4.15.4** decides eleven of the twelve `unsat` — i.e.
true — and the twelfth (`dimacs_negation_harness#panic`) `sat`, at the witness
`i32::MIN` that `plain_tests` runs concretely. And on both
`spec_lit_try_to_dimacs_contract` rows cargo-formal's own bit-level engine
produced and verified an LRAT proof, so the report's message reads *the
bit-level engine proved this (`vc/NNNN.lrat`); the SMT engine did not* —
which `--trust-cnf` would promote to `proved`, at the cost of putting
`formal-vcgen::cnf` in the trusted base. This package does not.

Two things make this row worth reading rather than skipping:

* **It is not a spelling problem.** Three re-spellings of the same property
  have now been measured. Two of them were
  measured against the same driver: replacing the three biconditionals with an
  `if raw { .. } else { .. }` case split gives 9 obligations of which **4** are
  model-rejected (strictly worse), and hoisting `value.negate()` into a local
  plus adding an explicit `assert(value.is_true() != value.is_false())` gives 9
  obligations with the **same 3** model-rejected. The third is the contract
  form itself: `spec_lbool_from_bool`'s three `#[ensures]` identities are
  rejected exactly like the three `assert`s. The harness is not hiding an
  encoder gap.
* **The bug that causes it is fixed in this very working tree.** U-Z10's root
  cause (the bit-vector scope rollback in `oxiz-theories/src/bv/solver.rs`) is
  repaired here, as are the two `Lit` bugs the two `refuted` rows below are
  about. cargo-formal keeps its `=0.3.3` pin until the fixes ship in an OxiZ
  release, so this package is measured with the *unfixed* solver against the
  *fixed* code. That is the self-host point in one sentence: the same tree
  contains the code under test, the bug that decided it, and the fix for both.

One harness-writing improvement was found this way and kept.
`try_to_dimacs_is_total_harness`'s `panic` row was `unknown`
(`solver-model-rejected`) until an explicit `assert(dimacs != i32::MIN)` was
added, which gave the solver the bound it was otherwise rediscovering from the
`i32::try_from` inside `try_to_dimacs`. That is a strengthening — the harness
now claims strictly more — and it is why 3 rather than 4 rows are rejected.

### The refutations

Both are the packing invariant, and both are exactly what the upstream fix
added a check for. Before it, neither had any runtime signal at all.

* **`Lit::pos` above the bound** (`../src/literal.rs:61`). `var.0 << 1` drops
  the top bit for an index at or above `2^31` — Rust's `<<` checks the *shift
  amount*, never the value — so `Lit::pos(Var(1 << 31))` used to denote
  **variable 0**, positively, and every clause built from it constrained the
  wrong variable. The measured counterexample is `u32::MAX`; the smallest
  witness is `Var::MAX_INDEX + 1`. Note the split verdict: the round trip
  itself is **proved**, because the encoder continues along the edge where the
  assertion holds. The refutation is precisely the statement "an index above
  the bound can reach this constructor", which is what a `requires` clause
  would forbid. `src/spec.rs` measures both sides of that sentence in one run:
  `spec_lit_pos` carries `requires(index <= Var::MAX_INDEX)` and its `panic`
  row is **proved**; `spec_lit_pos_unbounded` is the same wrapper with the
  `requires` removed and its `panic` row is **refuted** at `index = 4294967295`
  — the same witness, reached the same way.
* **`Lit::from_dimacs(i32::MIN)`** (`:89`). `i32::MIN`'s magnitude is `2^31`,
  one past what a `Lit` can pack; it used to build `Lit::neg(Var(2^31 - 1))`
  without complaint, whose `to_dimacs` computed `(2^31 - 1 + 1) as i32` =
  `i32::MIN` and negated it. The pre-fix twin vendored into cargo-formal as
  `examples/ecosystem/oxiz-sat-lit` measures that as `neg-overflow = refuted`.
  **Here `neg-overflow` is proved in every DIMACS harness**, and the
  refutation has moved to the assertion that now rejects the input. That
  migration — from a silent wrong number, to an overflow, to a checked
  rejection — is the clearest single thing this package measures. In the
  2026-09-15 run that refutation is recorded by the **contract** form,
  `spec_lit_from_dimacs_nonzero_contract#panic` (counterexample
  `dimacs = -2147483648`); the `#[harness]` form of the identical obligation
  came back `solver-model-rejected` that day, which is the one row this package
  has ever seen move without its source moving.

`Lit::to_dimacs`'s own unconditional panic (`:113`) is **proved** wherever a
harness reaches it, because no public *constructor* can produce a literal above
the bound any more. It is reachable only through `Lit::from_code`, and
`plain_tests::to_dimacs_panics_for_a_literal_above_the_bound` states it
concretely. `Lit::try_to_dimacs` is the total alternative, and
`try_to_dimacs_is_total_harness` proves it is `Some` exactly on the codes whose
index is within the bound, with the round trip holding there.

## In-source contract candidates

The honest end state is `#[oxiformal::requires]` / `#[ensures]` on the real
`oxiz-sat` functions, with the proof harnesses beside them. That needs
`oxiformal` on crates.io, so this package states the properties from outside
instead. Wave D2 moved one step closer without crossing that line: the same
properties are stated as contracts on **wrapper functions in `src/spec.rs`**
— the `spec_*` twin named in each row below — and measured with
`--evidence lrat` and `replay --all --solver z3`. The contract is on the
wrapper. Nothing in `oxiz-sat/src/literal.rs` is annotated, so the table below
is still the list of what would move *into* that file once `oxiformal` is
published. These are the contracts this trial would write into
`oxiz-sat/src/literal.rs`, each with the evidence that makes it a real finding
rather than decoration. Every one of them is currently a `debug_assert!`, a doc
sentence, or nothing.

| Where | Contract | Evidence |
|---|---|---|
| `../src/literal.rs:60` `Lit::pos` (`spec_lit_pos`, `spec_lit_pos_unbounded`) | `requires(var.0 <= Var::MAX_INDEX)` | measured `panic = refuted`, counterexample `index = 4294967295`; smallest witness `Var::MAX_INDEX + 1`. Load-bearing: above the bound the literal denotes a different variable, silently. The strongest candidate in the file |
| `../src/literal.rs:72` `Lit::neg` (`spec_lit_neg`) | same `requires` | same packing (`(var.0 << 1) \| 1`); its `debug_assert!` (`:73`) is measured `proved` under the bound by harness 2 |
| `../src/literal.rs:35` `Var::new` (reached through `spec_lit_pos`/`spec_lit_neg`) | same `requires` | the constructor's own `debug_assert!` (`:36`); measured `proved` under the bound by harnesses 1 and 2. It shares one obligation with `Lit::pos`'s, so a contract is what would separate them |
| `../src/literal.rs:88` `Lit::from_dimacs` (`spec_lit_from_dimacs`, `spec_lit_from_dimacs_nonzero`) | `requires(lit != 0 && lit != i32::MIN)` | measured `panic = refuted`, counterexample `i32::MIN`. The pre-fix `debug_assert!` covered only `lit != 0`; the `i32::MIN` half is new, and the vendored twin measures its absence as `neg-overflow = refuted` |
| `../src/literal.rs:110` `Lit::to_dimacs` (reached through both DIMACS wrappers) | `requires(self.var().0 <= Var::MAX_INDEX)`, `ensures(\|r\| *r != 0 && Lit::from_dimacs(*r) == self)` | the round trip is measured `proved` by harness 6; the `requires` is what the documented unconditional panic (`:113`) enforces at run time today, and `plain_tests` runs the panic. A contract would make it an obligation on the *caller* instead |
| `../src/literal.rs:127` `Lit::try_to_dimacs` (`spec_lit_try_to_dimacs`) | `ensures(\|r\| r.is_some() == (self.var().0 <= Var::MAX_INDEX))` | measured `proved` by harness 8, in both arms of the `match` |
| `../src/literal.rs:152` `Lit::negate` (`spec_lit_negate`) | `ensures(\|r\| r.negate() == self && r.var() == self.var() && r.is_pos() != self.is_pos())` | measured `proved` by harness 4 over all `2^32` raw codes |
| `../src/literal.rs:170` `Lit::from_code`, `:164` `code`, `:176` `index` (`spec_lit_from_code`) | `ensures` the raw round trip (`code() == c`, `index() == c as usize`, `var().0 == c >> 1`) | measured `proved` by harness 5 |
| `../src/literal.rs:203` `LBool::from_bool`, `:227` `negate` (`spec_lbool_from_bool`) | `ensures(\|r\| r.is_true() == b)` and the involution | the `from_bool` half and the `Undef` fixpoint are measured `proved`; the three involution clauses are `unknown` in both the `#[harness]` and the contract form, and would become provable with the fixed solver |

## Files

```
formal/
  Cargo.toml      own [workspace]; oxiz-sat (std) + oxiformal by path
  .gitignore      /target, /Cargo.lock  (load-bearing: the root .gitignore
                  anchors its Cargo.lock rule to the repository root)
  README.md       this file
  EXPECTED.toml   the measured verdict table, machine-readable
  src/lib.rs      module docs: self-host, the three builds, how to read a verdict
  src/harness.rs  the nine #[harness] entry points and their plain-build tests
  src/spec.rs     the nine spec_* contracted wrappers and their
                  #[proof_for_contract] harnesses
```

## A driver limitation this package measured

Four of the `#[ensures]` clauses in `src/spec.rs` are spelled as conjunctions,
or state their identity against `LBool::from_bool(raw)` rather than against the
returned value twice, and both spellings are forced rather than chosen. An
`#[ensures]` closure that captures **nothing** is a zero-sized type, so its MIR
operand is a *constant* of closure type, and the driver's constant decoder
(`driver/formal-driver/src/lower/konst.rs:134-137`) has no arm for it: it
refuses with `unsupported-rvalue(constant of type '{closure@…}')`, the whole
generated `__formal_check_*` body is lost, and the wrapper is reported
`unsupported(no-body)` — no verdict at all.

That was measured, not inferred. The first `--evidence lrat` run of `src/spec.rs`
(2026-09-15, artefacts kept) reported exactly the three wrappers that had a
capture-less clause — `spec_lit_pos`, `spec_lit_neg`, `spec_lbool_from_bool` —
as `unsupported(no-body)`, and the six whose every clause mentions a parameter
encoded. Re-spelling the four capture-less clauses so each mentions its
wrapper's parameter is what made all nine measurable. It is a change of
spelling and not of claim, and the price is visible in the table above: the two
fused clauses (`spec_lit_pos#ensures-2`, `spec_lbool_from_bool#ensures`) are
`solver-model-rejected`, while their halves are `proved` separately as
`assert`s in the `#[harness]` twins. The limitation is in `cargo-formal`, not
in `oxiz-sat`, and is reported there.
