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
actually returned for each of them.

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
   `oxiz-sat/src/literal.rs` *of this working tree*, reached by relative path.
   The solver that decides them is crates.io OxiZ `=0.3.3`, whose embedded SAT
   core still carries the **unfixed** copy of that same file. The code under
   test and the copy inside the solver deciding it are the same source file at
   two different versions.
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

## The three builds

All commands are run **from this directory**; `target/` and `Cargo.lock` here
are git-ignored (load-bearing in this repository: the root `.gitignore` anchors
its `Cargo.lock` rule to the root).

```sh
# 1. plain, stable: type-checks the package and runs `harness::plain_tests`
#    (10 ordinary tests, including a concrete run of every counterexample the
#    solver reported).
cargo build
cargo test

# 2. randomized execution: every harness becomes a #[test] that draws random
#    inputs. A harness whose counterexample is dense under uniform draws is
#    additionally marked #[should_panic]; exactly one of this package's two
#    refutations is.
RUSTFLAGS="--cfg oxiformal_runtime_checks" cargo test

# 3. the driver's own type-check of the #[cfg(formal)] copy of each harness.
RUSTFLAGS="--cfg formal -Zcrate-attr=feature(register_tool) -Zcrate-attr=register_tool(formal_tool)" \
  cargo +nightly-2026-06-20 check --target-dir target/formal-check
```

Measured 2026-09-08/09:

| build | result |
|---|---|
| `cargo build` | exit 0, 0 warnings |
| `cargo test` | 10 passed, 0 failed |
| `RUSTFLAGS="--cfg oxiformal_runtime_checks" cargo test` | 9 passed, 0 failed — the one `#[should_panic]` harness observed to panic |
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
  cargo formal check

cargo formal status
```

`cargo formal check` exits non-zero while any obligation is refuted, which is
the intended state here: two of them are refuted on purpose.

## Measured verdicts

**Measured, not predicted.** Every row is from a real `cargo formal check` run
with the release CLI (`cargo-formal` 0.1.0), the release driver with
dependency-body and monomorphic-instance lowering, OxiZ 0.3.3 and rustc
`nightly-2026-06-20`. `EXPECTED.toml` is the machine-readable mirror of this
table, and each harness's doc comment carries the same verdict.

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

18 property rows over 9 harnesses: **15 proved / 2 refuted / 1 unknown**.
Nothing is `unsupported`, `unverifiable` or `timeout`: every harness encodes,
and every obligation that is not decided is not decided by the *solver*.

### Layer counters

Including the incidental MIR-inserted checks the table above does not
enumerate:

| Layer | Counters |
|---|---|
| `hygiene` | PASS — 0 errors, 0 warnings, 0 notes, 2 files scanned, 0 `unsafe` sites |
| `bmc` | **65 proved / 2 refuted / 0 timeout / 3 unknown / 0 unsupported / 0 unverifiable** over 70 obligations, all 70 backed by a `vc/NNNN.smt2` reproduction |
| `contract` | 0 proved / 0 refuted (this package states no `#[requires]`/`#[ensures]` — see the contract candidates below) |
| `theorem` | not run |
| `audit` | `unverifiable` histogram **empty**; 0 trusted items, 0 uncovered `unsafe` |
| coverage | 9 harnesses; `annotated/public` 0.0 % (the package has no public functions of its own; coverage counts *home* functions, and every function under test lives in a dependency) |
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

### `solver-model-rejected`: 3

cargo-formal is pinned to OxiZ **0.3.3**, whose Boolean-structure-over-
bit-vector queries can return a model that does not satisfy the formula
(upstream item U-Z10). cargo-formal runs a **mandatory model check** on every
reported counterexample, so such an answer becomes `unknown` and never a
`refuted` with a fabricated witness.

All three rejections in this run are `assert` sites of
`lbool_negate_involution_harness` — the three negation identities
(`src/harness.rs:330`, `:331`, `:332`). Nothing else in the package is
`unknown`, so the three are also the only rows that can move when the pin
advances.

Two things make this row worth reading rather than skipping:

* **It is not a spelling problem.** Two re-spellings of the same property were
  measured against the same driver: replacing the three biconditionals with an
  `if raw { .. } else { .. }` case split gives 9 obligations of which **4** are
  model-rejected (strictly worse), and hoisting `value.negate()` into a local
  plus adding an explicit `assert(value.is_true() != value.is_false())` gives 9
  obligations with the **same 3** model-rejected. The harness is not hiding an
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

### The two refutations

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
  would forbid.
* **`Lit::from_dimacs(i32::MIN)`** (`:89`). `i32::MIN`'s magnitude is `2^31`,
  one past what a `Lit` can pack; it used to build `Lit::neg(Var(2^31 - 1))`
  without complaint, whose `to_dimacs` computed `(2^31 - 1 + 1) as i32` =
  `i32::MIN` and negated it. The pre-fix twin vendored into cargo-formal as
  `examples/ecosystem/oxiz-sat-lit` measures that as `neg-overflow = refuted`.
  **Here `neg-overflow` is proved in all three DIMACS harnesses**, and the
  refutation has moved to the assertion that now rejects the input. That
  migration — from a silent wrong number, to an overflow, to a checked
  rejection — is the clearest single thing this package measures.

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
instead. These are the contracts this trial would write into
`oxiz-sat/src/literal.rs`, each with the evidence that makes it a real finding
rather than decoration. Every one of them is currently a `debug_assert!`, a doc
sentence, or nothing.

| Where | Contract | Evidence |
|---|---|---|
| `../src/literal.rs:60` `Lit::pos` | `requires(var.0 <= Var::MAX_INDEX)` | measured `panic = refuted`, counterexample `index = 4294967295`; smallest witness `Var::MAX_INDEX + 1`. Load-bearing: above the bound the literal denotes a different variable, silently. The strongest candidate in the file |
| `../src/literal.rs:72` `Lit::neg` | same `requires` | same packing (`(var.0 << 1) \| 1`); its `debug_assert!` (`:73`) is measured `proved` under the bound by harness 2 |
| `../src/literal.rs:35` `Var::new` | same `requires` | the constructor's own `debug_assert!` (`:36`); measured `proved` under the bound by harnesses 1 and 2. It shares one obligation with `Lit::pos`'s, so a contract is what would separate them |
| `../src/literal.rs:88` `Lit::from_dimacs` | `requires(lit != 0 && lit != i32::MIN)` | measured `panic = refuted`, counterexample `i32::MIN`. The pre-fix `debug_assert!` covered only `lit != 0`; the `i32::MIN` half is new, and the vendored twin measures its absence as `neg-overflow = refuted` |
| `../src/literal.rs:110` `Lit::to_dimacs` | `requires(self.var().0 <= Var::MAX_INDEX)`, `ensures(\|r\| *r != 0 && Lit::from_dimacs(*r) == self)` | the round trip is measured `proved` by harness 6; the `requires` is what the documented unconditional panic (`:113`) enforces at run time today, and `plain_tests` runs the panic. A contract would make it an obligation on the *caller* instead |
| `../src/literal.rs:127` `Lit::try_to_dimacs` | `ensures(\|r\| r.is_some() == (self.var().0 <= Var::MAX_INDEX))` | measured `proved` by harness 8, in both arms of the `match` |
| `../src/literal.rs:152` `Lit::negate` | `ensures(\|r\| r.negate() == self && r.var() == self.var() && r.is_pos() != self.is_pos())` | measured `proved` by harness 4 over all `2^32` raw codes |
| `../src/literal.rs:170` `Lit::from_code`, `:164` `code`, `:176` `index` | `ensures` the raw round trip (`code() == c`, `index() == c as usize`, `var().0 == c >> 1`) | measured `proved` by harness 5 |
| `../src/literal.rs:203` `LBool::from_bool`, `:227` `negate` | `ensures(\|r\| r.is_true() == b)` and the involution | the `from_bool` half and the `Undef` fixpoint are measured `proved`; the three involution clauses are the package's only `unknown`, and would become provable with the fixed solver |

## Files

```
formal/
  Cargo.toml      own [workspace]; oxiz-sat (std) + oxiformal by path
  .gitignore      /target, /Cargo.lock  (load-bearing: the root .gitignore
                  anchors its Cargo.lock rule to the repository root)
  README.md       this file
  EXPECTED.toml   the measured verdict table, machine-readable
  src/lib.rs      module docs: self-host, the three builds, how to read a verdict
  src/harness.rs  the harnesses and their plain-build witness tests
```
