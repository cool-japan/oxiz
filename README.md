# OxiZ

Next-Generation SMT Solver in Pure Rust

[![Crates.io](https://img.shields.io/crates/v/oxiz.svg)](https://crates.io/crates/oxiz)
[![Documentation](https://docs.rs/oxiz/badge.svg)](https://docs.rs/oxiz)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## About This Project

OxiZ is a high-performance Satisfiability Modulo Theories (SMT) solver written entirely in Rust. This project reimplements [Z3](https://github.com/Z3Prover/z3) in Pure Rust with a focus on correctness, performance, and safety.

**Pure Rust is a fundamental requirement** - no C/C++ dependencies, no FFI bindings, just clean, safe Rust code.

### Implementation Status (v0.3.2)

OxiZ is under active development with core theories at production quality on its tested surface.

- **Pure Rust Implementation**: 451,853 lines of production Rust code across 1,276 files (564,303 total including comments/blank lines, per `tokei . --exclude target`)
- **Unit Tests**: 9,953 passing, 8 skipped (`cargo nextest run --workspace --all-features`, confirmed at release time), plus 110 doc-tests (`cargo test --doc --workspace --all-features`)
- **Z3 Parity**: honest, non-fabricated comparison against a real `z3` 4.15.4 binary (`bench/z3_parity`): **170/170 Correct, 0 Wrong, 0 Inconclusive, 0 Timeout, 0 Error** on the extended 19-logic / 170-benchmark differential suite under the honest comparator (an `Unknown` from either solver never counts as a match). All 19 logic families are individually at 100%, up from 154/168 with 12 Inconclusive and 2 Timeout at 0.3.0. This is 100% *of the differential parity suite* — it is not a blanket "100% Z3 compatibility" claim about the solver as a whole; see "Z3 Parity" below and [`TODO.md`](TODO.md) for what the suite does and does not cover
- **Audit + hardening (multiple waves)**: a 2026-07-16 production-readiness audit (19 scoped agents + adversarial verification), the 0.3.0 follow-on waves, the 0.3.1 soundness sweep, and this release's issue-[#25](https://github.com/cool-japan/oxiz/issues/25)-driven soundness sweep found and fixed soundness and honesty gaps across every crate — SMT-LIB parser coverage (now fully iterative), quantifier elimination (Ferrante-Rackoff, virtual substitution, MBI/Craig interpolants), MBQI SAT certification and completeness, Spacer MIC generalization and multi-threaded parallel PDR, IEEE-754 `fp.rem`/fused-multiply-add, proof-rule/checker validation, >64-bit bitvector arithmetic, process-crash fixes, and — this release — EUF congruence closure, Bool/EUF encoding and Arithmetic⇄EUF combination false-`sat` families plus NLSAT conflict-analysis polarity bugs. Re-verified status (which items are fixed vs. still open) is tracked per-item in [`TODO.md`](TODO.md); a small number of NLSAT items (notably irrational-root isolation) remain open and are called out there and in the [CHANGELOG](CHANGELOG.md#032---2026-08-05)

## What's New in 0.3.3 (2026-08-26)

A soundness release. Fifteen SMT-LIB benchmarks carrying `(set-info :status unsat)` — from GitHub issues [#44](https://github.com/cool-japan/oxiz/issues/44)-[#50](https://github.com/cool-japan/oxiz/issues/50) — all answered `sat` on 0.3.2, the worst class of SMT bug; none of them does now. Alongside that, three of this project's standing honest rejections became real support: recursive functions, a first-class `RoundingMode` sort, and algebraic-number witnesses for irrational roots. Full itemized detail is in [`CHANGELOG.md`](CHANGELOG.md#033---2026-08-26); highlights:

### ⚠️ Breaking change (0.x API)
`Command` gained a `DefineFunsRec` variant and `TermManager::mk_bv_concat` was replaced by a fallible `try_mk_bv_concat` (the old infallible version could fabricate a wrong bit-width on error, silently flipping a verdict). `SortKind`/`Value` gained `RoundingMode` variants, so `SortManager` now interns four built-in sorts instead of three — user sorts now start at raw id 4, not 3. `oxiz_core::theories::combination::Theory` was rewritten into an implementable trait, `NlDispatchResult::Sat` changed shape again (now carries either a rational witness or an algebraic one, never both), and `SolverConfig` grew six new public fields across `oxiz-sat`/`oxiz-solver`. All plain structs/enums, not `#[non_exhaustive]` — downstream code matching exhaustively or constructing via full struct-literal syntax needs updating. See CHANGELOG.md for the complete list.

### Soundness fixes
The headline: fifteen SMT-LIB benchmarks that should never have answered `sat` (`storecomm_*`, `ring_2exp*`, `xs-*`, a QF_ANIA case, `jobshop4-2-2-2-4-4-11`, `vhard7`) traced to eight independent root causes — a `let` wrapper that blinded every assert-time soundness pre-pass, a numeric disequality that constrained nothing once forced false, a violated disequality invisible to the model-refutation gate, lazy theory mode discarding assignments the SAT core hadn't undone, a `store`-only formula that never instantiated an array axiom, an exhausted array-axiom budget reported as full satisfaction, a verdict that could depend on machine load, and an integer term bounded only by a difference of two free variables (new `int_range_lp.rs`). Several of the fifteen (`xs_8_13`, `vhard7`, two of the `ring_2exp*` files, and `storecomm_t1_pp_nf_ni_00050_001`) are now honest timeouts rather than wrong answers — the search-performance work they need is deliberately deferred. Also fixed: a SAT-engine bug that could silently drop a level-0 implication after clause strengthening, a substitution path that could fabricate an ill-sorted `concat`, and a recursive-function retraction bug that could retract the wrong scope's applications.

### New capabilities
Recursive functions (`define-fun-rec`/`define-funs-rec`) are supported end to end, by fuel-bounded unfolding with a saturation-or-model-recomputation certificate for `sat`. `RoundingMode` is a real five-valued sort with cardinality axioms and quantifier-binder relativization, not the free uninterpreted sort it silently fell through to before 0.3.0's parse rejection. Algebraic-number witnesses mean an irrational root like `x² = 2` is now a model (`root-obj`), not an `unknown`. A new NIA-over-LP nonlinear-integer relaxation engine converts goals the cell-decomposition core could only call `unknown` on into decided `unsat`. Gomory mixed-integer cuts, real since they were written, are wired in and reachable for the first time. The SAT engine gained a pre-search "lucky" phase ([#35](https://github.com/cool-japan/oxiz/issues/35)) and bounded variable elimination is finally reachable ([#36](https://github.com/cool-japan/oxiz/issues/36)), alongside a working self-subsuming resolution pass.

See [CHANGELOG.md](CHANGELOG.md#033---2026-08-26) for the rest (the new `oxiz-time` wasm-clock crate, `rhai`/`oxiz-nlsat` becoming optional Cargo features, and the full soundness-fix root-cause writeups).

## What's New in 0.3.2 (2026-08-05)

A soundness release driven by an external report: GitHub issue [#25](https://github.com/cool-japan/oxiz/issues/25) ran a random 50-instance QF_UF differential sample against a real z3 binary and found a 34% (17/50) wrong-answer rate. Per project policy none of the 8 pull requests the reporter subsequently opened (#26-#33) were merged directly — this project has no CLA or contribution-provenance guarantee yet — but every one was read in full for its diagnostic value, and every fix below is an independent, from-scratch reimplementation done in-house, backed by a regression test derived from a minimal repro. ~270 such tests were added in the process. Full itemized detail is in [`CHANGELOG.md`](CHANGELOG.md#032---2026-08-05); highlights:

### ⚠️ Breaking change (0.x API)
`NlDispatchResult::Sat` gained a payload (`Sat` → `Sat(Box<Interpretation>)`), so the nonlinear (QF_NIA/QF_NRA) dispatcher's `sat` verdict now carries the witness model it was checked against; and `SolverConfig` grew thirteen new public fields (`use_vmtf`, `enable_bve`, `nonlinear_model_search`, and others — all defaulting conservatively). Both are plain structs, so downstream code matching `NlDispatchResult::Sat` or constructing `SolverConfig` via full struct-literal syntax needs updating. This is internal/library-API breakage, not a CLI or SMT-LIB behavior change.

### Soundness fixes
The headline fixes:
- **EUF congruence closure** could merge two nodes that were no longer actually congruent, because a re-canonicalized node's signature was republished without evicting its old signature-table entry — a spurious equality able to turn a `sat` formula `unsat`.
- **Bool/EUF encoding** had three gaps that together produced a false-`sat` family (internally tracked as `firewire_tree`): a non-Bool `ite` was interned as an opaque EUF leaf with no link to its conditional-equality meaning; a Bool-sorted `(= b1 b2)` was Tseitin-encoded but never registered as an EUF constraint; and a Bool term used as a UF argument was never completed for congruence.
- **Arithmetic⇄EUF combination**, the headline fix: when arithmetic facts entailed an equality (e.g. `x=2, y=x+1 ⊢ y=3`) but that equality never reached EUF, terms like `f(y)` and `f(3)` were treated as possibly distinct even though `y` and `3` provably aren't — so `f(y)≠f(3)` alongside those constraints reported `sat` when it should have been `unsat`. This was the headline false-`sat` family on QF_UFLIA/QF_UFIDL, fixed by purifying numeric UF arguments into arithmetic interface terms, adding cross-theory entailment probes backed by Farkas certificates, and case-splitting the remaining non-convex gap over small shared integer domains.
- **NLSAT conflict analysis** had a wrong-polarity 1-UIP resolution bug that could produce a learned clause not actually entailed by the problem, able to refute a genuinely satisfiable formula; and a level-0 `GreedyEmpty` arithmetic decision misread as global infeasibility (a satisfiable `x·y=35` reported `Unsat`), now fixed with a witness ledger that retries a different point from the same region instead of conceding defeat.
- **`distinct` over constants** could invert: `(assert (distinct 5 2))`, which constant-folds to `False`, reported `unsat` instead of `sat` because the Tseitin encoder's `False` arm returned the wrong-polarity literal.

See [CHANGELOG.md](CHANGELOG.md#032---2026-08-05) for the rest (`define-fun` argument substitution, DIMACS empty-clause handling, two SAT preprocessing gaps, and a quantified-model verification backstop).

### New capability
A CaDiCaL-style SAT search loop ships on by default: VMTF branching is now actually wired into decisions (previously dead code) alongside VSIDS/CHB/LRB, with focused/stable mode alternation, restart-trail reuse, and phase-based rephasing. An opt-in SAT inprocessing toolkit (failed-literal probing, bounded variable elimination, equivalent-literal substitution, gate-congruence closure) ships alongside online LRAT proof production and a new pure-Rust LRAT checker. QF_NIA/QF_NRA nonlinear solving is on by default: an exact `BigRational` evaluator double-checks every candidate model, with stochastic model-repair search and array/UF grammar reduction as fallbacks when the core dispatcher can't decide.

Issue #25 raised a broader concern than the specific bugs above: plain QF_UF still isn't part of the 170-benchmark parity suite, so that gap — and the harness work needed to close it — is tracked in TODO.md rather than in the issue tracker (see "Z3 Parity" below).

## What's New in 0.3.1 (2026-07-31)

A soundness-sweep, hardening and quantifier-completeness release on top of 0.3.0. Full itemized detail is in [`CHANGELOG.md`](CHANGELOG.md#031---2026-07-31); highlights:

### ⚠️ Breaking change (0.x API)
`ModelValue::BitVec` now carries a `num_bigint::BigUint` payload instead of a `u64`, so bit-vector model values wider than 64 bits are represented exactly rather than truncated. Code that pattern-matched the old `u64` payload must be updated. New helpers accompany it: `Model::assign_bitvec_big`, `ModelValue::from_bitvec_int`, `ModelValue::from_bitvec_bits`, `ModelValue::as_bitvec`, and `bitvec_mask`.

### Soundness sweep
GitHub issues [#12](https://github.com/cool-japan/oxiz/issues/12), [#14](https://github.com/cool-japan/oxiz/issues/14), [#17](https://github.com/cool-japan/oxiz/issues/17), [#18](https://github.com/cool-japan/oxiz/issues/18) and [#23](https://github.com/cool-japan/oxiz/issues/23) are closed, and auditing their common shape — *unhandled input silently dropped or defaulted instead of raising an error* — turned up 40+ further instances of the same bug across the workspace, all fixed. The SMT-LIB parser's recursion was made fully iterative in the process.

Separately verified and fixed: three >64-bit bit-vector soundness bugs (`assert_const` truncating the low limb, which produced a false `Sat`; a canonical interning key that used only the low 64 bits and therefore merged distinct wide constants, producing a false `Unsat`; and a model builder that zero-filled wide values); e-matching substitution that left quantified variables *free* in generated bit-vector/string lemmas; a theory-solver scope leak across MBQI rounds (a false `Unsat` on satisfiable re-checks) now closed by `rebase_theory_state`; a 4-way exponential expansion in Cooper's `Xor`/`Ite` handling; and the parser now **rejects** mixed-width bit-vector binary operands at parse time, as Z3 does, instead of accepting them.

### Hardening (~400 sites)
Every remaining unguarded recursive term walk — parsers, printers, the model evaluator, substitution, and the `Drop`/`Clone`/`PartialEq` impls on deep public enums — was converted to an explicit heap stack, so deeply nested input can no longer overflow the native stack. Tier-1 silent fallthroughs became exhaustive matches or honest errors; encode-depth memoization removed a 2^n DAG re-encoding blowup and `ENCODE_DEPTH_LIMIT` was measured and set to 512; string handling is now non-ASCII-safe; `oxiz-math` gained a real multivariate polynomial GCD (primitive PRS with pseudo-division) in place of stubs; Tarjan's SCC is iterative; and `powi(i32::MIN)` no longer recurses forever.

### MBQI completeness — the parity push
Finite-range quantifier expansion (`AUFLIA`), Skolem witness synthesis with CEGAR refinement (`UFLIA`), and symbolic model certification over the reals plus quasi-macro detection (`UFLRA`) close the three quantified logics that were the whole of the remaining parity gap. The three benchmarks that previously hit the 60s timeout now solve in roughly a millisecond, and the suite went **154/168 → 168/168 Correct**, all 19 logic families at 100%, verified over three consecutive full runs on an idle machine plus a fourth after the resource work below.

### Honesty & state hygiene
Models, unsat cores and proofs are invalidated on `push`/`pop`/`assert`, so a stale answer can never be handed back; an unjustified conflict clause now yields `Unknown` instead of a fabricated `Unsat`; derived reasons are Solver-owned and stamped with absolute scope depth; e-matching trigger inference is restricted to uninterpreted heads (matching Z3's `pattern_inference`, and fixing a matching loop); and `(get-unsat-core)` works when `:produce-unsat-cores` is enabled mid-session, because assertion names are now recorded unconditionally.

### Repeated `(check-sat)` behavior
Long incremental sessions no longer accumulate cost. Hyper-binary-resolution clauses are registered in the learned/assertion ledgers, so clause-DB reduction, `forget` and `pop` can actually reclaim them; `Solver::pop` retracts Tseitin memo entries per-entry through the undo journal instead of clearing the memo wholesale (the wholesale clear caused unbounded re-encoding — one goal grew from 25 to 361 original clauses over 30 push/pop-plus-check cycles); MBQI search state is checkpointed and restored around each check, which also fixes MBQI silently ceasing to instantiate after roughly ten checks on the same goal. New: a **verdict cache** makes a repeated `(check-sat)` on an untouched goal an O(1) cache hit, invalidated by `assert`/`push`/`pop`/`reset` and by every settings mutator.

### Quality gates
`clippy::unwrap_used` is denied in all 17 member crates, and clippy is clean in both the dev and release profiles; `rustdoc -D warnings` is clean; `cargo deny check bans` is clean; every source file is under the 2,000-line cap. `to_cnf_tseitin` (an equisatisfiable, linear-size CNF encoding) was added to `oxiz-core` and `TseitinCnfTactic` rewired to it.

## What's New in 0.3.0 (2026-07-22)

A large hardening-and-capability wave on top of 0.2.4's audit. Full itemized detail is in [`CHANGELOG.md`](CHANGELOG.md#030---2026-07-22); highlights:

### Soundness fixes
Exact `BigRational` floor/ceil and real polynomial GCD/resultant in `oxiz-math`; `QF_NIRA` variable-type dispatch and incomplete-atom tracking in `oxiz-nlsat`; real Ferrante-Rackoff and Loos-Weispfenning virtual-substitution quantifier elimination for LRA; a real propositional Craig interpolant for MBI; IEEE-754-correct `fp.rem` and single-rounded fused multiply-add (plus a separate `div128` remainder-overflow bug in the `ieee754_full` engine that returned `0.0` for divisions with a true quotient in `(0.5,1)`); an `ArithSolver::pop()` state-rollback bug where `term_to_var` was not rolled back in lockstep with `var_to_term` (recycled `VarId`s could attach a constraint to the wrong variable) plus a `lia_model` backtrack-clear fix; a simplex `register_var`/`ensure_var` hardening pass that closed both an index-out-of-bounds crash and a silent-drop of out-of-range bound-setting calls; several proof-checker rubber-stamps (resolution, parallel proof checking, per-theory checkers) replaced with real verification.

### New capabilities
`(get-consequences ...)` and `:named` assertions; a regex sublanguage for the string theory plus a new ground string decision procedure (definitional propagation, concat-splitting by known lengths, and regex-intersection search via the Brzozowski automata engine, gated by full concrete verification before returning `Sat`) that lifts `qf_s` from 3/10 to 10/10 on the parity suite; a sound concrete floating-point model finder that lifts `qf_fp` from 1/10 to 10/10; MBQI SAT certification (lifts `UFLIA`/`UFLRA`/`AUFLIA` from mostly-`Unknown` toward a majority-decisive verdict — see "Z3 Parity" below); real datatype and bitvector quantifier elimination; Spacer MIC generalization plus a genuine multi-threaded parallel PDR portfolio; a McMillan-style Craig interpolation system; `oxiz-ml` is now genuinely wired into `oxiz-cli` via the opt-in `--ml-tactic-selection` flag (off by default — see "Features" below); `oxiz-wasm` gains hard-preemptible solving via a dedicated Web Worker (`PreemptibleSolver`), `SharedArrayBuffer`-backed cooperative cancellation (`CancellationToken`), and a Rust-generated TypeScript `.d.ts` — see "WebAssembly" below.

### Honesty & robustness
Two process-crash panics fixed (a SAT theory-conflict-with-unassigned-literal panic; a simplex out-of-bounds panic); both SMT-LIB parser regressions surfaced by 0.2.4's stricter undeclared-symbol check (`to_fp` rounding-mode arguments, `re.allchar`) are closed; roughly a dozen previously-dead-but-tested `oxiz-nlsat` modules (subsumption, inprocessing, vivification, structure analysis, …) are now wired into the real solve loop; confirmed-dead GPU-flag scaffolding (zero references anywhere in the workspace) was deleted rather than left as a misleading placeholder.

### Z3 Parity gains
Measured at the 0.3.0 release: **154/168 Correct** (up from 122/168 at the 0.2.4 baseline), **0 Wrong**, **0 process crashes** (down from 3, then reported as `Inconclusive`/`Timeout`), 12 Inconclusive, 2 Timeout. `qf_fp` 1/10→10/10, `qf_s` 3/10→10/10 (both via new decision procedures, see "New capabilities"), `AUFLIA` 2/10→7/10, `UFLIA` 7/20→14/20, `UFLRA` 2/10→5/10 — leaving those three quantified logics as the entire remaining gap, which 0.3.1's MBQI completeness work then closed. See "Z3 Parity" below for the current breakdown.

For the 0.2.4 production-readiness audit and the 0.2.3 feature set (generic `DratWriter`/`LratWriter` proof writers, NLSAT root-isolation completions, the full `oxiz-opt` optimization pipeline, real BMC/k-induction in `oxiz-spacer`), see the [0.2.4](CHANGELOG.md#024---2026-07-19) and [0.2.3](CHANGELOG.md#023---2026-06-09) CHANGELOG entries.

## Theory Support Status

Numbers below are re-measured at release time against the current `bench/z3_parity` suite (a real `z3` 4.15.4 binary, an honest comparator that never counts `Unknown` as a match — see [`bench/z3_parity/src/comparator.rs`](bench/z3_parity/src/comparator.rs)). As of 0.3.2 every one of the 19 logic families in the suite is at 100% Correct.

### Core Logics at 100% Z3 Parity ✅

#### Arithmetic Theories
- **QF_LIA** (Linear Integer Arithmetic) - **100.0%** (16/16 tests)
  - Simplex with GCD-based infeasibility detection
  - Branch-and-bound for integer solutions
  - Cutting plane generation
- **QF_LRA** (Linear Real Arithmetic) - **100.0%** (16/16 tests)
  - Tableau-based simplex solver
  - Efficient pivot selection
  - Incremental constraint management
- **QF_NIA** (Nonlinear Integer Arithmetic) - **100.0%** (1/1 test)
  - NLSAT solver with CAD
  - Branch-and-bound for integers
  - Complete theory integration

#### Bit-Vector Theory
- **QF_BV** (Bit-Vectors) - **100.0%** (15/15 tests)
  - Bit-blasting with word-level reasoning
  - Constraint propagation for arithmetic ops
  - Signed/unsigned division and remainder (double-width, non-wrapping encoding)
  - Logical operations (NOT, XOR, OR, AND), barrel-shifter over-shift handling
  - Comparison conflict detection

#### Datatype Theory
- **QF_DT** (Datatypes) - **100.0%** (10/10 tests)
  - Constructor exclusivity enforcement
  - Tester predicate evaluation
  - Selector function semantics
  - Cross-variable constraint propagation
  - Enumeration type handling

#### Array Theory
- **QF_A** (Arrays) - **100.0%** (10/10 tests)
  - Read-over-write axioms
  - Extensionality reasoning
  - Store propagation (101 unit tests)

#### String Theory
- **QF_S** (Strings) - **100.0%** (10/10 tests) — at 100% since 0.3.0's ground string decision procedure
  - The ground string decision procedure (definitional propagation, concat-splitting by known operand lengths, and per-variable regular-constraint intersection search via the existing Brzozowski derivative automaton engine) synthesizes a candidate model and only returns `Sat` after concretely verifying it against every assertion — it never guesses, so the previous honest `Unknown` gate (`string_atoms_need_theory`) is bypassed only on a verified witness. Unsatisfiable cases are still caught by the pre-existing definite-conflict detector before this path runs.

#### Floating-Point Theory
- **QF_FP** (Floating Point) - **100.0%** (12/12 tests) — at 100% since 0.3.0's concrete FP model finder
  - A sound concrete FP model finder pins every FP-sorted term to a bit-exact IEEE-754 value (definitional-equality propagation plus predicate-driven witness synthesis for free NaN/Infinity-typed variables) and reports `Sat` only after verifying every assertion; directed-rounding (`RTP`/`RTN`/`RTZ`) division is bit-exact since 0.3.0 fixed a `div128` remainder-overflow bug in the `ieee754_full` engine (it dropped bit 127 whenever a division's true quotient fell in `(0.5,1)`, so e.g. `10/3` evaluated to `0.0`).

### Additional Logics (Extended Suite, 19 Logics / 170 Benchmarks Total)

`bench/z3_parity/benchmarks/` also covers `AUFLIA`, `AUFLIRA`, `QF_ABV`, `QF_ALIA`, `QF_AUFBV`, `QF_AUFLIA`, `QF_NIRA`, `QF_UFLIA`, `QF_UFLRA`, and `UFLIA`/`UFLRA` (quantified logics). Aggregate result across all 170 benchmarks: **170 Correct, 0 Wrong, 0 Inconclusive, 0 Timeout, 0 Error** — every benchmark decisive, and every decisive answer agreeing with z3. Per-logic Correct counts for this group: `AUFLIA` 10/10, `AUFLIRA` 5/5, `QF_ABV` 5/5, `QF_ALIA` 5/5, `QF_AUFBV` 5/5, `QF_AUFLIA` 5/5, `QF_NIRA` 5/5, `QF_UFLIA` 5/5, `QF_UFLRA` 5/5, `UFLIA` 20/20, `UFLRA` 10/10.

The three quantified logics that carried the whole of 0.3.0's remaining gap closed at 0.3.1 via the MBQI completeness work (`AUFLIA` 7/10 → 10/10, `UFLIA` 14/20 → 20/20, `UFLRA` 5/10 → 10/10); the three benchmarks that previously exhausted the 60s budget now solve in about a millisecond. This is 100% of the differential parity suite, not a blanket claim of Z3 compatibility outside it.

- **QF_UF** (Uninterpreted Functions) - E-graphs with congruence closure (not separately benchmarked; exercised indirectly by every other logic above)
- **QF_NRA** (Nonlinear Real) - CAD-based NLSAT solver (Alpha: irrational-root isolation still open; not part of this benchmark suite, see `TODO.md`)
- **AUFBV** (Arrays + UF + BV) - Theory combination via Nelson-Oppen (`QF_AUFBV` scores 5/5 on the parity suite)
- **UFLIA** (Quantified LIA) - 20/20 since 0.3.1: Skolem witness synthesis with CEGAR refinement decides the cases that previously fell back to `Unknown` or timed out
- **UFLRA** (Quantified LRA) - 10/10 via symbolic model certification over the reals plus quasi-macro detection
- **AUFLIA** (Quantified LIA + Arrays) - 10/10 via finite-range quantifier expansion
- **HORN** (Horn Clauses) - PDR/IC3 engine with MIC generalization and a multi-threaded parallel portfolio; real BMC, k-induction, and init/transition SMT queries (not part of this benchmark suite)

## Features

- **Pure Rust** - No C/C++ dependencies, memory-safe by design
- **CDCL(T) Architecture** - Conflict-Driven Clause Learning with theory integration
- **Comprehensive Theory Support** - EUF, LRA, LIA, BV, Arrays, Strings, FP, Datatypes
- **Advanced Quantifier Handling** - MBQI, E-matching, Skolemization, DER
- **SMT-LIB2 Support** - Full standard input/output format
- **WebAssembly Ready** - Run in browsers via WASM bindings
- **Incremental Solving** - Push/pop for efficient constraint management, with per-entry Tseitin-memo retraction and an O(1) verdict cache for repeated `(check-sat)` on an untouched goal
- **Proof Generation** - DRAT, Alethe, LFSC, Coq/Lean/Isabelle export
- **Optimization** - MaxSAT, OMT with Pareto optimization
- **Model Checking** - CHC solving with PDR/IC3
- **Z3 API Compatibility** - `TacticRegistry`, `FuncInterp`, sort/substitution/pattern APIs
- **ML-Guided Heuristics (opt-in)** - `oxiz-cli --ml-tactic-selection` (off by default) drives a real `oxiz-ml` tactic-recommendation engine (feature extraction, decision-tree training, outcome-based retraining); the SAT core separately exposes a `BranchingHeuristic` trait for custom LBD/conflict-driven heuristics
- **Recursive BV Encoding** - Full nested bit-vector term encoding with structured conflict diagnostics

## Z3 Parity: Differential Suite Results (Honest Comparator) ✅

`bench/z3_parity` compares OxiZ against a real `z3` 4.15.4 binary using a comparator that **never** counts an `Unknown` answer (from either solver) as a match (see [`bench/z3_parity/src/comparator.rs`](bench/z3_parity/src/comparator.rs)) — a solver cannot inflate its "parity" score by declining to answer. Results below are the full 19-logic, 170-benchmark suite as recorded in the tracked per-environment snapshot `bench/z3_parity/results.<os>-<arch>.json`; only [`results.macos-aarch64.json`](bench/z3_parity/results.macos-aarch64.json) is currently in the tree. The methodology's agreement rule — that every tracked snapshot must agree on the **verdict** of every benchmark (`oxiz_result`, `z3_result`, `match_status`), with only the machine-dependent timings (`oxiz_time`, `z3_time`) expected to differ — is what would make the table below a property of OxiZ rather than of one laptop once a second environment's snapshot exists; `bench/z3_parity/tests/cross_env_verdict_agreement.rs` enforces it on every `cargo test`, but with a single snapshot in the tree that check is currently vacuous, not yet exercised across environments. The un-suffixed `bench/z3_parity/results.json` is git-ignored scratch output of the most recent local run — not evidence:

| Logic | Tests | Result | Notes |
|-------|-------|--------|-----------|
| QF_LIA | 16/16 | ✅ 100% Correct | Simplex, branch-and-bound, cutting planes |
| QF_LRA | 16/16 | ✅ 100% Correct | Tableau-based simplex, pivot selection |
| QF_NIA | 1/1 | ✅ 100% Correct | NLSAT with CAD |
| QF_BV | 15/15 | ✅ 100% Correct | Constraint propagation, div/rem, logical ops |
| QF_S | 10/10 | ✅ 100% Correct | Ground string decision procedure (verified models) |
| QF_FP | 12/12 | ✅ 100% Correct | Concrete FP model finder, bit-exact directed rounding |
| QF_DT | 10/10 | ✅ 100% Correct | Constructor exclusivity, cross-variable propagation |
| QF_A | 10/10 | ✅ 100% Correct | Read-over-write, extensionality |
| UFLIA | 20/20 | ✅ 100% Correct | Skolem witness synthesis + CEGAR; up from 14/20 |
| UFLRA | 10/10 | ✅ 100% Correct | Symbolic model certification + quasi-macros; up from 5/10 |
| AUFLIA | 10/10 | ✅ 100% Correct | Finite-range quantifier expansion; up from 7/10 |
| AUFLIRA | 5/5 | ✅ 100% Correct | Quantified mixed int/real with arrays |
| QF_NIRA | 5/5 | ✅ 100% Correct | Mixed nonlinear integer/real arithmetic |
| QF_ABV | 5/5 | ✅ 100% Correct | Arrays + bit-vectors |
| QF_ALIA | 5/5 | ✅ 100% Correct | Arrays + linear integer arithmetic |
| QF_AUFBV | 5/5 | ✅ 100% Correct | Arrays + UF + bit-vectors (Nelson-Oppen) |
| QF_AUFLIA | 5/5 | ✅ 100% Correct | Arrays + UF + linear integer arithmetic |
| QF_UFLIA | 5/5 | ✅ 100% Correct | UF + linear integer arithmetic |
| QF_UFLRA | 5/5 | ✅ 100% Correct | UF + linear real arithmetic |
| **TOTAL** | **170/170 Correct** | **✅ 100%** | 0 Wrong, 0 Inconclusive, 0 Timeout, 0 Error — see caveats below |

The original 8-logic, 88-benchmark quickstart core (QF_LIA, QF_LRA, QF_NIA, QF_BV, QF_DT, QF_A, QF_S, QF_FP) remains 88/88 — it is the first eight rows above minus the two symbolic-`RoundingMode` QF_FP benchmarks added in 0.3.3 (the `QF_FP` row above is now 12/12: the 10 original quickstart benchmarks plus those 2). The result was verified over three consecutive full runs on an idle machine, plus a fourth after 0.3.1's repeated-`(check-sat)` work.

### What This Means

- ✅ **Every Benchmark Decisive, Every Answer Matching**: all 19 logic families reach 100% Correct, with no `Unknown`, no timeout and no process error anywhere in the run. Because the comparator refuses to score `Unknown` as a match, the score cannot be inflated by declining to answer — 170/170 means OxiZ committed to a verdict on every benchmark and z3 agreed with all of them.
- ⚠️ **100% of the Suite, Not "100% Z3 Compatibility"**: this is a claim about the differential parity suite and nothing wider. The suite is 170 benchmarks across 19 logics; it does not cover `QF_NRA`, `HORN`, or the long tail of SMT-LIB, and a perfect score on it is not evidence that any given formula outside it will be decided. Coverage gaps are tracked in [`TODO.md`](TODO.md).
- ⚠️ **Not a General Production-Readiness Claim**: the 2026-07-16 audit, 0.3.0's hardening waves, 0.3.1's soundness sweep and this release's soundness sweep found and fixed soundness gaps across the parser, quantifier elimination, MBQI, SAT conflict analysis, NLSAT, math, MaxSAT/QE, Spacer, EUF/arithmetic theory combination, and proof checking — but items such as NLSAT irrational-root isolation remain open; see [`TODO.md`](TODO.md) for the itemized gaps and fix status before relying on OxiZ outside this suite's scope
- ✅ **Pure Rust**: Achieved without any C/C++ dependencies

This snapshot validates OxiZ's arithmetic, BV, datatype, array, string, FP, combined-theory and quantified reasoning against Z3 across the whole differential suite, while being explicit that logics outside the suite remain ongoing work.

## Project Statistics (v0.3.2, 2026-08-05)

| Metric | Value |
|--------|-------|
| Rust Lines of Code (code) | 451,853 |
| Total Rust Lines (with comments/blanks) | 564,303 across 1,276 files |
| Total Tests | 9,953 passing, 8 skipped (`--all-features`) at the last full nextest run, plus 110 doc-tests |
| Z3 Parity (differential suite, 170 benchmarks / 19 logics) | **170/170 (100%) Correct**, 0 Wrong / 0 Inconclusive / 0 Timeout / 0 Error, 19/19 logics at 100% |
| Z3 Parity (quickstart core subset, 88 benchmarks) | **88/88 (100%) Correct**, 8/8 logics at 100% |
| Crates | 17 |

### Codebase Breakdown by Module

| Module | Description |
|--------|-------------|
| Core/AST/Tactics | Term management, sorts, tactics framework |
| Theories (EUF/BV/Arrays) | Theory solvers: EUF, LRA, LIA, BV, Arrays, Strings, FP, ADT |
| SAT Solver | CDCL SAT solver with optimizations and generic proof writers |
| Math Libraries | Simplex, matrix operations, polynomials |
| Proof System | Resolution, interpolation, DRAT/LRAT |
| NLSAT (CAD) | Non-linear arithmetic via CAD; root isolation and resultant completed |
| Main Solver | CDCL(T) integration layer; eval_in_model added |
| Optimization | MaxSMT/OMT/Pareto wired to real solver backend |
| Model Checking | SPACER/PDR/IC3; real BMC and sound k-induction wired |
| ML Integration | Neural network guided heuristics |

## Workspace Structure

```
oxiz/
├── oxiz/           # Meta-crate (unified API)
├── oxiz-core/      # Core AST, sorts, SMT-LIB parser, tactics, rewriters
├── oxiz-math/      # Mathematical algorithms (polynomials, matrices, LP)
├── oxiz-sat/       # CDCL SAT solver with VSIDS/LRB/VMTF
├── oxiz-nlsat/     # Nonlinear arithmetic (CAD, algebraic numbers)
├── oxiz-theories/  # Theory solvers (EUF, Arith, BV, Arrays, Strings, FP, ADT)
├── oxiz-solver/    # Main CDCL(T) orchestration, MBQI
├── oxiz-opt/       # Optimization (MaxSAT, OMT)
├── oxiz-spacer/    # CHC solving, PDR/IC3, BMC
├── oxiz-proof/     # Proof generation and verification
├── oxiz-py/        # Python bindings (PyO3/maturin)
├── oxiz-wasm/      # WebAssembly bindings
├── oxiz-smtcomp/   # SMT-COMP benchmarking utilities
├── oxiz-cli/       # Command-line interface
├── oxiz-ml/        # ML-guided heuristics (neural networks)
└── oxiz-vscode/    # VS Code extension (TypeScript, SMT-LIB2 language support)
```

## Requirements

**Minimum Rust Version:** 1.88.0 (stable), declared as `rust-version` in the workspace `Cargo.toml`.

Edition 2024 itself only requires rustc 1.85, but the workspace makes pervasive use of let-chains (`if ... && let Some(x) = ... { }`), which were stabilized in 1.88 — that is the real floor, not 1.85.

For optimal performance, we recommend:
- Rust 1.88.0 or later (stable)
- 8GB+ RAM for building
- 4GB+ RAM for running complex SMT queries

## Quick Start

### Installation

```toml
# Add to your Cargo.toml
[dependencies]
oxiz = "0.3.3"  # Default includes solver
```

Or with specific features:

```toml
[dependencies]
oxiz = { version = "0.3.3", features = ["nlsat", "optimization"] }
```

For all features:

```toml
[dependencies]
oxiz = { version = "0.3.3", features = ["full"] }
```

### Building from Source

```bash
git clone https://github.com/cool-japan/oxiz
cd oxiz
cargo build --release
```

### Running Tests

```bash
cargo nextest run --all-features
```

### Using the CLI

After installation:

```bash
# Install from crates.io
cargo install oxiz-cli

# Solve an SMT-LIB2 file
oxiz input.smt2

# Interactive mode
oxiz --interactive

# With verbose output
oxiz -v input.smt2
```

Or run directly from source:

```bash
# Solve an SMT-LIB2 file
cargo run --release -p oxiz-cli -- input.smt2

# Interactive mode
cargo run --release -p oxiz-cli -- --interactive

# With verbose output
cargo run --release -p oxiz-cli -- -v input.smt2
```

### Library Usage

```rust
use oxiz::solver::Context;

fn main() {
    let mut ctx = Context::new();

    let results = ctx.execute_script(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (declare-const y Int)
        (assert (> x 0))
        (assert (< y 10))
        (assert (= (+ x y) 15))
        (check-sat)
        (get-model)
    "#).unwrap();

    for result in results {
        println!("{}", result);
    }
}
```

## Supported Logics

Status reflects results on the `bench/z3_parity` suite against a real `z3` 4.15.4 binary with an honest comparator (`Unknown` never counts as a match); logics marked Alpha/Partial have known gaps documented in [`TODO.md`](TODO.md) (e.g. NLSAT root isolation for irrational roots, MBQI completeness, PDR/IC3 consecution checks).

| Logic | Description | Status |
|-------|-------------|--------|
| QF_UF | Uninterpreted Functions | ✅ Complete |
| QF_LRA | Linear Real Arithmetic | ✅ Complete (16/16) |
| QF_LIA | Linear Integer Arithmetic | ✅ Complete (16/16) |
| QF_BV | Fixed-size BitVectors | ✅ Complete (15/15) |
| QF_DT | Datatypes (ADT) | ✅ Complete (10/10) |
| QF_A | Arrays | ✅ Complete (10/10) |
| QF_NIA | Nonlinear Integer Arithmetic | ✅ Complete (1/1 quickstart test; broader NIA branch-and-bound has known scoping gaps, see `TODO.md`) |
| QF_S | Strings | ✅ Complete (10/10 Correct via the ground string decision procedure) |
| QF_FP | Floating Point | ✅ Complete (12/12 Correct via the concrete FP model finder) |
| QF_NRA | Nonlinear Real Arithmetic | 🔶 Alpha (irrational-root isolation still open) |
| AUFLIA / UFLIA / UFLRA | Quantified logics | ✅ Complete on the parity suite (AUFLIA 10/10, UFLIA 20/20, UFLRA 10/10 since 0.3.1, via finite-range expansion, Skolem-witness CEGAR and symbolic model certification); MBQI is still incomplete in general, see `TODO.md` |
| AUFLIRA | Quantified mixed Int/Real + Arrays | ✅ Complete (5/5 on the parity suite) |
| AUFBV | Arrays + UF + BV | 🔶 Alpha (`QF_AUFBV` variant scores 5/5 on the parity suite) |
| QF_NIRA | Nonlinear Integer + Real Arithmetic | ✅ Complete (5/5 on the parity suite) |
| HORN | Constrained Horn Clauses | 🔶 Partial (real PDR init/transition SMT queries; not part of the parity suite) |

## Key Components

### SAT Solver
- CDCL with two-watched literals
- Multiple branching heuristics (VSIDS, LRB, VMTF, CHB)
- Clause learning with minimization
- Preprocessing (BCE, BVE, subsumption)
- DRAT/LRAT proof generation via generic `DratWriter<W>` / `LratWriter<W>` (any `Write + Send` sink)
- Local search and lookahead
- AllSAT enumeration

### Theory Solvers
- EUF with congruence closure
- LRA with Simplex
- LIA with branch-and-bound, Cuts
- BV with bit-blasting and word-level reasoning
- Arrays with extensionality
- Strings with automata
- Floating-point with bit-precise semantics
- Datatypes (ADT) with testers/selectors
- Pseudo-Boolean constraints
- Special relations (partial/total orders)

### Quantifier Handling
- E-matching with triggers
- MBQI (Model-Based Quantifier Instantiation)
- Skolemization
- DER (Destructive Equality Resolution)
- Model-Based Projection

### Optimization
- MaxSAT (Fu-Malik, RC2, LNS)
- OMT with lexicographic/Pareto optimization
- Weighted soft constraints

### Model Checking
- PDR/IC3 for CHC solving
- BMC (Bounded Model Checking)
- Lemma generalization
- Craig interpolation

## Architecture

OxiZ follows a layered CDCL(T) architecture:

1. **SAT Core** (`oxiz-sat`) - CDCL solver with modern heuristics
2. **Theory Solvers** (`oxiz-theories`) - Modular theory implementations
3. **SMT Orchestration** (`oxiz-solver`) - Theory combination and DPLL(T)
4. **Tactics** (`oxiz-core`) - Preprocessing and simplification
5. **Proof Layer** (`oxiz-proof`) - Proof generation and verification

## Beyond Z3: Rust-Specific Enhancements

OxiZ goes beyond Z3 with Rust-native features:

### 🦀 Rust Advantages

- **Memory Safety**: No segfaults, buffer overflows, or undefined behavior
- **Zero-Cost Abstractions**: Generic programming without runtime overhead
- **Fearless Concurrency**: Safe parallel solving with work-stealing
- **Modern Type System**: Algebraic data types, pattern matching, trait-based design
- **Package Ecosystem**: Seamless integration with Rust's cargo ecosystem

### ⚡ Performance Optimizations

- **SIMD Operations**: Vectorized polynomial and matrix operations
- **Custom Allocators**: Arena allocation for AST nodes, clause pooling
- **Lock-Free Data Structures**: Concurrent clause database access
- **Compile-Time Optimization**: Monomorphization and inline expansion

### 🎯 Unique Features

1. **Enhanced Proof Systems**
   - Machine-checkable proofs exportable to Coq, Lean 4, and Isabelle/HOL, in addition to DRAT/Alethe/LFSC (`oxiz-proof`)
   - Proof compression and optimization
   - Interactive proof exploration

2. **WebAssembly Bindings**
   - Compact WASM bundle — see [`oxiz-wasm/README.md`](oxiz-wasm/README.md) for current, build-specific size measurements
   - Code splitting for lazy theory loading
   - Browser-optimized memory management

3. **ML-Guided Heuristics** (opt-in)
   - `oxiz-cli --ml-tactic-selection` (off by default) selects a solver posture per formula via a trained decision tree and learns from outcomes
   - Adaptive restart policies
   - Clause usefulness prediction

4. **Advanced Type Safety**
   - Compile-time logic validation
   - Type-safe term construction
   - Impossible state elimination

5. **Developer Experience**
   - Rich error messages with suggestions
   - Comprehensive documentation
   - Property-based testing with proptest

## Python Bindings

OxiZ provides Python bindings via PyO3:

```bash
# Install from PyPI (when published)
pip install oxiz

# Or build from source
cd oxiz-py
pip install maturin
maturin develop --release
```

```python
import oxiz

tm = oxiz.TermManager()
solver = oxiz.Solver()

x = tm.mk_var("x", "Int")
y = tm.mk_var("y", "Int")
solver.assert_term(tm.mk_gt(x, tm.mk_int(0)), tm)
solver.assert_term(tm.mk_eq(tm.mk_add([x, y]), tm.mk_int(10)), tm)

if solver.check_sat(tm) == oxiz.SolverResult.Sat:
    print(solver.get_model(tm))
```

## WebAssembly

OxiZ can be compiled to WebAssembly for browser use:

```bash
cd oxiz-wasm
wasm-pack build --target web
```

Beyond the basic async solver API, `oxiz-wasm` provides:
- **Hard-preemptible solving** (`PreemptibleSolver`): runs a solve inside a dedicated `Worker` and can `terminate()` it from the main thread on timeout — a real interrupt, unlike racing a `setTimeout` against a synchronous, non-yielding solve loop.
- **Cooperative cancellation** (`CancellationToken`): a `SharedArrayBuffer`/`Atomics`-backed flag a worker polls between operations, plus a plain-`JsValue` message protocol (`WorkerHandler.handleMessage`) for a real `Worker`'s `onmessage` handler.
- **Generated TypeScript definitions** (`generate_typescript_dts()`): the `oxiz.d.ts` type definitions are generated from the Rust source, so they can't drift from the JS API.

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## Sponsorship

OxiZ is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find OxiZ useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team KitaSan)

## Benchmarks

`bench/z3_parity` records real wall-clock time for both solvers, but OxiZ is invoked as an in-process library call while Z3 is invoked as an external subprocess (see [`bench/z3_parity/METHODOLOGY.md`](bench/z3_parity/METHODOLOGY.md)) — the two aren't measured under comparable conditions, so this repo does not publish a solver-vs-solver speed ratio from that data. There is no other in-repo apples-to-apples throughput benchmark against Z3 yet; `bench/regression` and `bench/profile` track OxiZ's own performance over time (see `docs/PROFILING_REPORT.md`) rather than a cross-solver comparison. Treat any "OxiZ is Nx faster/slower than Z3" claim you see elsewhere as unverified until it cites a same-process, same-hardware methodology.

## Roadmap

Historical high-level phases, not a parity percentage claim — see "Z3 Parity" above and [`TODO.md`](TODO.md) for the actual measured gaps still open.

### Phase 1: Quick Wins ✅ Complete
- Export unintegrated modules
- Fix API compatibility issues
- Complete enhanced MaxSAT solvers

### Phase 2: High-Impact Features ✅ Complete
- SMT integration layer enhancement
- Math libraries expansion
- Quantifier elimination expansion
- Tactics system expansion

### Phase 3: Rust-Specific Enhancements ✅ Mostly Complete
- ✅ Comprehensive error handling
- ✅ Trait-based architecture
- ✅ SIMD & parallel optimizations
- ✅ Property-based testing
- ✅ Documentation generation

### Phase 4: Advanced Features ✅ Mostly Complete
- ✅ Machine-checkable proof export (Coq/Lean/Isabelle)
- ✅ WebAssembly bindings
- ✅ ML-guided heuristics (opt-in, off by default)

### Phase 5: Gap Closure 🔄 In Progress
- Additional rewriters
- Muz/Datalog expansion
- SAT solver enhancements
- Closing the remaining `TODO.md` items surfaced by the production-readiness audit and the 0.3.1 soundness sweep (NLSAT irrational-root isolation, MBQI completeness beyond the parity suite, etc.)

## Acknowledgments

This project is inspired by and references the algorithms in:
- Z3 (Microsoft Research) - Primary reference implementation
- CVC5 (Stanford/Iowa) - Theory integration techniques
- MiniSat/Glucose - CDCL SAT solving
- Various academic papers on SMT solving

### Key References

- "Satisfiability Modulo Theories" (Barrett et al., 2018)
- "Programming Z3" (de Moura & Bjørner, 2008)
- "DPLL(T): Fast Decision Procedures" (Ganzinger et al., 2004)
- "Efficient E-matching for SMT Solvers" (de Moura & Bjørner, 2007)
