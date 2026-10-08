<!-- In-tree copy (decision (62), re-fix pass 14) of the round-4 scratch tooling's
REBUILD notes.  Paths are relative to the scratch root `$R` = `$OXIZ_ROUND4_SCRATCH`
(default `<repo>/target/round4`); the tools derive the repository root from their own
location (`scripts/round4/` -> `Path(__file__).resolve().parents[2]`). -->

# oxiz4 tooling — REBUILD (decision (42), re-fix pass 12, Fixer A, 2026-09-28)

Everything the round-4 passes measured with before 2026-09-28 (rk6/rc5 corpora,
score6.py, pairverd.py, qvsg.py, tt3.py, sweep.py, det.py and every minimal repro
outside the tree) was destroyed by macOS tmp cleanup.  This directory rebuilds it
once; Fixer B and the recheck reuse it as is.  Scratch root:
`$OXIZ_ROUND4_SCRATCH` (default `<repo>/target/round4`; below: `$R`).

## Probes (`$R/probes`, all release, one tiny bin over `Context::execute_script`)

| probe | source | build |
|---|---|---|
| `probe_base` | `c4b04b7` via `git archive` into `probes/src_base` | `cd probes/probe_base && cargo build --release -j 2` |
| `probe_033` | crates.io `oxiz-solver = "=0.3.3"` | same |
| `probe_head` | `c702310` via `git archive` into `probes/src_head` (the BEFORE of every before/after of pass 12) | same |
| `probe_tree` | the working tree's `oxiz-solver` (path dep) — REBUILD after every code edit | same |

`probes/build_all.sh [names]` builds them one at a time.  Each probe prints every
response line of `execute_script` (an `Err` folds into one `(error "...")` line, as the
in-tree pins and the conformance runner do) and `# elapsed_ms <ms>` on stderr.  The
`Cargo.lock` of the working tree was copied into head/base/tree so they resolve the same
dependency versions.  Build times on 2026-09-28 (machine shared with another workflow):
head 167 s, base 239 s, 033 228 s, tree 301 s.

## Corpora (`$R/corpus`)

* `qmbqi120/` — `tools/gen_qmbqi.py --seed 20260928 --count 120 --out corpus/qmbqi120`.
  120 pairs `qNNNN.smt2` (one `forall` over `(_ BitVec 7|8)`) / `gNNNN.smt2` (the same
  script with that `forall` written out as an `and` over every point, same assertion
  position).  `MANIFEST.txt` records the seed, the generator sha256 and one line per file
  (sha256 prefix, width, arrays, ground assertions, body shape).  Shape mix (seed
  20260928): ite 32, sel_eq 29, xor 17, two_arr 15, store_at 9, min_const 8, impl 10.
* `q/`, `qnoite/` — `tools/gen_q12.py --seed 20260929 --out corpus` (q = seed 20260929,
  qnoite = seed 20260930, `--no-ite`), 400 pairs each at index widths 1-2.
* `qeq120/` — `tools/gen_qmbqi.py --seed 20260931 --count 120 --array-eq --out corpus/qeq120`: the same shapes
  with one ground array equality `(= aK <store/const term>)` prefixed to every script (the `#P2b-60` shape).
  NOTE: `qmbqi120/MANIFEST.txt`'s header was written before `--array-eq` existed, so its `generator-sha256` differs
  from a regeneration today and it has no `array-eq` line; the 240 files themselves regenerate byte-identically
  (checked: MANIFEST bodies diff-clean).
* `named/` — repro scripts copied VERBATIM from the in-tree pins (`q33_a01.body`,
  `x10_q33_swapped.body` = `round4_pass10_recheck_pins` Q33_* constants in the two
  orders; `a6_three_line`, `c11_bool_element`, `p2b51_six_line`, `one_point_off`,
  `guarded`) and the wrong-`sat` repros found on 2026-09-28 (`wsat1/3/4/6`,
  `wsat_q0000_pinned`), and the pass-12 additions: `wreal`, `wusort2`, `wbool`, `wsigned` (`#P2b-60` spellings),
  `wdt`, `wdt_sat`, `wdt_ground` (`#P2b-61`, datatype constructors not distinct as array indices), `qeq_q0033`
  (the verdict the saturation-time completion keeps), `q0107` (pass-9 pin verbatim), `q0072` (the one base-decided
  qmbqi120 loss), `bench_array_ext` (`#P2b-62`), `au_pinned_{head,tree}_model` (pinned re-checks of array_update).

No generator emits `(set-option :timeout ...)` (asserted in gen_qmbqi.py and
fuzz_pushpop.py).

## Scripts (`$R/tools`), with how each was validated

| script | what it does | validated by |
|---|---|---|
| `common.py` | S-expression reader; compiler of the generated fragment (Bool, BV, arrays over BV/Bool index, ite, forall/exists over finite sorts, let) to Python; brute force over every interpretation; exact model evaluation; probe runner (`timeout cap`, panic = non-zero exit or "panicked") | hand cases (a sat/unsat pair at width 2, a `store`-chain model value) |
| `gen_core.py`, `gen_qmbqi.py` | seeded pair generator (decision (42)(a)) | same seed twice -> identical MANIFEST; `min_const` shape reaches #P2b-51 (q0000 falsifying on head) |
| `gen_q12.py` | q / qnoite (42)(b) | brute force decides all 800 |
| `score.py` | #P2b-51's instrument (42)(c): totals `{pairs, q_sat, q_sat_g_unsat, q_unsat_g_sat, falsifying_models, models_confirmed, models_unresolved, panics, agree}` + solver-free `eval_false/eval_true/eval_na` + `pinned_wrong_sat` | exact evaluation vs pin re-check on probe_head: the pin re-check CONFIRMED 19 models the exact evaluator refutes -> found the wrong-`sat` family below; `eval_false` is the decisive figure |
| `score12.py` | q/qnoite against the brute-force oracle: wrong_sat / wrong_unsat / falsifying / unknown_q / q_vs_g / agree | probe_base: 27+41 wrong `sat`, 217+122 falsifying; probe_head 0/0/0 |
| `qvsg.py` | quantified vs ground twin (42)(e) | probe_base: WRONG_SAT 5 on qmbqi120 |
| `pairverd.py` | base-vs-tree LOST / GAINED / FLIPPED with counters (42)(d) | base vs head: LOST 6, GAINED 41, FLIPPED 5 |
| `sweep.py` | the 217-script sweep (42)(f): the composition is `bench/**/*.smt2` = 217 files (170 z3_parity + 43 extended_theories + 4 regression; CHANGELOG: "the 217-script `bench/` corpus"); refuses to report if the count is not 217; `--get-model` appends `(get-model)` after every `(check-sat)` | base vs head: 0 verdict / 0 response (the recorded historical figure); with `--get-model` 18 response differences (the round's model fixes) |
| `det.py` | two runs of one probe over the sweep (plain + get-model) and any file globs (42)(g) | head: see BEFORE.md |
| `ladder.py` | one script at `:max-bv-embedded-checks` rungs, verdict + ms + the five counters per rung | q33_a01 / swapped counters on HEAD identical to TODO's recorded 9/75, 10/76, 10/76, 59/199 and 8/74, 25/97, 33/122, 33/122 |
| `replay_pins.py` | the pass-12 pin scripts replayed against a probe with the exact evaluator (models) or the verdict; `--env K=V` for the env-gated mutations of the isolated copy | HEAD: 7 of 13 red; tree: 0 of 13 |
| `sweep_models.py` | for every changed response of a `--get-model` sweep, both published models pinned back into the script and re-solved by a judge probe | classified the 5 changed responses (HEAD-vs-tree) |
| `fuzz_pushpop.py` | incremental QF_BV scripts (2-3 vars, widths 1-8, <= 18 bits of state, push/pop depth <= 2, >= 2 check-sat) against a numpy brute-force oracle (42)(i); every seed independent (`Random("seed/index")`) | probe_033 vs probe_head, 500 scripts seed 1, see BEFORE.md |

`rslines` is not installed on this machine; the 2,000-line gate is checked with `wc -l`
(excluding `.claude/worktrees/`).

Limits honoured: every campaign 2 worker processes, never beside a build or a nextest
run; every probe under `timeout 130` (a corpus never gets 900).


## Re-fix pass 13 additions (2026-09-29, fix13)

* **Decision (50), the memory guard, is now in the shared `common.py`** (ported from the recheck's guarded copy
  `recheck12/tools/common.py`, with `PROBES` kept relative to this directory): `MAX_POINTS = 2^16`; `domain_size`
  refuses a bit-vector index sort wider than 16 bits, `domain()` an array domain of more than 2^16 values, the
  compiler a quantifier over more than 2^16 points, `brute_force` sizes the interpretation space before it
  materialises anything, and `eval_model` / `const_value` answer `None` (= unresolved) on `TooBig`.
  **`setrlimit(RLIMIT_AS, …)` is a no-op on this macOS host** — it raises `ValueError: current limit exceeds
  maximum limit` (RLIMIT_DATA likewise) and a 6 GiB allocation succeeds afterwards — so the structural guard is
  what protects the machine; the rlimit is still attempted and its outcome is `common.RLIMIT_STATUS`. The
  22 GB / 31 GB kills were `((v,)*2^32)` tuples compiled from `(as const (Array (_ BitVec 32) …))`.
* `fuzz_abv.py` — the recheck-12 incremental QF_AUFBV fuzzer (index width 1-2, element width 1-3, 1-2 arrays,
  optional UF, store/select/const/array ite/array (dis)equality, push k / pop k to depth 3, >= 2 check-sat,
  get-value of every cell, named cores; numpy oracle over <= 16 bits), copied here verbatim except that it
  imports this directory's `common.py`. Validated by the recheck: 0.3.3 seed 4242 x 400 -> 76 wrong sat.

## Re-fix pass 13, finishing instance (2026-09-29, fix13b)

* `$R/probes/probe_treeda` — `probe_tree`'s crate with `debug-assertions = true` and `overflow-checks = true` in its
  release profile, so the `debug_assert!`s of the nextest test profile run at release speed (`fuzz_abv.py --probe
  <absolute path>` accepts it: `common.probe_path` passes a path through). Build: `cd $R/probes/probe_treeda && cargo
  build --release -j 2` (3 min).
* No tool in this directory changed. The campaigns of the pass are `$R/fix13b/camp.sh` (see `$R/fix13b/REBUILD.md`);
  every run used 2 workers and never ran beside a build or a nextest run; 0 watchdog kills.

## Recheck 14 / re-fix pass 15 additions (2026-09-30)

* `g14a/` — `gen14.py --seed 30093001 --count 3000 --out <dir>`: recheck 14's mixed-theory corpus (`QF_UFLIA` /
  `QF_UFLRA` / `QF_AUFLIA`, quantified UF and arrays over `Int` / bit-vectors, datatype indices, push/pop, a
  `(get-model)` and a `(get-value)` after every check).  Judged with z3 by `camp14.py <dir> --probe P --out O --jobs 2
  --cap 10 --zcap 10` (every verdict against z3's, every printed model and `(get-value)` pair replayed by z3).
* `bench_gm/` — the 217 `bench/**/*.smt2` with `(get-model)` after every `(check-sat)`, named `bNNN.smt2` with a
  `MAP.txt`; judged the same way (`--cap 130 --zcap 20`).
* z3 is an external binary on `PATH` (4.15.4 in the round); see `README.md`, "z3 as an independent judge".
* The campaigns of re-fix pass 15 are `$R/fix15/final_camp.sh`, `elig_ab.sh` (an A/B under one mutation switch) and
  `capedge.sh` (cap-edge scripts re-run alone at 130 s); see `$R/fix15/REBUILD.md`.  2 workers, never beside a build.

## Recheck 15 / re-fix pass 16 additions (2026-09-30)

* Recheck 15's generators, moved in-tree with no path baked in: `gen_mix.py --seed 30093153 --count 3000`,
  `gen_dt.py --seed 30093154 --count 3000` (`dt6` = its first 600 scripts), `gen_guard.py --seed 30093152 --count
  2000`; each judged with `camp14.py --jobs 2 --cap 10 --zcap 10`.  `cmpcamp.py OLD/results.jsonl NEW/results.jsonl`
  compares two runs check by check; `ddmin.py` reduces a script while a verdict property of two probes holds.
* `camp14.py` judges the longest answered prefix of a timed-out script (`--no-prefixes` turns it off), `zjudge.py`
  counts a refuted `@uc_` model of a quantifier-free script as falsifying, and `det.py` names cap-edge pairs and re-runs
  them uncapped.  Re-fix pass 16's campaigns are `$R/fix16/final_camp.sh` (see `$R/fix16/REBUILD.md`).

## Re-fix pass 18 additions (2026-10-02)

* `gen_dtite.py --seed 30101803 --count 300 --out <dir>`: `TODO.md` `#P2b-90`'s family by construction (testers,
  selectors and datatype equalities over datatype `ite`s — chains, nests, under uninterpreted and constructor
  arguments, enumeration and uninterpreted-sort `ite`s, push/pop, a quantified assertion in about a third of the
  scripts); judged with `camp14.py --jobs 2 --cap 10 --zcap 10`.  Seed 30101802 with `--v1` is the first version's corpus (byte-identical regeneration checked), whose
  selectors are not guarded by their tester: a printed model leaves `(hd nil)` unspecified and z3 then chooses a value
  that falsifies the script, so `selcheck.py PROBE SCRIPT...` re-asks z3 whether the model satisfies the assertions for
  SOME value of the unspecified terms (every one of that corpus's 14 "falsifying" models does).
* `zjudge.py` writes `sat/uc_unresolved` in `per_check` for a quantified script's refuted `@uc_` model, and
  `cmpcamp.py` re-classes a `UC_SAT` event the same way (adversarial recheck 17's minor 7; no such event occurred on
  any corpus).  Re-fix pass 18's campaigns are `$R/fix18/phases.sh` (see `$R/fix18/REBUILD.md`).

## Re-fix pass 19 additions (2026-10-02)

* `camp14.py` keeps, for a script flagged on its judged prefix, the prefix (`flagged/<script>.prefix.smt2`) and that
  prefix's responses in `flagged/<script>.out` (adversarial recheck 18's minor 5: the whole script's empty stdout was
  kept, so `r00005` needed `check_prefixes` by hand).  No new campaign: re-fix pass 19 records adversarial recheck 18's
  rows (`$R/recheck18/final/`, compared by `$R/fix19/tools/cmp19.py`, recheck 18's `cmp18r.py` widened to its fresh
  seeds and to decision (24a)'s mechanism (H)); see `$R/fix19/REBUILD.md`.
