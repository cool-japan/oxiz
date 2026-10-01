# Round-4 measurement tooling (in-tree copy)

The instruments the round-4 passes (`TODO.md` "cargo-formal intake") measured with, copied
from the round's scratch directory by re-fix pass 14 (decision (62)).  They drive release
*probes* — one tiny binary per build over `oxiz_solver::Context::execute_script` — under
`$OXIZ_ROUND4_SCRATCH/probes/probe_<name>/target/release/probe_<name>`.  The scratch root
comes from `OXIZ_ROUND4_SCRATCH` (default `<repo>/target/round4`); `--root` defaults to the
repository these scripts live in.  No absolute path is written in any file here.  How the
probes and corpora are built is in `REBUILD.md`.  Every campaign runs at most two worker
processes and never beside a build or a test run; `common.py` refuses brute force above
2^16 index points (decision (50)).

| tool | what it does |
|---|---|
| `common.py` | S-expression reader, exact evaluator / brute-force oracle for the generated fragment, probe runner, the memory guard |
| `gen_core.py` | the shared generator core behind `gen_qmbqi.py` |
| `gen_qmbqi.py` | seeded width-7/8 quantified / ground-twin pair corpus (`qmbqi120`, `qeq120 --array-eq`, `fq200`, `feq200`) |
| `gen_q12.py` | seeded width-1/2 paired corpora `q` / `qnoite`, decidable by brute force |
| `score.py` | `#P2b-51`'s instrument: sat / falsifying / confirmed / unresolved models, exact evaluation and pinned re-check |
| `score12.py` | `q` / `qnoite` against the brute-force oracle: wrong sat / wrong unsat / falsifying |
| `qvsg.py` | quantified script against its ground twin: wrong sat / wrong unsat / agree |
| `pairverd.py` | base-vs-tree verdicts over a corpus: LOST / GAINED / FLIPPED with counters |
| `sweep.py` | the 217-script `bench/` sweep, verdict and full response text, plain or with `(get-model)` |
| `sweep_models.py` | every changed `(get-model)` response of a sweep, both models pinned back and re-solved by a judge probe |
| `det.py` | determinism: two runs of one probe, response for response; TIMEOUT-in-both and cap-edge pairs named and re-run uncapped |
| `ladder.py` | one script at a ladder of `:max-bv-embedded-checks` budgets with the five counters per rung |
| `lossrun.py` | named losses of a corpus run to 130 s on two probes with `:reason-unknown` and counters |
| `replay_pins.py` | the pass-12 pin scripts replayed against a probe (env-gated mutations of an isolated copy) |
| `fuzz_pushpop.py` | incremental QF_BV push/pop fuzzer against a brute-force oracle (decision (45)) |
| `fuzz_abv.py` | incremental QF_AUFBV fuzzer with models, values and named cores against a brute-force oracle |
| `fuzz_qc.py` | quantified-completion fuzzer: every published model evaluated exactly, every `unsat` re-judged on the ground twin, withheld models counted |
| `qc_eval.py` | `fuzz_qc.py`'s exact model evaluator (arrays, uninterpreted functions, quantifiers expanded) |
| `qc_classify.py` | buckets `fuzz_qc.py`'s falsifying cases by shape |
| `qc_twin_head.py` | re-judges a tree's `fuzz_qc` unsats on the ground twin with an independent probe |
| `zjudge.py` | recheck 14's independent judge: every verdict against z3's, every printed model and `(get-value)` replayed by z3, withheld models counted; a refuted `@uc_` model of a quantifier-free script counts as falsifying |
| `camp14.py` | `zjudge.py` over a directory of scripts with at most two workers (`results.jsonl`, `summary.json`, `flagged/`); a timed-out script's longest answered prefix is judged (`prefix_checks`) |
| `gen14.py` | recheck 14's seeded mixed-theory generator (`QF_UFLIA` / `QF_UFLRA` / `QF_AUFLIA`, quantified UF and arrays, datatypes, push/pop) |
| `gen_mix.py` | recheck 15's seeded mixed `Int` / `Real` generator (`to_real`, `ite`, UF, a guarded universal in 15 %, push/pop) |
| `gen_dt.py` | recheck 15's seeded quantifier-free datatype generator (list / enumeration / record, an uninterpreted sort, UFs into them) |
| `gen_guard.py` | recheck 15's seeded guarded-universal generator (`Int` / `Real` / bit-vector guards under every connective, UF and array bodies) |
| `cmpcamp.py` | per-check comparison of two `camp14.py` runs: LOST (a timed-out script's unanswered checks included), GAINED, model regressions (`sat/ok` → withheld / falsifying / unresolved) and improvements, every class change; prefix-judged checks marked |
| `ddmin.py` | greedy line reducer over a verdict property of two probes |

## z3 as an independent judge

Re-fix pass 15 (decision (69)(11)) moved recheck 14's `zjudge.py` in-tree: `sweep_models.py
--judge z3` and `fuzz_qc.py --judge z3` replay every printed model with z3 instead of (or beside)
an OxiZ probe, because a probe that confirms its own falsifying model proves nothing.  z3 is an
external binary found on `PATH` (override with `Z3=...`); the round measured with z3 4.15.4,
each query run as `z3 -T:<cap> -memory:2048`.  It is used only by these tools, never by the
crates.

## camp14 and prefix judging (adversarial recheck 16's minor 8)

`camp14.py` re-runs a script that timed out prefix by prefix and judges its longest answered
prefix (`prefix_checks` in `results.jsonl`).  A prefix is **a different script** from the whole
one: `Context::execute_script` parses the whole text before it runs a command, so the terms of
later commands are interned first, and a check's search can take another trajectory in the whole
script than in its prefix.  Deterministic, pre-existing (`c702310` behaves the same), and seen on
the round's corpora: `gen_dt.py` seed 30093154 `d00239`'s first check prints a falsifying model in
full and a correct one in its one-check prefix (`c702310` and pass 14 the other way), `d00552`'s
first check is `unknown` in full and `sat` in its prefix, `c702310` answers `gen14.py` seed
30093155 `s00262`'s second check `unsat` in full and `unknown` in its two-check prefix, and
`s02254`'s two prefixes flip pass 14 against pass 16 in both directions.  So `cmpcamp.py` marks
every check of a prefix-judged script `(prefix)`; a comparison that must not mix the two
readings excludes those rows, and a regression is only ever claimed on the whole script as
generated.
