# Round-4 closed items, long form

The long-form text of closed round-4 `#P2b` items, moved **verbatim** out of [`TODO.md`](../TODO.md)
by re-fix pass 14 (2026-09-29, decision (58)) to keep that file under its 1,900-line budget
(`#P2b-76` and `#P2b-77` by re-fix pass 15, 2026-09-30, and `#P2b-70`, `#P2b-71`, `#P2b-72`, `#P2b-74`,
`#P2b-75` and `#P2b-79`, and this pass's own `#P2b-85`, `#P2b-86` and `#P2b-87`, by re-fix pass 16, and `#P2b-1`, `#P2b-2`, `#P2b-53`,
`#P2b-82`, `#P2b-83` and `#P2b-84` by re-fix pass 17, 2026-10-01, decision (79), for the same reason; that pass filed and closed `#P2b-89`
and wrote its long form here directly; `#P2b-59` by re-fix pass 18, decision (87), which filed and closed `#P2b-90` and
reopened and closed `#P2b-89`'s nested case here directly). Each
item keeps a one-line pointer in `TODO.md`'s "cargo-formal intake" section, in its original place;
every id, decision number and file:line reference below is as it stood in `TODO.md` at the move.
`#P2b-58` (f) and `#P2b-63` were reworded in place by the same pass before the move (decision (58):
a declined completion keeps no falsifying model once decision (54) landed).

## #P2b-58

- [x] **#P2b-58 (2026-09-21) — COMPLETENESS, honest `unknown`, base-decided: MBQI cannot certify a `sat` on a
  satisfiable quantified array script whose index sort is `(_ BitVec w)` with `w >= 7`, one bit above
  `finite_expand`'s 64-point budget.** **FIXED AT THE ROOT 2026-09-22 (re-fix pass 11): decision (36)'s model
  completion, published only behind a quantifier-free certificate — see the close-out at the end of this entry
  for the design, the measurements and the mutation table. The two hole pins of pass 9 and the two of pass 10
  are INVERTED and now assert the verdict and the published model.**
  Opened by the cargo-formal adversarial recheck pass 9 on
  `rk6/corpus/qmbqi120` — the round's own 120-pair width-7/8 corpus, the one `#P2b-51` is measured on, which had
  never been run base-vs-tree. `c4b04b7` and crates.io 0.3.3 answer `sat` in milliseconds, by the pre-`#P2b-48`
  free-value accident (the read under the binder is an opaque value of the element sort); this tree answers an
  honest `unknown`. **Measured, release, both probes back to back** (`rk9/pairverd.py`, and `rk9/min/*_st.smt2`
  for the counters): `rk6/corpus/qmbqi120/q0074` `unknown` 26.3 ms here against the base's `sat` 1.0 ms, and the
  SAME formula written out over all 128 points of its index sort is `sat` in 136.7 ms **on this tree** — so the
  truth is `sat` and it rests on no external oracle; `q0106` `unknown` 78.5 ms against `sat` 0.6 ms, its ground
  twin `sat` 1,009.2 ms here; `q0075` `unknown` 25.1 ms against `sat` 6.1 ms. Two further members of the same
  mechanism sit in the round's attack battery: `rk8/atk/f5_binder_collide_index.smt2` and `rf9/atk/wu1`, both
  satisfiable, both decided by the base, and they are the only two tree-side `unknown`s in that 102-script
  battery where the base decides.
  **The counters name the mechanism rather than leaving it to a clock, and they rule out both mechanisms
  decision (24a) named before this one.** None of the three carries `(set-option :max-conflicts N)`, and the
  calibrated ceiling is 250,000 embedded checks against `q0074` `:conflicts 38 :array-refinement-rounds 15
  :array-lemma-instances 45 :bv-embedded-checks 1167 :bv-embedded-conflicts 0`, `q0075` `:conflicts 61
  :array-refinement-rounds 12 :bv-embedded-checks 1745 :bv-embedded-conflicts 0` and `q0106` `:conflicts 34
  :array-refinement-rounds 38 :bv-embedded-checks 5411 :bv-embedded-conflicts 0`. **No budget is exhausted.**
  The body of `q0074`'s quantifier reduces to `a1[i] = #b1` for every `i` over `(_ BitVec 7)`, which needs model
  completion for an array default **under a binder** before a `sat` can be certified.
  **What this is NOT.** Not mechanism (i), `#P2b-46` (f): no budget is installed, and the counters above sit
  **46x to 214x below** the calibrated 250,000-check ceiling (5,411 on `q0106`, 1,745 on `q0075`, 1,167 on
  `q0074`), so nothing ran out. Not mechanism (ii), `#P2b-57`: that family is closed at the root
  and its repros are vacuous positive-polarity obligations, not array defaults. Not `#P2b-50` **as written**:
  `#P2b-50` is a *declared* sort whose cardinality nothing pins, and its lever is finite-model finding over an
  uninterpreted sort; here the index sort's cardinality is pinned at `2^w` and the only thing missing is a model
  over it. Not `#P2b-53`: its repro `rk6/atk3/p4_bv7_two_stores.smt2` answers `unknown` on `c4b04b7` as well, so
  no verdict is lost there, while the defining property of this item is that the base **decides**. `#P2b-50` and
  this item are therefore kept **SEPARATE** — the alternative, widening `#P2b-50`'s scope sentence to "every sort
  `finite_expand` declines, bit-vector index sorts included", was considered and declined because the two want
  different levers — and `#P2b-50` carries a pointer to here.
  **Lever**: model completion for an array default under a binder (Ge & de Moura's MBQI model construction) —
  `#P2b-51`'s lever, costing a **verdict** here and not only a published model. Pinned green as holes by
  `round4_pass9_recheck_pins::a_satisfiable_width_seven_array_script_the_base_decides_is_undecided_here`, with
  `::the_same_formula_expanded_over_its_whole_index_sort_is_sat` as the in-test oracle so the pin is a statement
  about the solver and not about the formula, and by
  `::a_satisfiable_name_collision_script_the_base_decides_is_undecided_here` for the same mechanism on
  `rk8/atk/f5_binder_collide_index.smt2`, with `::the_pinned_twin_of_the_name_collision_script_is_refuted` as its
  control. Both go red the moment the verdict comes back, which is what they are for. *(Both were
  inverted and RENAMED on 2026-09-22 — to `…_is_decided_by_a_certified_completion` in each case — so the two
  names above no longer exist in the tree; they are kept here as the record of what was pinned.)*

  **CLOSED AT THE ROOT 2026-09-22 (re-fix pass 11), by decision (36)'s model completion behind a
  quantifier-free certificate.** New module `oxiz-solver/src/solver/array_completion_certify.rs`
  (661 lines). **(a) The completion.** Every array-sorted free variable the goal mentions is given a
  *total* interpretation — the constant array `((as const A) d)` over a default `d` searched in a
  bounded pool (the element-sort values the candidate model already committed to, then the
  element-sort literals the script spells out, then the two ends of the sort; at most 6 per array,
  at most 3 arrays, at most 24 combinations). That is Ge & de Moura's pins-plus-default model
  construction (CAV 2009), and it is the same rule `mbqi::model_certify` already uses for an
  uninterpreted function's default over `Int` and `Real` — which is exactly why that module could not
  be extended instead: `model_certify::value::value_sort` interprets `Int`, `Real` and `Bool` and
  `None` for every other sort, so an array over a bit-vector index sort needs a value domain, an
  evaluator and a region-stability argument it does not have. **(b) The certificate, and why a `sat`
  can never rest on the completion alone.** Nothing is published until two kinds of *validity* query
  have been discharged by ordinary quantifier-free solves, each budgeted at 20,000 conflicts and
  never at a clock: for every maximal `forall` sub-term `∀x⃗. ψ`, `¬ψ[completion]` with `x⃗` replaced
  by fresh reserved constants must be `Unsat`; and then
  `(or (not A₁[completion]) … (not Aₙ[completion]))` must be `Unsat` — **one** quantifier-free query
  over the assertions, which is the shape decision (36) asked for. `Unsat` means the formula holds
  under *every* interpretation of whatever symbol the module failed to interpret, so the conclusion
  survives an incomplete interpretation; the converse does not, which is why a `forall` that cannot
  be certified **true** makes the attempt decline rather than be recorded as false, and why `exists`
  is declined outright. **(c) Where it is hooked, and what that costs.** In `Solver::check`, after
  `check_core` and *before* the honesty gates, and only where a verdict would otherwise be given up:
  `Unknown`, or a `Sat` one of the gates is about to take away. A `Sat` that survives the gates needs
  nothing from it and an `Unsat` is never revisited. It declines before it starts on a goal above
  4,096 assertion-DAG nodes or carrying an application of a symbol it does not interpret (an
  uninterpreted function survives into the validity query, where it is certain to decline), so the
  common path pays one bounded walk.
  **(d) The assumption `#P2b-58` recorded for the next pass is DISCHARGED, not worked around.** The
  close-out note of re-fix pass 10 asked that decision (36)'s "the index variable is the only free
  symbol left, so it is one QF_BV query" be verified before the certificate was designed, and the
  adversarial recheck pass 10 pinned the script that breaks it
  (`round4_pass10_recheck_pins::a_binder_constraining_two_arrays_at_once_is_decided`, then named
  `…_is_undecided_here`): a body that reads a **second** array whose default the candidate model also
  leaves free. The assumption is false as written and the repair is not a different certificate but a
  wider completion — interpret *every* array whose default is free at the same time, and the negated
  body has the index variable as its only free symbol again. That is why the search is a product over
  per-array pools rather than a search for one default, and it is why `q0106` (three arrays under one
  binder) is in scope at all.
  **(e) Measured on the final tree** (decision (36)(c)), release, one process at a time; "base" is `c4b04b7`.

  | script | base (`c4b04b7`) | tree before | tree after |
  |---|---|---|---|
  | `rk6/corpus/qmbqi120/q0074` | `sat` 7.48 ms | `unknown` 7.0 ms | **`sat` 22.5 ms** |
  | `rk6/corpus/qmbqi120/q0075` | `sat` 0.25 ms | `unknown` 12.5 ms | **`sat` 30.4 ms** |
  | `rk6/corpus/qmbqi120/q0106` | `sat` 0.12 ms | `unknown` 11.3 ms | **`sat` 21.3 ms** |
  | `rk8/atk/f5_binder_collide_index.smt2` | `sat` 0.12 ms | `unknown` | **`sat` 4.6 ms** |
  | `rf9/atk/wu1.smt2` | `sat` 0.07 ms | `unknown` | **`sat` 5.2 ms** |
  | `rk11/atk/m2_name_bound_twice.smt2` | `sat` 0.09 ms | `unknown` 1.1 ms | **`sat` 3.7 ms** |
  | `rk11/atk/m5_model_determinism.smt2` | `sat` 0.10 ms | `unknown` 1.2 ms | **`sat` 4.2 ms** |
  | `rk11/atk/m1_two_arrays_one_binder.smt2` | **`unknown`** 0.39 ms | `unknown` | **`sat` 2.7 ms** |
  | `rk8/atk/f4_binder_collide_array.smt2` | **`unknown`** 0.06 ms | `unknown` | **`sat`** |
  | `rk11/atk/m4_body_false_at_a_point.smt2` | `unsat` 0.04 ms | `unsat` | **`unsat` 2.3 ms** |
  | `rk11/atk/m4b_stored_point_conflict.smt2` | **`unknown`** 0.05 ms | `unsat` | **`unsat` 1.6 ms** |

  (Every figure re-measured on the final tree with the release probes, one process at a time, while the
  machine also carried a concurrent workspace build — so the millisecond columns are upper bounds and the
  verdicts, which are what the entry claims, are not.)
  `m1` and `f4` are verdicts the tree **gains over the base**, not ones it lost. `m4`/`m4b` are the controls: a
  completion the pool offers and the certificate refuses must stay refuted, and both do.
  Decision (36)(c)'s corpus figures, all re-taken on the final tree:
  `rk9/pairverd.py` base-vs-tree over `rk6/corpus/qmbqi120` (cap 20 s, 2 workers) — **LOST 3** where it
  lost 6 (`q0033`, `q0047` and `q0103`, all three `#P2b-59`'s residue on mechanism (i) and all three TIMEOUT at the 20 s cap; the three `#P2b-58` losses `q0074`, `q0075` and `q0106` are gone), GAINED 30, FLIPPED 8 (every flip base `sat` → tree `unsat`, the
  round's soundness fixes); `rk9/qvsg.py` over the same corpus — **WRONG_SAT 0, WRONG_UNSAT 0**, quantified side
  85 `sat` / 26 `unsat` / 3 `unknown` / 6 TIMEOUT against 78 / 26 / 10 / 6 before, agree **96** against 90; `rk6/score6.py` (`#P2b-51`'s measurement, cap 20 s, 2 jobs) — `{pairs 120, q_sat 85, q_sat_g_unsat 0, q_unsat_g_sat 0, falsifying_models 8, models_confirmed 51, models_unresolved 26, panics 0, agree 95}`, which is the half of
  decision (36)(c) that required the falsifying-model count **not to rise**.
  **`#P2b-51` is NOT closed by this** *(superseded 2026-09-28: decision (40) runs the completion at every `Sat`
  exit and `#P2b-51` is closed — see its close-out)* and the reason is structural rather than incidental: the completion runs
  only where a verdict would otherwise be given up, so a script that already answers `sat` with a partial
  candidate model keeps that model. What the completion buys `#P2b-51` is the scripts it moves from `unknown`
  to `sat`, each of which is published with the interpretation its certificate was discharged over.
  `round4_pass6_recheck_pins::a_published_model_still_falsifies_its_own_quantified_assertion` is therefore
  still green as a hole, correctly. Widening the hook to every quantified array `sat` was considered and
  **declined on a gate**: it would change the published model of scripts the 217-benchmark sweep compares
  response-for-response, and the sweep's "0 response differences" is a release gate, not a preference.

  **(f) Scope, stated as what declines rather than as what works (rewritten 2026-09-29 by re-fix pass 14,
  decision (58); the pass-13 wording is in this file's history).** The completion's search declines — the goal
  keeps its `unknown`, or on a `Sat` its candidate model, **which the published-model certificate then checks as
  printed** (decision (54)(i), `array_completion_certify/printed.rs`): a candidate that does not certify is withheld,
  `(get-model)` answering `(error "model not certified: …")`, so a declined completion keeps **no** falsifying model —
  when: over a finite index no constant array over a pooled default and no default-plus-pins interpretation
  certifies (pins: the goal's index literals and the candidate's index-sort constant values; the fill query samples
  one representative per gap of every order a guard can use); over `Int` / `Real` no pooled default (the candidate's
  element values, the literals, each `± 1`) over the candidate's own points certifies (`infinite.rs`; the universal
  is certified pointwise, `split.rs`, where every read under the binder is at the bare bound variable or a ground
  index), or a read at a compound index (`a[i + 1]`) leaves only the validity query over a store chain; an
  uninterpreted application or an array read under a binder that is not one of the arrays being completed
  (`recheck13/atk/c02`); two ground readings of one function or array the candidate gives different values at equal
  arguments; a **group** of more than 3 arrays one chain of assertions links (independent arrays are one search each,
  `groups.rs`); an alternation that survives the peel of a consecutive `forall` chain (`c06`, `d07`); more than 4,096
  assertion-DAG nodes; a query out of its 20,000-conflict budget. A quantifier at negative polarity no longer
  declines: a negated `exists` is certified false through its dual `∀x⃗. ¬φ`, a negated `forall` through a witness of
  `¬φ` (`polarity.rs`); a sub-term neither of whose values certifies is left undetermined, the assertions having to hold for
  both (`(=> p (forall …))` under `p = false`). Where a goal's groups need different treatment the printed candidate is completed around what
  it got right (`Solver::complete_around_printed`). A `Sat` whose candidate model already satisfies every assertion
  as printed keeps it (decision (48), `candidate.rs`). `#P2b-50` stays separate and open: this module never invents a
  domain.
  **(g) Pins and mutations** (decision (36)(d)). The four hole pins are **inverted**:
  `round4_pass9_recheck_pins::a_satisfiable_width_seven_array_script_is_decided_by_a_certified_completion`
  (asserts `sat` **and** replays the published model against the quantifier written out at all 128 points of
  its index sort — the in-test oracle
  `::the_same_formula_expanded_over_its_whole_index_sort_is_sat` stays beside it),
  `::a_satisfiable_name_collision_script_is_decided_by_a_certified_completion`,
  `round4_pass10_recheck_pins::the_guarded_universal_member_of_the_completion_family_is_decided`
  (**new**: `rf9/atk/wu1.smt2` verbatim — the third script on mechanism (iii)'s list, which had no
  in-tree pin at all and was carried only by the round's out-of-tree 102-script battery, so a
  regression there would have been invisible to `cargo nextest`),
  `round4_pass10_recheck_pins::a_satisfiable_script_with_the_index_name_bound_twice_is_decided` and
  `::a_binder_constraining_two_arrays_at_once_is_decided` (asserts `sat` **and** that the published model names
  one constant array per array, `#b1` beside `#b0`). Two new guards sit with them:
  `::a_completion_that_is_wrong_at_a_point_is_refuted_and_never_published` and
  `::the_completed_model_is_the_same_on_two_runs`.

  | mutation | effect, measured |
  |---|---|
  | **M11a** `certify_sat_by_array_completion` returns `false` at entry (no completion at all) | **5 red**: the four inverted pins and `the_completed_model_is_the_same_on_two_runs`. The two refutation guards stay green, which is what makes them independent of the completion. |
  | **M11b** keep the completion, make `certificate_passes` return `true` at entry (completion **without** its certificate) | **2 red**, and they are red for the right reason: the tree answers `sat` for `∀i. a[i] = ¬b[i]` with `a = b = ((as const …) #b1)` — a published model that falsifies its own formula. That is the wrong `sat` the certificate exists to prevent, and the two *model*-checking pins are its witnesses. |
  | **M11b on `m4`/`m4b`** | **no effect, measured rather than assumed** — so whatever refutes them, it is not the completion. Which path *does* is not established here and is not claimed; what is measured is that both answer `unsat` with the certificate disabled, and that the hook runs only where the verdict would otherwise be `unknown` or a gate-downgraded `sat`. They are kept as guards for the *other* direction: they redden if a later pass moves the hook earlier. |

  **Amended 2026-09-28 (re-fix pass 12, decisions (40), (41)).** The hook now also runs at every `Sat` exit
  (model only, `#P2b-51`) and at MBQI's saturation point (`#P2b-60`), and the completion is a default **plus
  pinned points** where no constant certifies (`array_completion_certify/pinned.rs`: one fill query over a
  symbolic default and symbolic point values proposes, the evaluation pre-filter and the same certificate
  decide). `round4_pass11_recheck_pins::an_interpretation_that_is_not_constant_anywhere_is_certified_with_pins`
  is the inverted `…_is_declined_and_never_sat` (both scripts `sat`, each model replayed at all 128 points).
  The M11 guards were **not** re-derived: `a_completion_that_is_wrong_at_a_point_is_refuted_and_never_published`,
  `m4` and `m4b` are unsatisfiable, `check_core` answers them `unsat`, and an `Unsat` never reaches the hook on
  either path — all three stay green unchanged, and the mutation rows for this pass are in the CHANGELOG.

## #P2b-60

- [x] **#P2b-60 (2026-09-28) — SOUNDNESS, REGRESSION vs `c4b04b7` on bit-vector indices: an array pinned by a
  ground equality, read under a binder above the finite-expansion budget, answered a WRONG `sat`.** Found by the
  rebuilt `score.py` (its pin re-check CONFIRMED 19 of HEAD's 21 falsifying models on `$R/corpus/qmbqi120`, seed
  20260928). Minimal (`$R/corpus/named/wsat1`): `a1 = (store ((as const …) #b0) #b0000000 #b1)` beside `(forall ((i
  (_ BitVec 7))) (= (select a1 i) #b1))` — `unsat`; `c4b04b7` / 0.3.3 `unknown`, HEAD `sat`; also `wsat3`, `wsat4`,
  the pinned `q0000`, and wrong `sat`s on base and 0.3.3 too at `Int` (`wsat6`) and a declared sort (`wusort2`),
  `wreal` HEAD only. **Cause**: `mbqi::sat_certify`'s saturation test instantiated an essentially-uninterpreted
  universal over the index terms the model reads; an array is not an uninterpreted function (`store` and the
  array constant pin it where no read names it). **Fixed at the root 2026-09-28 (re-fix pass 12)**:
  `mbqi/sat_certify/unnamed_region.rs` completes an array index position's relevant set at the saturation re-check
  (every named index value, one representative per gap of every order a guard uses, every ground term of a declared
  sort, every constructor of an enumeration — and, re-fix pass 13, every constant of a declared sort the decided
  atoms mention, and a decline over any sort with no complete point set: `#P2b-64`); `Solver::certify_at_mbqi_saturation`
  concludes with a certified completion before paying for those instances (`$R/corpus/qeq120/q0033`: HEAD `sat`
  12 ms, no answer in 60 s with the instances every round, `sat` 8–84 ms now). All eight repros correct;
  `pinned_wrong_sat` 19 → 0 and 10 → 0 on the two corpora; six non-literal-index spellings (`$R/corpus/named/adv_*`)
  `unsat` (HEAD wrong `sat`). Pins `round4_pass12_unnamed_region_pins` (6). Cost: `bench/…/AUFLIA/array_update.smt2`
  7 / 24 → 11 / 37 refinement rounds / instances (verdict `sat`, re-attributed in `round4_pass8_budget_losses`).

## #P2b-61

- [x] **#P2b-61 (2026-09-28) — SOUNDNESS, pre-existing (`c4b04b7`, 0.3.3, HEAD): datatype constructors were not
  distinct as array indices.** `a = (store ((as const (Array Color Int)) 0) red 1)` beside `(= (select a green) 1)`
  — QUANTIFIER-FREE, `unsat` — answered `sat` on every build (`$R/corpus/named/wdt_ground.smt2`; the quantified
  `wdt.smt2` likewise). The read-over-write lemma builds the atom `(= red green)` itself and nothing told it the two
  constructors differ (`dt_axioms` skips the congruence instances of two different constructors, and the atom is
  minted after its scan). **Fixed at the root 2026-09-29 (re-fix pass 13):** `TermManager::mk_eq` folds an
  equality between two different constructors of one datatype to `false`, wherever it is built. Both repros and
  the QF core of `#P2b-64` (1) `unsat`; pin `round4_pass12_recheck_pins::p2b61_enumeration_constructors_are_distinct_as_array_indices`.

## #P2b-62

- [x] **#P2b-62 (2026-09-28) — a published model falsified its own GROUND assertion: a quantifier body's read
  over its bound variable was rendered as an array entry (REGRESSION vs `c4b04b7`).**
  `bench/z3_parity/benchmarks/AUFLIA/array_extensionality.smt2` with `(get-model)`: HEAD printed `a = (store
  (store ((as const …) 0) 0 0) 1 20)` beside `(assert (= (select a 0) 10))` and a `(get-value)` answering
  `10`; `extended_theories/AUFLIA/02_extensionality.smt2` the same. The body term `(select a i)` and the bound
  `i` had model entries (0 / 0); laid down as a read at index 0 it shadowed `(select a 0) = 10`. **Fixed
  2026-09-28 (re-fix pass 12)**: the renderer and its read-side twin skip every read mentioning a variable
  a quantifier binds and no declaration names (`Context::bound_only_variables`). Both scripts now print
  models their pinned re-check accepts (`$R/tools/sweep_models.py`).

## #P2b-63

- [x] **#P2b-63 (2026-09-28) — where the completion DECLINED, a `sat` still published a model falsifying its own
  universal.** `∀i:(_ BitVec 7). a[i] = #b1` beside an unrelated uninterpreted function, a fourth array, or an
  `exists` — `sat`, published with `a` false at most points. **Fixed 2026-09-29 (re-fix pass 13)**
  (`array_completion_certify/declined.rs`): whatever no universal reads through a binder keeps the candidate's
  value — a ground uninterpreted application whose arguments are literals or pinned scalars, and (when there are
  more than three arrays) an array read only at ground indices, each under a congruence check — and a
  positive-polarity `exists` is certified by a witness among the goal's points; the rest is completed and
  certified as before. All three shapes publish `a = ((as const …) #b1)`, replayed at 128 points: pin
  `round4_pass12_completion_pins::a_formerly_declined_completion_publishes_a_certified_model` (inverted).
  What still declines is `#P2b-58` (f); since re-fix pass 14 (decision (54)(i)) a declined completion keeps **no**
  falsifying model — the published-model certificate withholds a candidate that does not certify as printed (`#P2b-51`).

## #P2b-64

- [x] **#P2b-64 (2026-09-28) — SOUNDNESS, pre-existing on `c4b04b7`, 0.3.3, HEAD: `#P2b-60`'s family at two index
  sorts its repair did not reach.** (1) A datatype with a field: `a = (store ((as const (Array L Int)) 0) nil 1)`
  beside `(forall ((x L)) (= (select a x) 1))` answered `sat` (`dt_field.smt2`); its QUANTIFIER-FREE core
  `(= (select a (cons 0 nil)) 1)` (adversarial recheck 12) is `#P2b-61`'s mechanism. (2) A declared index sort whose
  second element exists only as the Skolem witness of `(exists ((x U)) (distinct x u))` (`usort_skolem.smt2`): the
  saturation test's points over a declared sort were the terms the goal *spells*, and the witness lives only in
  the encoded assertion. **Fixed 2026-09-29 (re-fix pass 13):** the QF core by `#P2b-61`'s fold (`unsat`); the
  quantified (1) by `unnamed_region` declining to certify over a datatype with a field, or any index sort it has
  no complete point set for (floating point, arrays, strings) — `unknown`, never `sat` (no value there is known to
  be unnamed; constructor representatives are the completeness lever, not needed for soundness); (2) by adding
  every constant of the sort the search's decided atoms mention to the points (`unsat`). Pins
  `round4_pass12_recheck_pins::{p2b64_a_constructor_with_a_field_is_distinct_as_an_array_index, p2b64_a_skolem_witness_is_in_the_index_universe}`.

## #P2b-65

- [x] **#P2b-65 (2026-09-28) — SOUNDNESS, a wrong `sat` on `c4b04b7`, HEAD and re-fix pass 12's tree: a disequality
  between two bound variables was accepted as an almost-uninterpreted guard.** `(forall ((i (_ BitVec 7)) (j (_ BitVec
  7))) (=> (not (= i j)) (distinct (select a i) (select a j))))` beside `(= (select a #b0000000) #b0)` — a pigeonhole,
  128 indices into 2 values, `unsat` — answered `sat` (`$R/corpus/named/pigeon2.smt2`). **Corrected by adversarial
  recheck 12:** 0.3.3 answers `unknown` at a bit-vector index and the correct `unsat` at an `Int` index with a
  bit-vector element (so that family was a **regression against the released crate**), and the same wrong `sat` at
  `Int`/`Bool`. Cause: `mbqi::sat_certify::eu_walk` descended `Not` without polarity, so `premise_safe` admitted `i ≠
  j` as the monotone var-var guard, which the projection does not preserve. **Fixed at the root 2026-09-29 (re-fix
  pass 13):** the walk tracks polarity (`not` and an inner `=>`'s left side flip it; an `ite` condition, a Boolean `=`
  / `distinct` operand, an arithmetic or uninterpreted operand make it unknown) and admits a var-var guard at positive
  polarity only. The recheck's 156 spellings (`recheck12/atk/pig2/`, release): **0 `sat`** (was 29), 81 `unsat` (every
  `Int` one), 75 `unknown` (every bit-vector-index one); `bv7_bool_neg_impl_2asserts` and its `_g`, refuted on pass 12
  only through the unsound classification (base / HEAD `unknown` or a wrong `sat`), are `unknown` now. Pins
  `round4_pass12_recheck_pins::p2b65_*` (inverted; the bit-vector-index pin asserts never-`sat`) and the unit test
  `mbqi::sat_certify::tests::premise_safe_admits_a_var_var_guard_at_positive_polarity_only`.

## #P2b-66

- [x] **#P2b-66 (2026-09-28) — SOUNDNESS, public API: `oxiz_sat::Solver::solve_with_assumptions` answered `sat`
  with a model falsifying one of its own assumptions, and its core dropped the partner of a complementary pair.**
  Found by re-fix pass 12 building decision (45) on it. (a) The assumptions were decided once, before the search
  loop; a conflict backjumped to `max(bt, 1)`, a limited restart went to level 0, and nothing re-decided what they
  undid, so a propagation could set an assumption false and the call still answered `sat` — pre-fix,
  `oxiz-sat/tests/assumption_retention.rs` fails at its first instance (seed 1, round 0: incremental `Sat`, fresh
  solver with the assumptions as units `Unsat`). (b) `analyze_final_core` keyed assumptions by *variable*, so with
  `x` and `¬x` both assumed the core came back `{¬x}` — satisfiable on its own (found by `oxiz-theories`'
  `bv_root_scoped_definitions` differential test, `a <u b` beside `a >=u b`). — **(fixed at the root:
  `oxiz-sat/src/solver/assumption_search.rs` is MiniSat's scheme — assumption `i` owns level `i + 1`, an empty one
  when already implied, and is re-decided after every backjump and restart; the core walk keys by literal. Learned
  clauses stay valid across calls, which (45) relies on; the debug-only fixpoint scan runs at the `Sat` exit rather
  than at every decision (a per-decision scan made the debug-build tests of an incremental caller two orders of
  magnitude slower). Four tests.)**

## #P2b-67

- [x] **#P2b-67 (2026-09-28) — `oxiz-opt` PMRES (WPM1) relaxed every core clause from its ORIGINAL body, so a core
  needing two violations reproduced itself for ever — masked by `#P2b-66`.** A clause relaxed twice could be paid
  only by its newest blocking variable; `pmres::tests::test_pmres_stratified` ("at most one of `x0..x2`", all three
  wanted) passed only because the pre-fix solver dropped an assumption and answered `sat`, and hit nextest's 180 s
  ceiling once `#P2b-66` was fixed. — **(fixed: the relaxed body keeps every blocking variable, WPM1 as published;
  346 of 346 `oxiz-opt` tests pass.)**

## #P2b-68

- [x] **#P2b-68 (2026-09-29) — SOUNDNESS, a wrong `unsat` in an incremental QF_AUFBV script (regression against
  crates.io 0.3.3; present on HEAD `c702310`).** Found by the adversarial recheck's `fuzz_abv.py` (seed 29092028,
  script 12503; 2 wrong `unsat` in 78,658 checks), delta-debugged to eight lines
  (`round4_pass12_recheck_pins::an_incremental_array_script_is_satisfiable_at_both_checks`): 0.3.3 `sat sat`,
  `c4b04b7` `sat unknown`, HEAD and pass 12 `sat unsat` (debug: the empty-clause `debug_assert!`). Cause: the operand
  `(ite (distinct a0 a0) #b1 i0)` was partly bit-blasted and never asserted, so no check followed;
  `bv_bridge::model_partition_lemma` read `#b1` off the stale snapshot as `#b0`, merged the two literals under their
  tautological reasons, and release turned the all-tautology conflict into the negation of the whole assignment.
  **Fixed at the root 2026-09-29 (re-fix pass 13):** a literal is bucketed by its literal value; a candidate the
  snapshot does not cover forces a refresh check (`BvSolver::snapshot_covers`); an all-tautology conflict is aborted
  (`unknown`), never a clause; `(distinct x x)` folds to `false` in the encoder (the second check `sat`). Guard
  `::a_stale_circuit_snapshot_never_manufactures_a_refutation`. `fuzz_abv.py` on the final tree (`$R/fix13b/camp/`): seed 29092028 0 wrong `sat` / 0 wrong `unsat` (was 0 / 2), seeds 29092914 / 29092915 / 29092916 (20,000 scripts each) 0 / 0, 0 bad models, 0 panics; seed 29092028 again on a release build with debug assertions on: 0 panics, identical counters.

## #P2b-76

- [x] **#P2b-76 (2026-09-30) — SOUNDNESS, a WRONG `sat` on every build, quantifier-free: two list values that differ
  two constructors deep were equated.** `(= (cons 1 (cons 2 nil)) (cons 1 (cons 2 (cons 3 nil))))` (`recheck14/atk/d04`),
  and through a constant (`d05`); pass 13's constructor fold refuted one level. — **(fixed at the root, re-fix pass 15,
  decision (66)), two layers.** `TermManager::mk_eq` decides an equality of two constructor applications at every depth
  (`oxiz_core` `dt_eq`: different constructors `false`, the same one the conjunction of its fields; iterative), and
  `solver::dt_refinement` checks every congruence class holding two constructor applications at the candidate model —
  distinct constructors refuted, the same one's unmerged fields equated — by a lemma justified by the closure's own
  explanation (`try_explain_eq`), charged to the refinement round budget. The one lemma that must MERGE two applications
  (`⋀ aᵢ = bᵢ ⇒ C(a⃗) = C(b⃗)`) keeps its atom (`TermManager::mk_eq_atom`). d04, d05, one level through a constant,
  four cells, two chained constants, a selector chain: `unsat`; `bench/qf_dt` unchanged. Pins
  `round4_pass14_recheck_pins::list_values_differing_two_constructors_deep_are_distinct` (inverted),
  `round4_pass15_fix_pins::constructor_injectivity_reaches_every_depth_and_route`; mutation `NO_CTOR_FOLD` +
  `NO_DT_REFINE`: d04, d05 wrong `sat`, d03 falsifying again.)**

## #P2b-77

- [x] **#P2b-77 (2026-09-30) — REGRESSION of re-fix pass 13 (0.3.3, `c4b04b7`, HEAD and pass 12 correct): two list
  values sharing a prefix printed as one.** `l1 = [1,2]`, `l3 = [1,2,3]` printed `l3 = (cons 1 (cons 2 nil))`
  (`recheck14/atk/d03`). Bisected in an isolated copy to pass 13's constructor fold in `mk_eq` (switched off: correct):
  the fold removed `(= nil (cons 3 nil))` from the selector-congruence lemma that had kept `l1 ≠ l3`, nothing else could
  refute `l1 = l3` (`#P2b-76`), the search set it true and the model builder printed one value for the class. — **(fixed
  by `#P2b-76`'s two layers; pin `round4_pass14_recheck_pins::two_list_values_sharing_a_prefix_print_two_values`,
  replayed cell by cell and as the asserted equalities.)**

## #P2b-70

- [x] **#P2b-70 (2026-09-29) — COMPLETENESS, a verdict HEAD `c702310` reaches and re-fix pass 12's second half does
  not: a bit-vector order guard over an array pinned by a ground equality.** `$R/corpus/named/wsigned.smt2` (`a =
  (store (… #b0) #b0000101 #b1)` beside `∀i. i <s 0 ⇒ a[i] = #b1`) and its `bvuge` / `bvugt` / `bvule` / `(not bvult)`
  spellings (`$R/fix13b/g_{uge,ugt5,ule,nult}.smt2`) are `unsat`: HEAD and Fixer A's half answer `unsat`, Fixer B's copy, pass 12's
  final tree and this one `unknown` (`c4b04b7`, 0.3.3 `unknown`: not a (24a) base-decided loss). Isolated in Fixer B's
  mutation copy (`$R/probes/probe_isoBmut`): only `OXIZ_MUT_FRESH_NEQ` (`assert_neq` minting fresh variables per call
  again) brings back `unsat`; the other 14 switches, `NO_CANON` among them, leave `unknown` — decision (45)'s memoised
  `assert_neq` gate, as a trajectory change. Working explanation (read from the code, not traced): the guard is
  outside `sat_certify`'s fragment, so only MBQI's counterexample search can refute it, and its candidates for a
  bit-vector variable are the values the candidate model gives terms of that sort
  (`counterexample::build_candidate_lists`); rounds 5–16 add no lemma (`OXIZ_TRACE_AR`), and a ground read inside the
  region makes it `unsat`. Lever: candidates from the literals the body compares a bound variable with (±1), as
  `sat_certify::augment_guard_grounds` does — a trajectory change for the sweep, so its own pass. No in-tree pin yet.
  — **(FIXED at the root 2026-09-29, re-fix pass 14, decision (60)). Traced rather than read:** the counterexample
  search's candidate lists for `i` were `{#b0000000, #b0000101}` (or the injected binder witness alone), both outside
  the guard, in every round. `mbqi::counterexample::guard_points` adds, where the model's own values find no
  counterexample, the points the body's guards name — every literal a bound variable is compared with and its `± 1`
  (mod `2ʷ`) — as extra combinations; a counterexample there is an ordinary instance of the asserted universal, and the
  phase never changes whether a round counts as fully evaluated. All five spellings `unsat` again (the memo stays: the
  root was the candidate set, not the gate). Pin `round4_pass13_recheck_pins::p2b70_a_bit_vector_order_guard_over_a_pinned_array_is_refuted`
  (recheck 13's pin, inverted); `fuzz_qc` seed 29093020 on the final tree `unsat` 922 → 1,061 and `unknown` 502 → 73
  with 0 wrong `unsat` (every `unsat` re-judged on its ground twin); `bench/` sweep 0 verdict differences.)**

## #P2b-71

- [x] **#P2b-71 (2026-09-29) — QUANTIFIER-FREE, pre-existing (0.3.3, `c4b04b7`, HEAD): numeric constants no theory
  valued printed one value for congruence classes the search kept apart.** `(= (f k) 5)` beside `(= (f j) 6)` over
  `Int` printed `k = j = 0` and `f` constantly `5`; `(select a k) = 5` beside `(select a j) = 6` printed `a = (store
  ((as const (Array Int Int)) 0) 0 5)` and `(get-value ((= k j)))` answered `true`; the same over `Real`
  (`recheck13/atk/g08`, `g09`, `g11`, `g13`). The model builder filled every such term with the sort default (an
  arithmetic column nothing constrained, an array index or argument no theory read), and every renderer printed it.
  The same gap dropped an `exists`'s witness (`#P2b-51`): the Skolem constant of `∃i. a[i] = #b10` has no value, so
  the read `a[sk] = #b10` named no position and `a` printed as the constant `#b00`. — **(fixed at the root, re-fix pass
  14: the model builder records such an entry as a default (`Model::set_default`: an arithmetic value of a term in
  no parsed arithmetic atom, an index leaf or bit-vector no circuit valued), and `context::model_fmt::published` gives
  every class of an `Int` / `Real` / bit-vector variable that no theory valued — a declared constant the congruence
  closure knows, an index or argument, an asserted `exists`' Skolem constant — the value a member already prints
  when no other class uses it, else the smallest fresh value of its sort, one per class; `(get-model)`, `(get-value)`
  and the interpretations all read that one model. g08 `k = 1`, `j = 2`, g11 / g13 `k = 0`, `j = 1`, `(= k j)` answers
  `false`. Pin `round4_pass13_recheck_pins::numeric_constants_seen_only_as_arguments_get_distinct_values`
  (inverted). Incremental quantifier-free models on the final tree: `fuzz_abv.py` fresh seed 29093030, 20,000 scripts
  / 78,511 checks, 0 bad models of 50,523, 0 bad cores, 0 wrong `sat` / `unsat`.)** **REOPENED 2026-09-30 by adversarial
  recheck 14 (decisions (64), (68)) and RE-CLOSED by re-fix pass 15.** Nested application classes still printed one value
  (`recheck14/atk/m06`: `(f (g k)) = 1`, `(f (g j)) = 2`, `g` constantly `0`; `m09` over `Real`), and pass 14's fresh value
  for `k` re-keyed `f`'s table so `(f j)`'s entry was lost (`r03`, a regression: `c702310` printed a correct model).
  **Fix (`context::model_fmt::mint`):** every congruence class no theory valued — application and `select` members
  included, innermost first — gets a value keyed by the POINT it reads (the function and its argument values, or the
  array's class and index value): a point a theory valued, or an earlier member, shares that value, so one point prints
  one value and every table is a function; a quantifier-free `sat` whose fresh values make the printed model (constants
  and tables, read back) falsify an assertion prints the solver's own model (`printed_check`); and at a
  quantifier-free candidate two applications whose arguments the theories valued alike and whose results they valued
  differently get the Ackermann lemma and the search runs again (`solver::uf_consistency`, model-based theory
  combination — `g14a/s00303`, `s00330`, `s00411`). m06, m09, r03 replay; `(get-value ((= (g k) (g j))))` is `false`.
  Pins `round4_pass14_recheck_pins::{quantifier_free_models_replay_on_their_script,
  get_value_of_a_comparison_over_separated_classes_is_false}` (inverted); mutation `NO_MEMBERS`: m05, m06, m09, r04, r05
  falsifying again, with `NO_QF_CHECK` also r03.

## #P2b-72

- [x] **#P2b-72 (2026-09-29) — QUANTIFIER-FREE, pre-existing (0.3.3, `c4b04b7`, HEAD): a datatype-indexed array's
  `(get-model)` dropped every entry (the datatype twin of `#P2b-43`, which fixed the uninterpreted-sort index).**
  `(assert (= (select a red) 1))` over `(Array C Int)` printed `a = ((as const (Array C Int)) 0)` while
  `(get-value ((select a red)))` answered `1` (`recheck13/atk/e07`, `e08`, `e09`). `array_model.rs`'s
  `index_position_string` named a position only for a literal or an uninterpreted-sort class. — **(fixed: a
  constructor value — a constructor applied to values — spells itself (`is_constructor_value`), and the read-side
  twin takes a constructor index as its own value; e08 prints `(store ((as const (Array C Int)) 0) red 1)` and
  `(get-value ((select a green)))` answers `0` instead of echoing. Pin
  `round4_pass13_recheck_pins::a_datatype_indexed_array_model_prints_its_entries` (inverted).)**

## #P2b-74

- [x] **#P2b-74 (2026-09-30) — MODEL, pre-existing; filed by re-fix pass 14 so that no hole pin points at a closed item
  (decision (46)): a `(get-value)` beside `(get-model)` gives one uninterpreted function two interpretation entries at
  the same evaluated arguments.** The `QF_AUFLIA` script `round4_pass2_recheck_pins::a_get_value_query_beside_get_model_breaks_the_one_reading`
  carries verbatim (`f` over `(f 0)`, `w` and a read of a `store` over the array constant, `g 1 3`, then `(get-model)`
  and `(get-value (j k v w (f j)))`) trips `Context::get_func_interp_raw`'s `debug_assert!` in a debug build; the same
  script without the `(get-value)` does not. The pin was written by the round-4 recheck pass 2 against the `#P2b-34`
  amendment, which the condensed record no longer names. Measured on re-fix pass 14's final tree: the debug
  assertion still fires (nextest subset, dev profile), and the release build prints `f` constantly `0`, `w = 1`,
  `j = k = 3` — a model that satisfies the script. No fix this pass. **Re-fix pass 15 (decisions (67), (68)):** the
  mechanism is theory combination — the congruence closure keeps `f(y)`, `f(4)` apart while the arithmetic gives `y`
  the value `4`, so the candidate is not a function. Quantifier-free, the lever landed (`solver::uf_consistency`, the
  Ackermann lemma at the candidate); quantified, the published-model certificate reads such a table as printed and
  withholds it where it falsifies (`u04`, `QUANTIFIED_BENCHMARK` of `scope_rebase_tests`).
  The same lemma on the quantified path was built and measured and did not land: it made
  `pr30_soundness::..._ground_diseq_is_not_sat` answer `unsat` (correct; pinned `unknown`), but it kept adding
  original clauses on repeated checks of an unchanged goal (`two-quant-arith` 47 → 62 → 67 → 72 over 40 forced
  re-runs) and ran `scope_rebase_tests::a_check_leaves_the_mbqi_search_state_where_it_found_it` past 180 s.
  — **(CLOSED 2026-09-30, re-fix pass 15.)** On the pin's own script the lemma resolves the collision: the debug
  assertion no longer fires and the published model replays; pin inverted as
  `round4_pass2_recheck_pins::a_get_value_query_beside_get_model_keeps_the_one_reading`. A quantified goal's table
  is read as printed (the first entry at a tuple) and certified — withheld where it falsifies — so the printer's
  two-entry assertion stands for quantifier-free goals only. On the quantified path the lemma did land where it is
  cheap — only at a `sat` exit whose candidate the ground gate refuses (`assert_exit_consistency_lemmas`): `g14a/s00761`
  and `s01200` answered `unknown` there on this tree where `c702310` and pass 14 answered `sat` (a trajectory of
  `#P2b-75`), and `pr30_soundness::test_pr30_quantifier_trigger_function_ground_diseq_is_not_sat` now answers `unsat`
  (z3 agrees; pinned `unknown` before, inverted); `scope_rebase_tests` unchanged (mutation `NO_EXIT_ACK`).

## #P2b-75

- [x] **#P2b-75 (2026-09-30) — SOUNDNESS, a WRONG `sat` on every build (0.3.3, `c4b04b7`, `c702310`, re-fix pass 14):
  a universal whose guard holds on an interval open at its boundary.** `(assert (forall ((q Int)) (=> (> q 7) false)))`
  (`recheck14/atk/w01`), `(=> (not (= x 3)) false)`, `(=> (< x 10) (= m 1))` beside `m = 2`, `(=> (> x m) false)`, two
  guarded universals over `f`, `(=> (> q 1) (< (f 2) (f q)))`; z3 `unsat` on all. Adversarial recheck 14, g14a: 8 of the
  tree's 3,408 decided checks. Re-fix pass 15 found the same at `Real` (`1.0 < q < 2.0`), a bit-vector `≠` over 256 points,
  a declared sort (`¬(x = a)` beside `(distinct a b)`) and a datatype with a field (`¬(x = nil)`). — **(fixed at the root,
  re-fix pass 15, decision (65), `mbqi::sat_certify`).** Traced: the relevant set held the ground side `t` of each guard
  `x ⊕ t` alone, and `>`, `<`, `≠` are false at `t`, so every instance was vacuous and the set saturated. Now: `t ± 1` as
  TERMS beside every guard ground (`guard_neighbours`; `(+ m 1)` stays one point however the model moves `m`) — with
  `π(v) = max{s ∈ S : s ≤ v}` that preserves every guard over `Int` and bit-vectors; over `Real` the midpoint of every
  pair of guard grounds (`real_regions`: every open region between two guard values gets a point), a goal ordering two
  `Real` bound variables keeping the monotone projection and declining only beside a guard open at its lower end; a
  compared sort no term brackets gets `unnamed_region`'s points (every ground term of a declared sort, a value no named
  term denotes over a datatype with a field — `datatype_points` — where pass 14 declined); `f`'s argument spelled only
  inside a binder (`(f 2)`) is named. The guard walk descends every sub-term of the premise. All spellings `unsat`,
  their satisfiable controls `sat`. Pins `round4_pass14_recheck_pins::strict_order_guards_over_an_int_binder_are_refuted`
  (inverted), `round4_pass15_fix_pins::{guards_at_every_sort_get_an_instance_past_their_boundary,
  satisfiable_guarded_goals_stay_decided_and_their_models_replay}`; mutation (`OXIZ_MUT15_NO_NEIGHBOURS`: 23 wrong `sat`
  on the 154-script attack set; `NO_REAL_MIDPOINTS` 2; `NO_COMPARED_SORTS` 2). Cost: `AUFLIA/array_update` 16 / 53 →
  **11 / 43** rounds / instances (100 → 52 conflicts; attributed with `NO_NEIGHBOURS`). The neighbours also exposed a
  latent leak, fixed at the root before landing: the certificate built and interned the instances of the quantifiers
  that came first and then declined on a later one (`g14a/s00942` 0.1 s → 10.6 s, `s00756` 1.8 → 51 s; `NO_NEIGHBOURS`
  and a neighbours-interned-but-unused A/B attributed it); every eligibility decision now precedes any instance
  (`collect_fragment_instances`; pin `mbqi::sat_certify::tests::a_declined_round_interns_no_instance`, the arena's size
  across a declined round; mutation `NO_ELIG_FIRST`). Traced
  consumer: E-matching (`oxiz-core` `ematching::quantifier_inst::match_round`) matches every trigger against EVERY term
  in the arena, so the discarded bodies were ground terms for it — which is also how `c702310` refuted `qeq120/q0005`
  (the instance at `i = c` of `(select (store a1 i v) c)`), lost with the leak. The counterexample search now takes a
  read's index as a guard point of every write under it (`guard_points`, `#P2b-70`'s probe): `q0005` `unsat` again
  (pin `round4_pass15_fix_pins::a_read_over_a_write_at_the_bound_variable_is_refuted_at_the_read_index`; mutation
  `NO_ROW_POINTS`). E-matching over the whole arena stays as it is: what a trigger may match is still a side channel
  of what else was interned.)**

## #P2b-79

- [x] **#P2b-79 (2026-09-30) — SOUNDNESS, a WRONG `sat` on every build (0.3.3, `c4b04b7`, `c702310`, re-fix pass 14):
  an `Int` term was solved over the reals under `(set-logic ALL)` or no logic.** `(declare-const x Int) (assert (= (* 2 x)
  1))` answered `sat` (z3 `unsat`), and so did `0 < 3x < 3`, `2x = 2y + 1`, `x < y < x + 1`, `2.0·to_real(x) = r = 1.0` and
  each beside a universal; `QF_LIA` answered `unsat`. Found by re-fix pass 15 tracing `g14a/s00184` (its `sat` lost to a
  model gate refusing `j`'s relaxation value `-2/3`, which the model builder printed as its numerator `-2`). Traced:
  `Solver::set_logic` picks the arithmetic mode from the logic's name and `ALL` keeps the default `LRA`, which has no
  integer variable at all. — **(fixed at the root, re-fix pass 15.)** `ArithSolver::mark_int_term` (called by the encoder
  at every site that registers an `Int`-sorted term) and `ArithSolver::check_integrality`, run by the theory manager once
  per final check after Nelson-Oppen combination (`theory_manager/integrality.rs`): the same branch-and-bound `LIA` mode
  runs, over the marked terms, its leaf snapshot holding every variable (the reals with the leaf's `δ` instantiated); a
  strict bound over an all-`Int` row is an integer bound; an all-`Int` equality gets the divisibility and Diophantine
  checks (`oxiz-theories` `arithmetic/solver/mixed.rs`). Three variants built and measured on `g14a` (`s00942` /
  `s01200`): branch-and-bound in every check 6.6 s / > 60 s; at the final check only, strict all-`Int` rows keeping
  their `δ` 33 s / > 60 s; landed (final check, rows tightened) 0.5 s / 1.0 s. The tightening (what `LIA` mode does)
  is a trade on the final code (A/B `OXIZ_DBG15_NO_TIGHTEN`): with it `s00942` 0.4 s (21 s without), `s01200` check 2
  and `s00756` check 1 `sat` (`unknown` without); without it `s02265` 0.2 s (no answer in 130 s with it), `s01963` 0.2
  s (38 s) and `s02748` `sat` (`unknown`) — see decision (24a)'s pass-15 leg. Pins
  `round4_pass15_fix_pins::{an_int_term_is_an_integer_under_every_logic, a_mixed_integer_goal_publishes_an_integral_model}`
  (every kind of `Int` leaf, `push`/`pop`, `check-sat-assuming`, a universal); mutation `NO_MIXED_INT` (the marks off).)**

## #P2b-85

- [x] **#P2b-85 (2026-09-30) — REGRESSION of re-fix pass 15 (printed model): a record field over an application
  printed a model falsifying its script.** `p = (mk (+ (q 3) 1) 5)` printed `p = (mk 0 5)` beside `q` constantly `0`
  (`recheck15/atk/dt/pq_model_regression`; `c702310` and pass 14 printed `q` constantly `-1`). Traced (class dump at each
  refinement round): after `solver::dt_refinement`'s re-solve the arithmetic held `(px p) = 1`, `(q 3) = 0`, but the
  model builder rebuilt the literal `(mk (+ (q 3) 1) 5)` with its compound field filled by the sort default `0` —
  `scalar_value` returned `None` for a term with no theory variable of its own. `c702310` was right by coincidence (its
  candidate had `q = -1`, where the default is the value). — **(fixed at the root, re-fix pass 16):** a compound field is
  folded from its leaves (`model_builder::folded_field_value`), and an `Int` field's value is never a numerator of a
  fraction. Pin `round4_pass15_recheck_pins::a_record_field_over_an_application_prints_a_model_that_replays` (inverted);
  mutation `OXIZ_MUT16_NO_FIELD_FOLD`.

## #P2b-86

- [x] **#P2b-86 (2026-09-30) — REGRESSION of re-fix pass 15 (verdict): a false tester after a `push`/`pop` answered
  `unknown`.** `recheck15/atk/dt/u03_min.smt2` (`gen_dt.py` `d00274`): the third check is `unsat` on `c702310`, pass 14
  and z3, `unknown` on pass 15. Traced: the unknown is `model_blocking`'s downgrade — the SECOND check (after the `pop`)
  refused five candidates and blocked them, and a later `Unsat` under a blocking clause is `unknown`. The refusals were
  spurious: `model_builder::separate_dt_value` re-valued `(hd l2)` from the tableau's `2` to `3` to pull two datatype
  values apart, but `(hd l2)` sits in the datatype axioms' congruence atoms and the search had kept it below `(hd (h
  2))` (`(> (hd (h 2)) (hd l2))` true, `(= …)` false), so the gate's negated-equality check refused the model. —
  **(fixed at the root, re-fix pass 16):** the separation repair keeps a bump only when no numeric disequality the
  search decided becomes an equality (the gate's own check), trying the next value first
  (`model_builder/separation.rs`). The third check's tester also folds to
  `false` at construction (`#P2b-82`), so each fix alone keeps `unsat`: mutation `OXIZ_MUT16_NO_SEP_CHECK` with
  `NO_DT_FOLD` gives `unknown` again, either alone does not. Pin `round4_pass15_recheck_pins::a_false_tester_after_a_pop_is_refuted` (inverted). Recorded,
  not changed: a blocking clause is a search restriction that outlives its check at the same level (issue #40's
  design), so one spurious refusal costs every later `unsat` of the scope.

## #P2b-87

- [x] **#P2b-87 (2026-09-30) — REGRESSION of re-fix pass 15 (verdicts, against `c702310` and pass 14): `#P2b-79`'s
  integrality machinery changed the MBQI loop's trajectory.** `(forall ((i Int)) (distinct i j))` `unknown` after 176
  conflicts (`unsat` in 14 on `c702310`); `gen_mix.py` seed 30093153: 25 checks in 12 of 324 quantified scripts;
  `gen14.py` seed 30093155: 7 checks; `gen_guard.py` seed 30093152: `g01406`, `g00076`; `g14a/s00670` — every one
  restored by `OXIZ_MUT15_NO_MIXED_INT`; `forall_neq` needs both the strict-row tightening and the branch-and-bound off,
  `m00128` also the rounded `value()`. — **(fixed at the root, re-fix pass 16, decision (72)(b)):** a quantified goal
  defers the integrality (`ArithSolver::set_defer_integrality`, `oxiz-theories` `arithmetic/solver/mixed.rs`): every
  MBQI round searches the relaxation exactly as `c702310` did and the counterexample search reads its values; only where
  the loop would conclude — a `sat` exit with no pending instance, or the give-up after ten rounds — does a marked `Int`
  term the relaxation left fractional take the deferral back, and the loop searches on with `#P2b-79`'s machinery
  (`solver::integrality_exit`). Every named script is decided as on `c702310` — `forall_neq` `unsat` in 14 conflicts
  again, `gen_mix.py` 0 of its 25 checks lost (and 0 wrong, 0 falsifying of 4,671 models), `g14b` / `gen_guard` /
  `g14a/s00670` restored; `(= (* 2 x) 1)` under `ALL` and no logic, beside a universal, and `(forall ((i Int)) (= (* 2
  (f i)) (+ (* 2 i) 1)))` stay `unsat`, every `#P2b-79` pin is green. Pins `round4_pass15_recheck_pins::guarded_universals_head_refutes_are_refuted_again`
  ((a), (c) inverted), unit tests `arithmetic::solver::tests::deferred_integrality_*`; mutation `OXIZ_MUT16_NO_DEFER_INT`.
  Its cost, named in decision (24a)'s pass-16 leg: taking the integrality back where the loop gives up turned
  `c702310`'s quick `unknown` on `gen14.py` seed 30093155 `s00262` into a search, and the script's second check
  (`unsat` on `c702310`) gets no answer in 130 s.

## #P2b-53

- [x] **#P2b-53 (2026-09-21) — COMPLETENESS, pre-existing: a `forall` over an index sort the finite expansion
  declines answered `unknown` to a read-over-write at the binder's own index.** `(declare-const a (Array (_ BitVec
  7) (_ BitVec 1)))` and `(assert (forall ((i (_ BitVec 7))) (distinct (select (store a i #b1) i) #b1)))` —
  `rk6/atk3/p4_bv7_two_stores.smt2`, whose name outlived its second `store` — is **unsatisfiable**: the read-over-write
  axiom makes the `select` `#b1` at every `i`, so the body is false everywhere and no enumeration of the 128-point
  index domain is needed, only the rewrite under the binder. `c4b04b7` (crates.io 0.3.3) and `00add07` (the pass-6
  checkpoint, where this entry was opened) both answer `unknown`. **CLOSED 2026-09-21 by re-fix pass 8/9
  (`ff3965d`) as a side effect** of `binder_row`'s read-over-write expansion becoming polarity-complete and its
  decline guards no longer evadable (`#P2b-54`, `#P2b-55`): the tree answers `unsat` in 32 ms through the release
  CLI, re-measured 2026-09-22 by the gatekeeper, who also completed this entry — it had been cut off after
  "the finite expansion" at `00add07` and never re-measured by the pass that closed it. Regression pin:
  `round4_pass7_recheck_pins::a_read_over_write_at_the_binder_index_is_refuted_above_the_expansion_budget`.
  **What this is NOT.** Not `#P2b-58`: that item's scripts are *satisfiable* ones the base decides and MBQI cannot
  certify above the expansion budget; this one is refuted by rewriting and never needed a model. Not `#P2b-50`: the
  index sort's cardinality is pinned at `2^7`.

## #P2b-82

- [x] **#P2b-82 (2026-09-30) — SOUNDNESS, a WRONG `sat` on every build (0.3.3, `c4b04b7`, `c702310`, re-fix pass 14, 15),
  quantifier-free: a selector over a constructor application inside an uninterpreted argument.**
  `(distinct (f 1) (f (hd (cons 1 l1))))` answered `sat` (z3 `unsat`; `recheck15/atk/dt/sel1`–`sel3`, `gen_dt.py` seed
  30093154 `d00238`), while `(distinct 1 (hd (cons 1 l1)))` was refuted: the datatype axioms' selector-over-constructor
  lemma decided the atom but the congruence closure met `(f (hd (cons 1 l1)))` as an application over an opaque
  selector node. The same gap withheld correct models (`wh2`, `g14b/s01780`: the certificate could not evaluate `(fst
  (mk k blue))`) and echoed `(fst (mk j red))` from `(get-value)`. — **(fixed at the root, re-fix pass 16, decision
  (73)):** `oxiz_core` `ast/manager/dt_fold.rs` folds a selector of the constructor that built its argument to the field
  and a tester of a constructor application to `true` / `false` where the term is built, substitution rebuilds through
  the same entry points, and `TermManager::fold_constructor_accessors` folds a term built before (the printed-model
  certificate and `(get-value)`'s printed reading call it); `(hd nil)` stays a selector node. dt6 (z3-judged, cap 10 s,
  the final tree, `$R/fix16/final/dt6_probe_tree`): 0 wrong `sat` (`c702310` 13), 45 timeouts (96). **Decision (73)'s "no
  printed-model regression" cannot hold for it, or for any trajectory change on this corpus (`#P2b-88`):** the fold alone
  turned 24 checks `c702310` and pass 14 both got right into falsifying models, the tree without it 31; with decision
  (72)(a)'s net 17 remain — 15 withheld, 1 falsifying (`d00239#0`), 1 past the cap (`d00364#2`), named under (24a)'s
  pass-16 leg — and dt6 prints 4 falsifying models of 549 (`c702310` 171 of 507) and withholds 97. Pins
  `round4_pass15_recheck_pins::a_selector_over_a_constructor_inside_an_application_is_refuted`,
  `::a_correct_model_over_a_selector_of_a_constructor_is_printed` (inverted), unit tests in `dt_fold.rs`; mutation
  `OXIZ_MUT16_NO_DT_FOLD`.

## #P2b-83

- [x] **#P2b-83 (2026-09-30) — SOUNDNESS, a WRONG `sat` on every build, quantifier-free: a cycle through an
  uninterpreted application two constructors deep.** `(= (h 1) (cons x (cons 2 (h 1))))` answered `sat` (z3 `unsat`;
  `recheck15/atk/dt/cyc4`, `cyc6`; `gen_dt.py` `d00315`); the dev profile's datatype-model net fired on the candidate.
  Traced (lemma dump of the isolated copy): the encoder purifies the numeric argument (`encode::numeric_purification`),
  so the atom the search holds is `(= (h n) (cons x (cons 2 (h n))))` beside `n = 1`, while the datatype axioms' size
  ordering is instantiated over the terms as written, `(h 1)`; one constructor deep `#P2b-76`'s injectivity lemmas
  happened to close the gap. — **(fixed at the root, re-fix pass 16):** the occurs check Z3 runs, over the candidate's
  congruence classes (`solver::dt_refinement::occurs`): a class holding `C(…, a, …)` whose datatype argument lies in
  another class is an edge, and a cycle's merges `E` get the lemma `¬E` (acyclicity; valid at every level). Pin
  `round4_pass15_recheck_pins::a_cycle_two_constructors_deep_through_an_application_is_refuted` (inverted, both profiles);
  mutation `OXIZ_MUT16_NO_OCCURS`.

## #P2b-84

- [x] **#P2b-84 (2026-09-30) — MODEL, every build since `c4b04b7` (0.3.3 omits the table), quantifier-free: a function
  into a datatype, an enumeration or an uninterpreted sort printed one value at two points its arguments' theory
  valued alike.** `(h x) = [1]`, `(h y) = [2]`, `x, y ∈ [0, 5]` printed `x = y = 0` beside `h` constantly `[1]`
  (`recheck15/atk/dt/d01`, `d08` (range `U`), `d10` (enumeration), `d11` (`h : L → L`)); the dev profile's table printer
  asserted. `solver::uf_consistency::theory_class_value` answered nothing unless a class member was a scalar literal, so
  no Ackermann lemma was built for such a result. — **(fixed at the root, re-fix pass 16):** a result class is keyed by
  the value it prints — a scalar literal, a datatype value (a literal constructor application of the class, or the
  model's reconstructed value), and for an uninterpreted sort its congruence class (the printer gives every class its own
  witness). Pins `round4_pass15_recheck_pins::a_function_into_a_datatype_is_kept_a_function_at_two_separated_points`
  (inverted, both profiles), `round4_pass16_fix_pins::a_function_into_or_over_a_declared_sort_is_kept_a_function`;
  mutation `OXIZ_MUT16_NO_CLASS_KEY` (`d01`, `d02`, `d03`, `d08`, `d10` falsifying again).

## #P2b-89

- [x] **#P2b-89 (2026-10-01) — MODEL, every build (0.3.3, `c4b04b7`, `c702310`, re-fix pass 14, 16), quantifier-free:
  an array whose element sort is a datatype, an enumeration or an uninterpreted sort, read at two indices the
  arithmetic valued alike, printed one entry.** The array twin of `#P2b-84` (adversarial recheck 16's unrecorded
  blocker). `(declare-const a (Array Int L))`, `x, y ∈ [0, 5]`, `(= (select a x) (cons 1 nil))`, `(= (select a y) (cons 2
  nil))` printed `x = y = 0` beside `a = (store ((as const (Array Int L)) nil) 0 (cons 1 nil))` (`recheck15/atk/dt/d05`;
  `recheck16/atk/arr/a_enum` printed `a` constantly `red`, `a_usort` constantly `@uc_U_0`; the `Int`-element twin was
  right). Two gaps: the read-congruence family (`Solver::build_index_congruence`) compared two reads by `EvalVal`, which
  a datatype or uninterpreted read has none of, so the violated lemma `x = y ⇒ a[x] = a[y]` was never built; and a read of
  an uninterpreted sort has no model entry, so its array printed one constant. **Fixed at the root (re-fix pass 17,
  decision (79)(b)):** such reads are compared by the value their class prints, keyed as `#P2b-84` keys results
  (`Solver::theory_class_value`), and the array printer and `(get-value)`'s reader give an uninterpreted read its class
  witness. No recorded corpus holds such an array (`gen_dt.py`, `gen14.py` arrays are `Int`-, bit-vector- or
  datatype-INDEXED with scalar elements), so the fix is measured on the pins, the `bench/` sweep and the 716-script
  battery. Pins `round4_pass16_recheck_pins::an_array_into_a_datatype_read_at_two_free_indices_keeps_both_entries`
  (inverted; list, enumeration and uninterpreted elements, each replayed); mutation `OXIZ_MUT17_NO_ARRAY_CLASS_KEY` (list
  and enumeration withheld by the net), `OXIZ_MUT17_NO_U_ENTRIES` (uninterpreted withheld), both with
  `OXIZ_MUT17_NO_VALUE_SORT_NET` (all three falsifying, as on every earlier build).
  **REOPENED (adversarial recheck 17) and closed again at the root (re-fix pass 18, decision (86)).** One level down:
  `a : (Array Int (Array Int U))`, `x, y ∈ [0, 5]`, `a[x][0] = u1`, `a[y][0] = u2`, `u1 ≠ u2`
  (`$R/recheck17/atk/p89/b05`) printed `x = y = 0` beside the one entry `a[0][0] = @uc_U_0` on every build (0.3.3,
  `c4b04b7`, `c702310`, re-fix passes 14, 16 and 17; z3: falsifying, the script `sat`). Pass 17's key covered an
  element sort that IS a datatype or an uninterpreted sort, and here the element is an array; and pass 17's net read
  no array value, so the read of a read stayed open and the model was published. **Fixed:** a read whose element
  sort CONTAINS such a sort at any depth is compared by class, an array-valued read — and an array-valued function
  result — by its congruence class (`Solver::theory_class_value`), so the violated lemma `x = y ⇒ a[x] = a[y]` is
  built and the two inner reads then merge; the printed model's parse-back reads witnesses and constructors through
  array sorts at any depth and through declared functions' ranges and arguments; and the exact value reader
  (`printed_eval::ground_values`) reads array values — store chains, constant arrays, nested ones — so the honesty net
  (decision (85)) reads such a model false. Measured on recheck 17's battery (13 `sat` shapes and an `unsat` twin) and
  re-fix pass 18's sixteen nested shapes (`$R/fix18/atk/nest/`, z3-judged): every model printed holds but `n11`'s —
  the same shape over `Int` elements, `#P2b-81`'s (no datatype, so not checked) and pinned as its hole; pass 17 printed
  falsifying models for `b05`, `n01`-`n03`, `n05`, `n08`, `n10`, `n12`, `n13` and `n16` (a function into `(Array Int
  C)`). No recorded corpus holds an array of arrays over such a sort (`bench/` has two over `Int`, `qf_a/array_07`,
  `array_08`), so the fix is measured on the pins, the batteries and the `bench/` sweep (0 response differences against
  pass 17). Mutation (`$R/probes/probe_iso18`): `OXIZ_MUT18_NO_NESTED_CLASS_KEY` turns `b05`, `n01`-`n03`, `n05`, `n08`, `n10`, `n12`-`n14` and `n16` into `unknown`
  (the honesty net reads the printed array values false); with `OXIZ_MUT18_NO_ARRAY_VALUES` as well they print
  falsifying models again (`n14` withheld; `n16` `unknown` until `OXIZ_MUT18_NO_HONESTY_NET` is added too). Pins
  `round4_pass17_recheck_pins::an_array_of_arrays_into_an_uninterpreted_sort_keeps_both_entries` (inverted),
  `round4_pass18_fix_pins::arrays_of_arrays_over_value_sorts_print_models_that_hold`,
  `::an_int_only_array_of_arrays_still_prints_one_entry` (`#P2b-81`'s hole).

## #P2b-90

- [x] **#P2b-90 (2026-10-01) — SOUNDNESS, a WRONG `sat` on every build (0.3.3, `c4b04b7`, `c702310`, re-fix passes 14,
  16 and 17): a tester, a selector or an equality applied to a datatype-sorted `ite` was not tied to the `ite`'s value;
  and a datatype term only a quantifier body names had no datatype axiom.** Found by adversarial recheck 17 on its
  fresh `gen_dt.py` seed 30100192 (`d00370`#0: `l2 = (cons (px p) nil)`, the pushed `(= (ite ((_ is cons) l2) (tl l2)
  nil) l1)` makes `l1 = nil`, so `(< 0 (ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) … 0))` is false; z3
  `unsat`) — 1 of the 1,159 checks of that seed, 0 of the 2,850 of the four `gen_dt.py` seeds (24a) covered. Reductions
  (`$R/recheck17/atk/`): `m15` — `c` false beside `(< 0 (ite ((_ is cons) (ite c l1 nil)) 1 0))` — and `m10` — `l1 =
  nil` beside the same tester over `(ite ((_ is cons) l1) (tl l1) nil)` — `sat` on all six builds; the GROUND `c318_b`
  (two testers / equalities over `ite` chains of list literals, no declared symbol) `sat` too, and it is the closed
  assertion through which `d00318`#1's falsifying model escaped pass 17's net (`#P2b-88`). Naming the inner `ite`
  (`M10_NAMED`) or pushing the tester into the branches (`m10_pushed`) answered `unsat`; asserting the tester bare
  (`m14`) `unknown`. **Mechanism, traced:** the encoder replaces every non-Bool `ite` by a proxy `p` beside `c ⇒ p = a`,
  `¬c ⇒ p = b` (`encode::bool_euf_encoding`), while `solver::dt_axioms` scanned `self.assertions` — the terms as
  written — and asserted its lemmas over the written `ite`: the lemmas' `((_ is cons) (ite c l1 nil))` and the
  assertion's `((_ is cons) p)` were two unrelated SAT atoms, the second free (`#P2b-88`'s written-vs-encoded
  mechanism, here producing a wrong verdict). The same scan stops at a binder, so a datatype term that only a
  quantifier's encoding brings into the search — the ground expansion of a bounded universal, an MBQI, e-matching,
  blind or finite-domain instance — had no axiom at all: `(= l1 l2)`, `((_ is cons) l1)`, `((_ is cons) (tl l1))`
  beside `(forall ((n Int)) (=> (and (<= 0 n) (<= n 2)) ((_ is nil) (tl l2))))` answered `sat` on every build (z3
  `unsat`; found by re-fix pass 18's `scripts/round4/gen_dtite.py --v1` seed 30101802, `t00088`; `$R/fix18/atk/q/q1`, `q2`,
  `q5`, and `q3`, `q7` `unknown`). **Fixed at the root (re-fix pass 18, decision (84)), by both of the decision's
  routes and the quantified half:** (1) a tester, a selector or an equality over a datatype-sorted `ite` is lifted into
  its branches at term construction and on substitution (`oxiz-core` `ast/manager/dt_ite_lift.rs`, memoised, at most
  256 `ite` nodes; an equality of two `ite`s while the product of their sizes stays in that bound) —
  `((_ is cons) (ite c l1 nil))` is `(ite c ((_ is cons) l1) false)`; (2) the datatype axioms and the model builder
  read every member in its encoded spelling, a written `ite` as its proxy (`Solver::encoded_dt_scan`), so an `ite` the
  lift does not reach — under an uninterpreted or a constructor argument — is axiomatised as the search decides it;
  (3) every quantified assertion's encoding and every quantifier instance that holds a datatype term is a root of the
  datatype axioms (`Solver::register_ground_dt_root`, journalled; axiomatised before each MBQI round). The pass's
  draft also encoded each lemma through the `ite` elimination; beside (2) that was measured redundant (`gen_dt.py`
  seeds 30100192 and 30093154, z3-judged: the same class at every check but cap-edge timeouts) and dropped. **Why both
  routes, by mutation** (`$R/probes/probe_iso18`, `$R/fix18/make_iso18.py`, `$R/fix18/logs/mut18.out`):
  `OXIZ_MUT18_NO_ITE_LIFT` (the scan alone) leaves the ground `c318_b` and `c318_a6` `unknown`;
  `OXIZ_MUT18_NO_ENCODED_SCAN` (the lift alone) leaves every `ite` under an uninterpreted or a constructor argument
  `unknown` (`f01`, `f02`, `f05`, `f14`, `t00088`'s first check); both together leave `m10`, `m15`, `d00370`, `c318_b`,
  `c318_a6`, `f03`, `f05`, `f07`, `f08`, `f09`, `f12` and `f16`'s first check `unknown` — the honesty net (decision
  (85)) catching what the axioms no longer see; `OXIZ_MUT18_NO_DT_ROOTS` makes `q1`, `q2`, `q5` and both of `t00088`'s
  checks wrong `sat`s again; all four (the net's switch too) bring back every wrong `sat`: `m10`, `m15`, `d00370`,
  `c318_b`, `c318_a6`, `f09`, `q1`, `q2`, `q5`, `t00088`. Against re-fix pass 17 the two routes cost `gen_dt.py` seed
  30100192 `d00401`'s two checks (`unknown` here, `:reason-unknown incomplete`; restored only by both switches
  together; `c702310` and pass 14 do not decide them either) and move the search's trajectory on every `gen_dt.py`
  seed — decision (24a)'s re-fix pass 18 leg names each check. **Measured** (z3 4.15.4, cap
  10 s, 2 workers): wrong `sat` on `gen_dt.py` seed 30100192 1 → 0; 0 wrong verdicts on the four other `gen_dt.py`
  seeds, `gen14.py`'s six, and re-fix pass 18's family corpora `gen_dtite.py --v1` 30101802 (300 scripts, 485 checks,
  unguarded selectors: the 14 models zjudge calls falsifying are consistent for a value of the wrong-constructor
  selector they leave unspecified, `scripts/round4/selcheck.py`) and 30101803 (300, 506 checks, selectors under their
  tester as in `gen_dt.py`, 105 quantified: 0 falsifying, 2 withheld, both quantified) — re-fix pass 17 gives 3 and 6
  wrong `sat`s there and leaves 154 / 156 scripts unanswered at the 10 s cap, this tree 68 / 68. Pins
  `round4_pass17_recheck_pins::a_tester_over_a_datatype_ite_is_refuted` (inverted: `d00370`, `m10`, `m15`, `c318_b`
  `unsat` in both profiles), `round4_pass18_fix_pins::testers_selectors_and_equalities_over_datatype_ites_are_decided`
  (sixteen positions: an `ite` under a constructor and an uninterpreted argument, both sides of an equality, nested,
  push / pop, a quantifier body, enumeration and uninterpreted-sort `ite`s), `::a_datatype_term_only_a_quantifier_body_names_is_axiomatised`. **Scoped by re-fix pass 19 (decision
  (92)(b)), after adversarial recheck 18:** the soundness closure holds at every size measured (a tester over a chain of
  10 to 258 `ite` nodes is never answered `sat`), while the VERDICT is reached only within the lift's bound: past
  `MAX_ITE_LIFT_NODES` (256) nodes the tester is left as written and the proxy route alone does not decide it at scale
  (250 nodes `unsat` in 0.47 s, 258 no answer in 130 s) — open as `#P2b-93`, pinned
  `round4_pass18_recheck_pins::a_tester_over_an_ite_chain_past_the_lift_bound_is_not_decided`. The quantified half (3)
  has a completeness cost on the family corpora, decision (24a)'s mechanism (H).

## #P2b-1

- [x] **#P2b-1 — `BvSolver::assert_ule` memoised nothing, which the U-Z10 rollback made more expensive.** **Closed
  2026-09-29 (re-fix pass 12, decision (45)):** `assert_ule(a, b)` asserts `¬ult_gate(b, a)`, the gate memoised per
  ordered pair in `ult_cache` and defined once at the embedded solver's root, so no rollback rebuilds it
  (`oxiz-theories/src/bv/solver/scope.rs`; the U-Z10 journals no longer exist — see `#P2b-46` (f)). History: each
  call used to allocate a fresh comparison variable and re-run `encode_ult_result`, unlike `assert_ult`; evidence
  cargo-formal Phase 2b `p2b/w0/I0-a.md` §2.2.

## #P2b-2

- [x] **#P2b-2 — keep BV *definitional* clauses at the SAT solver's base level and scope only the *assertion*
  units.** **Closed 2026-09-29 (re-fix pass 12, decision (45)):** `oxiz_sat::Solver::add_clause_at_root`
  (`oxiz-sat/src/solver/root_clause.rs`) installs every `BvSolver` encoder clause; the assertions are scoped as
  *assumption literals* of `solve_with_assumptions`, not as unit clauses under an embedded `push`/`pop` — see
  `#P2b-46` (f). History: the U-Z10 fix retracted circuit nodes on `pop()`, so every circuit above a popped level was
  rebuilt with fresh SAT variables and the old ids leaked; evidence `p2b/w0/I0-a.md` §2.2.

## #P2b-59

- [x] **#P2b-59 (2026-09-21) — REGRESSION against `c4b04b7` and crates.io 0.3.3, present in the committed
  checkpoint `00add07`: a QUANTIFIER-FREE `QF_ABV` script's lazy array refinement does not converge.**
  **FIXED at the root 2026-09-22 (re-fix pass 10); the residual cost is reclassified to `#P2b-46` (f) with its
  counters — see the close-out at the end of this entry, which also WITHDRAWS two of the claims below.** Opened by
  the cargo-formal adversarial recheck pass 9, by delta-debugging `rk6/corpus/qmbqi120/q0033.smt2` one and two
  assertions at a time. The minimal script is `rk9/min/q33_a01.smt2`: two width-8 arrays with `ite`-selected
  bases, one `store` each, one `(as const …)`, and `grep -c forall` is **0** — no binder is involved at all, so
  none of this round's quantifier work can be the reason. **Measured, release, four probes on that one script:**
  `c4b04b7` `sat` 7.3 ms; crates.io 0.3.3 `sat` 8.4 ms; the pass-6 checkpoint `00add07` (`rk8/probe_p6`) **no
  answer in 120 s**; this tree **no answer in 90 s**. The regression therefore arrived at or before `00add07` —
  it is in the **committed** work, not in re-fix passes 7, 8 or 9. Its siblings decide: `q33_a0` `sat` 581.9 ms,
  `q33_a1` 2.4 ms, `q33_a2` 0.5 ms, `q33_a02` 69.8 ms, `q33_a12` 5.6 ms; only the pair of the two
  quantifier-free assertions hangs.
  **It fails to terminate rather than being slow, and the deterministic counters say so**
  (`rk9/min/q33_a01_b{500,5000}.smt2`): at `(set-option :max-bv-embedded-checks 500)` it is `unknown` in 19.0 ms
  with `:array-refinement-rounds 7` and `:array-lemma-instances 68`; at `5000` it is `unknown` in 477.7 ms with
  `:array-refinement-rounds 26` and `:array-lemma-instances 104`. **The round count grows with whatever budget
  it is given instead of reaching a fixpoint**, and each round then pays `#P2b-46` (f)'s `O(num_vars)`
  embedded-check price. Four more members of the family are in the same corpus — `rk6/corpus/qmbqi120/q0033`,
  `q0047`, `q0103` and `q0107`, `sat` in 6.6–11.4 ms on the base, no answer in 120 s here, and their ground twins
  do not finish either. *(`q0107` is **WITHDRAWN** from that list on 2026-09-22: it is `sat` in 160.2 ms on the
  fixed tree — see the close-out, (e).)*
  **The mechanism is named by counters rather than by bisection.** `#P2b-48`'s hoist of
  `Solver::array_refinement_round` out of `check_core`'s `!has_quantifiers` branch is what makes the refinement
  machinery run on scripts where the base never ran it: on
  `bench/z3_parity/benchmarks/AUFLIA/array_update.smt2` this tree and `00add07` are counter-IDENTICAL
  (`:conflicts 17 :propagations 1308 :array-refinement-rounds 7 :array-lemma-instances 24`) while `c4b04b7`
  publishes **no array-refinement counters at all** (`:conflicts 6 :propagations 141`) — the same signature at a
  survivable size, which is why `#P2b-46` (f) already attributes that one `bench/` regression to the same hoist.
  **What this is NOT.** Not `#P2b-46` (f): no budget is installed on the script and `:conflicts` is tiny, where
  that entry is about scripts a `:max-conflicts` budget stops. Not any quantifier item — not `#P2b-50`,
  `#P2b-51`, `#P2b-53`, `#P2b-54`, `#P2b-55`, `#P2b-56`, `#P2b-57` or `#P2b-58` — because the script carries no
  binder, which the pin **asserts** rather than assumes.
  **Lever**: `Solver::array_refinement_round`'s termination argument on `ite`-selected array bases. 7 → 26 rounds
  as the budget grows means the round is re-deriving lemmas it already has, or the `ite` spine is minting a fresh
  array root per round. Bisecting passes 1–6 of this round will name the commit; the recheck deliberately did not,
  and the bracket it measured is `c4b04b7` fast / `00add07` and later not terminating. Pinned by
  `round4_pass9_recheck_pins::a_quantifier_free_array_script_the_base_decides_is_undecided_here` *(inverted
  and since renamed `the_refinement_round_count_is_bounded_and_equal_inside_the_plateau`)*, which asserts
  that the script carries no binder, asserts `unknown` at two deterministic budgets (500 and 2,000 in the pin;
  500 and 5,000 in the measurement above) and asserts that the refinement-round count **strictly grows** between
  them — so a tree that converged reddens it.

  ---

  **CLOSE-OUT (2026-09-22, re-fix pass 10, decision (35)). Two claims above are withdrawn in place, by
  measurement.**

  **(a) The bisection the entry asked for, run rather than reasoned** (decision (35)(a); one reused detached
  `git worktree` under the scratchpad, removed afterwards, `git worktree prune` run, `git worktree list` back to
  two entries; driver and logs at `rf11/bisect/`). Every row is `rk9/min/q33_a01.smt2` at
  `:max-bv-embedded-checks` 500 / 5,000, release:

  | commit | 500 | 5,000 | counters |
  |---|---|---|---|
  | `c4b04b7` (base) | `sat` 8.1 ms | `sat` 0.34 ms | `:conflicts 0 :propagations 9`; no refinement counters exist yet |
  | `9dcfd0d` MOS | `sat` 80.5 s | `sat` 50.2 s | `:conflicts 1465 :propagations 140238` |
  | `7cd140c` MOS | no answer in 120 s | no answer in 120 s | — |
  | `b402c80` MOS | `sat` 13.0 ms | `sat` 5.9 ms | `:conflicts 15 :propagations 1928` |
  | `ed101fb` MOS | no answer in 120 s | no answer in 120 s | — |
  | `400988e` (passes 4–5) | no answer in 120 s | no answer in 120 s | — |
  | `8a423bd` MOS | no answer in 120 s | no answer in 120 s | — |
  | `00add07` (pass 6) | `unknown` 10.2 ms | `unknown` 194.3 ms | 7 rounds / 68 instances; 26 / 104 |
  | `0559fb3` (HEAD) | `unknown` 11.0 ms | `unknown` 186.7 ms | 7 / 68; 26 / 104 — **counter-identical to `00add07`** |

  `:max-bv-embedded-checks` did not exist before `400988e`, which is why the two columns agree on the earlier
  rows. The first departure from the base is **`9dcfd0d`** (0.34 ms → 50 s), the last commit that still decides
  it is **`b402c80`**, and from **`ed101fb`** on it never finishes again. So the regression lives in the
  array-axiom families of `9dcfd0d`…`ed101fb` and **not** in `#P2b-48`'s hoist: `00add07` and HEAD are
  counter-identical, and the script carries no binder, so a mechanism that only fires on quantified input cannot
  reach it. **The "named by counters rather than by bisection" attribution above is WITHDRAWN**; the
  `array_update.smt2` half of it stands on its own and stays under `#P2b-46` (f).

  **(b) "It fails to terminate rather than being slow" is WITHDRAWN.** The ladder was measured at 500 vs 5,000,
  inside the ramp. Above it the counts saturate: HEAD gives 91 rounds / 226 instances at 20,000 checks and
  **93 / 228** at 50,000 — +2 and +2 for 2.5x the budget — while the wall clock goes 552 ms → 45,401 ms and the
  conflicts 1,973 → 11,968. The lemma set is deduplicated on the interned lemma term and it does reach a
  fixpoint; what grows is the search over it. The second lever decision (35)(b) named — "a lemma already derived
  is never re-derived" — is therefore **not** the defect, and was not implemented.

  **(c) The mechanism, measured at index width 8 — above the extensionality enumeration limit, which is where
  the fix is restricted to** (`rf11/out/struct20k.txt`, an instrumented release probe printing the collected
  `ArrayStructure` each round). `Solver::assert` stores the **pre**-rewrite term in
  `Solver::assertions` and registered the **encoded** term as a ground array root
  (`register_encoded_assertion_root`, the pass-5 seam). Where the encoding chain replaced an array-sorted
  `(ite c a b)` with the fresh proxy `eliminate_nonbool_ite` mints, the collector — which walks both root sets —
  saw the same array twice: **19** array terms where 13 is the truth (five `\oxiz.iteelim!*` proxies plus one
  `store` over a proxy), **21** extensionality witnesses of which ten pair a term with the proxy of the same
  array, and 58 of 226 asserted lemmas spelled over a proxy. Every pair set is quadratic in that count, and
  phases 3a/3b/3c/4 spend one *round* — one whole re-solve — per pair. Ruled out on the way: the model-guided
  filter is not degenerate (`rf11/out/eval.txt`: 4,485 candidate instances skipped because the candidate model
  already satisfies them, against 142 asserted).

  **(d) The fix** is `Solver::array_root_spelling` (`solver/ground_instance.rs`) over the new
  `Solver::ite_elim_aliases` map (proxy → the `ite` term it names, recorded by `eliminate_nonbool_ite`, keyed on
  the `ite`'s own id so a re-mint re-learns the same entry and no trail op is needed): an assertion's encoded
  root is registered in the **`ite`** spelling, the one the other half of the root set already uses and the one
  that carries the branches for `build_array_ite_reads`. Nothing is lost on the encoding side — every lemma goes
  back through `Solver::encode`, which re-derives the proxy for the SAT core. **It is narrowed twice, each time
  by a measurement rather than by taste.** (i) *To the assertion path.* Applying it to ground instances too took
  `rk6/corpus/qmbqi120/q0075` from `unknown` in 18.4 ms to no answer in 120 s and `q0106` from 78.5 ms to
  205.5 ms, while leaving `q33_a01` — which has no quantifier and so no instance — bit-identical; only an
  assertion has the *same term* already in the root set, so only an assertion is re-spelled. (ii) *To index
  sorts above `ARRAY_INDEX_ENUMERATION_LIMIT`.* An enumerated pair mints no Skolem witness, so the duplicate
  spelling costs at most a constant factor of lemmas over one shared index set, while putting the `ite` back
  hands `build_array_ite_reads` a conditional read pair at every element of the domain for every read the
  enumerated family creates. Measured on the 300-pair width-1/2 corpus of `round4_pass5_recheck_pins`: pair 10
  (a `forall` over `(_ BitVec 2)` beside a ground `ite` over two `store`s) went `sat` 8.5 ms → `sat` **2,702 ms**
  with the verdict unchanged, and the whole test went from passing to a **TIMEOUT at the gate's 180 s ceiling**;
  with the limit respected that pair is `sat` in **2.9 ms**, faster than before the fix, and the test runs in
  10.6 s. Both halves are pinned (see (g)).

  **(e) Measured on the final tree**, release, two runs of every unbudgeted figure.
  `rk9/min/q33_a01.smt2` unbudgeted: **`sat` in 30.4 s** (66 refinement rounds, 208 lemma
  instances, 30,402 embedded checks, 8,294 conflicts) where HEAD gave no answer in 90 s and the
  pass-9 tree none in 400 s.
  **The budget ladder, re-measured (adversarial recheck pass 10; the "reaches a fixpoint" reading of
  it is WITHDRAWN).** What re-fix pass 10 recorded — "the budget ladder reaches a fixpoint: 10 / 76 at
  2,000 and 10 / 76 at 5,000" — is false as a fixpoint claim: the equality is a **plateau inside a
  ramp**, and the same method one rung higher gives 41 / 129.

  | `:max-bv-embedded-checks` | verdict | rounds / instances |
  |---|---|---|
  | 500 | `unknown` | 9 / 75 |
  | 2,000 | `unknown` | 10 / 76 |
  | 5,000 | `unknown` | 10 / 76 |
  | **8,000** | `unknown` | **41 / 129** |
  | 10,000 | `unknown` | 51 / 181 |
  | 12,000 | `unknown` | 57 / 197 |
  | 20,000 | `unknown` | 59 / 199 |
  | **50,000** | **`sat`** | **66 / 208**, 30,402 of 50,002 checks spent |

  The refinement **does** reach a fixpoint — at **66 rounds / 208 instances**, where it stops asking
  for budget and publishes `sat`. Every rung below that is a *truncation by the budget*, not a
  fixpoint at a smaller place, and decision (35)(c)'s "counts IDENTICAL at 500 and 5,000" is **refuted
  at its own two budgets** — 9 / 75 at 500 against 10 / 76 at 5,000, as the table above prints (recheck
  pass 11 re-measured both). The only budgets at which the counts are identical are above the 30,402-check
  saturation point: 50,000 and 200,000 both answer `sat` with byte-identical 66 / 208 / 30,402 / 8,294.
  Taking that assertion costs 30–50 s of release time and minutes of the gate's, which is why no test
  takes it and why this entry states the ladder instead; the gate pins the plateau
  (`round4_pass9_recheck_pins::the_refinement_round_count_is_bounded_and_equal_inside_the_plateau`) and
  the recheck's `round4_pass11_recheck_pins::the_round_counts_at_five_hundred_and_five_thousand_checks_differ`
  keeps the two figures apart.

  **(f) What is left, and where it belongs.** `q33_a01`'s 30.4 s against the base's 0.34 ms is **`#P2b-46` (f)**,
  not this entry. The price is the `O(num_vars)` embedded check, and it is **a curve, not a rate**: on this
  script the tree spends 532.1 ms on 8,002 checks (**0.07 ms each**), 4,618.3 ms on 20,002 (**0.23 ms each**)
  and 30,433.4 ms on 30,402 (**1.00 ms each**). A per-check cost that grows as the variable table grows *is*
  mechanism (i); the single point "~1.0 ms each" that stood here until 2026-09-22 was the last rung of that
  curve quoted as a rate, which would have a later pass budget a script by multiplying its check count by a
  millisecond. (`rf6/cal/st10_w3.smt2` sits further along the same curve at ~1.8 ms per check over 250,002.) At
  `(set-option :max-conflicts 200)` — the spelling mechanism (i)'s five named scripts use — `q33_a01` is
  `unknown` in 22.8 ms having spent 194 of its 200 conflicts, which is that family's signature. Three qmbqi120
  members — `q0033`, `q0047` and `q0103` — still do not finish in 130 s and are carried to `#P2b-46` (f) **by
  path**, as instances of mechanism (i) rather than of a refinement that fails to converge; the fourth,
  `q0107`, is decided here.
  **A fourth script is carried there too, and it is this entry's own repro in the other spelling.**
  `rk11/atk/x10_q33_swapped.smt2` is `rk9/min/q33_a01.smt2` with its two assertions written in the opposite
  order and nothing else changed (the sorted assertion lists `diff` empty, so it is the same formula).
  `c4b04b7` and crates.io 0.3.3 answer `sat` in 0.1 ms; this tree gives **no answer in 300 s**, and its ladder
  is 8 / 74 at 500 embedded checks, 25 / 97 at 2,000, 33 / 122 at 5,000 and at 20,000, and 52 / 171 at 50,000
  with all 50,002 checks spent — while at 120,000 checks it produces **no answer in 900 s**
  (`rf12/min/sw_b120000.smt2`, exit 124), so its saturation point is not quoted because reaching it costs more
  than the measurement budget. `array_root_spelling` normalises the array-root **spelling**; it does not
  normalise the assertion **order**, and the order is what decides which foreign pair the one-pair-per-round
  phases (`#P2b-46` (2)'s phases 3a/3b/3c/4) reach first, so the two spellings take different trajectories
  through the same lemma set. **`#P2b-59` is therefore NOT closed with no open scripts of its own** — that
  sentence, written by re-fix pass 10, is withdrawn — and this spelling is open under mechanism (i) with its
  path, pinned by `round4_pass10_recheck_pins::the_same_script_with_its_two_assertions_swapped_is_not_decided`,
  whose in-test control asserts that the *original* order's count does not climb, so the pin is a statement
  about the order and not about the script.
  Decision (35)(c)'s target — "decide `sat` in milliseconds" — is not met for either spelling and is not
  reported as reached: the original order decides in 30.4 s and the swapped order does not decide.

  **(g) Pin** (decision (35)(d)): `round4_pass9_recheck_pins::the_refinement_round_count_is_bounded_and_equal_inside_the_plateau`
  — the pass-9 hole pin inverted, and **renamed on 2026-09-22** (re-fix pass 11) from
  `a_quantifier_free_array_script_reaches_a_refinement_fixpoint`, which named a property the ladder in (e)
  refutes. It still asserts the script carries **no binder**, and asserts that `:array-refinement-rounds` and
  `:array-lemma-instances` are identical at budgets 2,000 and 5,000 and at or below 12 / 90 — a **bounded,
  equal count inside the plateau**, which is what it measures. It remains a working regression guard: the
  pass-9 tree reports 20 and 26 at exactly those two budgets and fails both halves. Beside it,
  `round4_pass10_recheck_pins::the_array_refinement_round_count_still_grows_with_the_budget` pins the cheapest
  rung that breaks the plateau (5,000 against 8,000: 10 / 76 against 41 / 129), so a tree whose plateau
  *moved* reddens there rather than passing silently, and
  `::the_same_script_with_its_two_assertions_swapped_is_not_decided` pins the open half with the original
  order's stable count as its control. Two white-box unit tests in
  `solver::array_axioms::tests::one_spelling_per_array_tests` state the rule and its boundary directly:
  `an_ite_selected_array_base_is_one_array_and_not_two` (at index width 8 the collector reaches four array
  terms and no proxy, where the unfixed tree reaches five) and
  `below_the_enumeration_limit_the_proxy_deliberately_stays` (at index width 2 the proxy is still there,
  which reddens if a later pass widens the rule past the enumeration limit and re-creates the 2,702 ms case).
  A third pin carries the verdict the fix buys back:
  `round4_pass9_recheck_pins::the_corpus_member_the_root_spelling_buys_back_is_decided` is
  `rk6/corpus/qmbqi120/q0107.smt2` verbatim, `sat` in 0.51 s in the gate profile.
  Mutation table, re-fix pass 10, with **M10a's one inferred row replaced by the measurement**
  (adversarial recheck pass 10; the pass had written "1 timed out" and inferred which test it was):

  | mutation | effect |
  |---|---|
  | **M10a** `array_root_spelling` returns its input | 3 red: the width-8 unit test (5 array terms, one of them `(store \oxiz.iteelim!9 …)`), the plateau pin (`[20, 26]` rounds), and the `q0107` verdict pin. **Measured** rather than inferred for the third: the unfixed solver (`rf10/bin/probe_p9`, which is M10a behaviourally) answers `unknown` on `q0107` in **174,103.3 ms** release against the fixed tree's `sat` in 160.6 ms — so the pin reddens on the **verdict**, and 174 s release would also cross the gate's 180 s ceiling in the dev profile. The boundary test stays green. |
  | **M10b** drop the equality half of the plateau pin, keep only the `<= 12` cap | insufficient **in general**: the unfixed tree reports 91 and 93 rounds at 20,000 and 50,000 checks, so any bound above 93 passes on it. |
  | **M10c** drop the cap, keep only the equality | insufficient **in general**: 91 vs 93 is two apart, so an equality test with any tolerance passes on the unfixed tree. |
  | **M10d** widen the re-spelling past `ARRAY_INDEX_ENUMERATION_LIMIT` | `below_the_enumeration_limit_the_proxy_deliberately_stays` goes red while `an_ite_selected_array_base_is_one_array_and_not_two` stays green (so the boundary is pinned independently of the rule), and the 300-pair width-1/2 corpus goes from 10.6 s to a TIMEOUT at 180 s. |

  M10b and M10c are stated as what they are — an argument from the measured ladder, not two more builds —
  which is why the delivered pin asserts the conjunction at budgets 2,000 and 5,000, where the unfixed tree
  gives 20 vs 26 and fails both halves.
  **(h) Re-measured 2026-09-29 on re-fix pass 12's final tree (decisions (43)-(45); release, `$R/fixB/final/ladder_*`, both probes back to back beside two campaign workers; `c702310` is the pre-pass tree).** Decision (45) — root-scoped bit-blasting with failed-assumption cores, `#P2b-46` (f) — is what decides both spellings; decision (43)'s canonical order is in; decision (44) did not land.

  | `:max-bv-embedded-checks` | `q33_a01` `c702310` | `q33_a01` this tree | swapped `c702310` | swapped this tree |
  |---|---|---|---|---|
  | 500 | `unknown` 9 / 75 | `unknown` 9 / 69 | `unknown` 8 / 74 | `unknown` 6 / 74 |
  | 2,000 | `unknown` 10 / 76 | `unknown` 30 / 108 | `unknown` 25 / 97 | `unknown` 26 / 110 |
  | 5,000 | `unknown` 10 / 76 | `unknown` 47 / 158 | `unknown` 33 / 122 | `unknown` 52 / 144 |
  | 8,000 | `unknown` 41 / 129 | `unknown` 64 / 188 | `unknown` 33 / 122 | `unknown` 66 / 173 |
  | 20,000 | `unknown` 59 / 199, 6.0 s | **`sat` 70 / 195**, 9,302 checks | `unknown` 33 / 122, 42.7 s | `unknown` 107 / 254, 0.5–0.9 s |
  | 50,000 | `sat` 66 / 208, 30,402 checks, 59.1 s | **`sat` 70 / 195**, 9,302 checks, 0.16–0.36 s | no answer in 130 s | **`sat` 124 / 295**, 29,189 checks, 0.8–1.2 s |

  (e)'s fixpoint moves from 66 / 208 after 30,402 checks to **70 / 195 after 9,302**, and the swapped order is decided for the first time. Decision (35)(c)'s "decide `sat` in milliseconds" is **not met as written**: 0.16–0.36 s for the original order and 0.8–1.2 s for the swapped one, against 0.1 ms on `c4b04b7`. **Decision (44) is NOT met:** below 9,302 checks the count still grows with the budget (9 → 47 → 70 rounds at 500 / 5,000 / 50,000). Both variants it names were built on this tree and measured unbudgeted — every deferred family for every violated pair in each round (`q33_a01` 5 rounds but 27,700 checks, 1.0 s; swapped 9 rounds, 123,970 checks, 4.2 s) and the same staged behind the eager families (20 rounds, 74,363 checks, 2.1 s; swapped 18, 43,122) — and both **raise** the checks spent against 9,302 and 29,189, so neither landed (patches `$R/fixB/patches/p44_*`). Laddered at 500 / 5,000 / 20,000 / 50,000 checks (`$R/fixB/final/v44/`), neither meets the criterion itself either: every-family rounds `q33_a01` 2 / 2 / 2 / **5** (`sat` only at 50,000) and swapped 2 / 2 / 4 / 4 (**`unknown` at 50,000**, which the landed tree decides); staged 4 / 9 / 16 / 16 (`q33_a01` **`unknown` at 50,000**) and 4 / 5 / 13 / 18. Each variant loses one of the two verdicts the landed tree reaches at 50,000. **Decision (43)'s outer half is NOT met:** round 1 of the two orders asserts the same instances, and the trajectories part after it because the outer search numbers the assertions' SAT variables in the order written (70 / 195 against 124 / 295).
  **Pins, re-fix pass 12:** the plateau pin became `round4_pass9_recheck_pins::the_refinement_reaches_a_fixpoint_above_its_saturation_point` (`sat` with identical rounds / instances / checks at 20,000 and 50,000 and fewer checks spent than offered — the fixpoint (e) could only state, now cheap enough to assert); `round4_pass10_recheck_pins::the_same_script_with_its_two_assertions_swapped_is_not_decided` is inverted as `::the_same_script_with_its_two_assertions_swapped_is_decided` (both orders `sat`, each with identical rounds / instances / checks at 35,000 and 50,000 and fewer checks spent than offered — a fixpoint per order; the counters *between* the orders are the open hole below); the holes stay green with corrected docs — `::the_array_refinement_round_count_still_grows_with_the_budget` (47 / 158 at 5,000 against 64 / 188 at 8,000), `round4_pass11_recheck_pins::the_round_counts_at_five_hundred_and_five_thousand_checks_differ` (9 / 69 against 47 / 158) — and `round4_pass10_recheck_pins::the_two_assertion_orders_still_take_different_refinement_trajectories` is new. Unit tests `solver::array_axioms::canonical::tests::{two_interning_orders_give_every_term_the_same_key, distinct_structures_get_distinct_keys}`. `rk6/corpus/qmbqi120/q0033`, `q0047` and `q0103` could not be re-run (destroyed with the earlier scratch directory).
  **Re-measured 2026-09-29 by adversarial recheck 12 and again on re-fix pass 13's final tree (decision (51); `$R/fix13b/camp/ladder_probe_tree_*`): the table's tree columns and `q0072` (`sat` 8 / 78 at 500, `unknown` at 5,000 / 8,000, `sat` 92 / 227 from 20,000, 16,017 checks) reproduce exactly — (43)'s outer half and (44) stay OPEN as measured.** Budgets a trajectory change makes lose, all satisfiable: `det_w3_n12` decided by `c702310` at 75,741 checks and here at 232,481; `$R/corpus/qmbqi120/g0103` (the ground twin) by `c702310` at 17,374 and here at 83,387 (16 / 1,340 at 20,000 `unknown`) — the latter is `#P2b-69`'s price; and (fresh corpora, decision (59)) `fq200/g0029`, `g0122`, `g0127`, `feq200/g0142`, decided by `c702310` at 5,785 / 7,548 / 7,400 / 8,501 checks and not here at 20,000. Re-fix pass 14's final tree reproduces every ladder above exactly (`$R/fix14/final/ladder_tree_*`: `q33_a01` 64 / 188 at 8,000 and `sat` 70 / 195 from 20,000; swapped 107 / 254 `unknown` at 20,000, `sat` 124 / 295; `q0072` `sat` 92 / 227 from 20,000; `det_w3_n12` 2.4 µs per check at 20k and 100k).
  **Trajectory also depends on push/pop history (decision (61)(10); adversarial recheck 13, re-taken on re-fix pass
  14's final tree, `$R/fix14/final/ord_*`).** `q33_a01`'s body checked in a fresh scope takes 70 / 195 rounds / instances
  (9,302 checks); the identical body re-asserted after a `(push 1)` / `(pop 1)` around it takes 73 / 237 (13,528 checks),
  `sat` both times; other spellings of the same formula: declarations permuted 70 / 195, `or` disjuncts swapped
  103 / 302, the two assertions swapped 124 / 295; a fresh three-array `ite`-spine script in three orders 72 / 242,
  70 / 256, 79 / 248 — every one `sat`. The same open half of decisions (43) / (44) as above: what the refinement
  instantiates is order- and history-independent, the outer search's variable numbering is not.
