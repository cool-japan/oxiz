//! Internal regression pins for [`Solver::rebase_theory_state`] (task #26).
//!
//! The end-to-end verdict pins live in `tests/scope_leak_hazard.rs`.  What
//! cannot be observed from an integration test is the *shape* of the fix, and
//! two properties of it are load-bearing:
//!
//! * **The rebase really reaches the base scope.**  The absolute depth counter
//!   in [`DerivedReasons`](super::theory_manager::DerivedReasons) is the only
//!   place the EUF / arithmetic / bit-vector solvers' true depth is tracked
//!   across `TheoryManager` lifetimes — a manager numbers scopes from its own
//!   `level_stack`, which restarts at zero.  A rebase that left the counter
//!   above zero would mean scopes are still open and the leak is still there.
//! * **Re-derivation costs no clauses.**  The facts the next round is entitled
//!   to are recovered by replaying the SAT trail through
//!   `TheoryManager::on_assignment`, never by re-encoding the kept lemmas.  Had
//!   it been done through `Solver::encode` instead, each round would re-emit
//!   those lemmas' clauses: still *correct*, but the clause database would grow
//!   without bound across rounds.
//!
//! A clause total on its own cannot tell those two designs apart, because it
//! cannot tell whether a round boundary was crossed at all — see
//! `mbqi_round_boundaries_are_crossed_once_and_then_saturate` for the mistake
//! that made, and for why the boundary is now counted directly
//! (`Solver::mbqi_round_clauses`) before anything is claimed about its cost.
//!
//! # Clause-database growth across repeated `check-sat` (task #28)
//!
//! The second half of this file pins the *other* direction of the same concern:
//! not what one MBQI round costs inside a single `check`, but what a whole
//! `check` costs when the caller repeats it on a goal it has not touched.  The
//! answer must be "nothing", and three separate mechanisms used to make it
//! something —
//!
//! * hyper-binary-resolution clauses that no ledger recorded, so they could be
//!   neither forgotten nor retracted (and were misreported as original clauses
//!   by every "total minus registry" computation, which is how the growth was
//!   first noticed):
//!   `learned_clauses_are_all_registered_in_the_learned_clause_ledger`;
//! * a `pop` that dropped the whole Tseitin memo, including entries whose
//!   clauses it did *not* retract, so the next `check` re-emitted them verbatim
//!   — linear, unbounded growth per `(push)(pop)` pair:
//!   `a_no_op_push_pop_between_checks_does_not_re_encode_the_goal`;
//! * quantifier-search state outliving the search that produced it, so a
//!   repeated `check` started somewhere else and derived genuinely new lemmas
//!   several calls in: `a_check_leaves_the_mbqi_search_state_where_it_found_it`
//!   at the root cause, and `re_running_the_search_on_an_unchanged_goal_converges`
//!   end to end.
//!
//! Above all three sits a fourth change — the repeated-`check` verdict cache —
//! and it is the reason the pins come in pairs.  A caller repeating
//! `(check-sat)` never reaches the search machinery at all now, so a pin written
//! from the caller's vantage point pins the cache and nothing beneath it.  Every
//! mechanism above therefore also has a pin that drops the cached verdict first
//! (`VerdictCache::Bypassed`) and makes all twelve calls search for real.  The
//! cache's own contract — that a hit answers what a fresh solver would, and that
//! any settings change makes the next call search again — lives in
//! `solver::verdict_cache::tests`.
//!
//! All three pins measure `ClauseDatabase::num_original` through
//! `oxiz_sat::Solver::num_original_clauses`, never `num_clauses()` minus
//! `learned_clause_count()`.  The latter subtracts the *registry* size from the
//! *database* size, so an unregistered learned clause shows up as an original
//! one — a pin written that way would have pinned the accounting bug in place
//! instead of catching it.

use crate::Context;
use crate::solver::SolverResult;
use oxiz_core::ast::TermManager;

/// A satisfiable quantified UFLIA goal from the fragment
/// `tests/mbqi_sat_certification.rs` covers, with two branching integers so the
/// ground search has to take decisions (and therefore opens theory scopes).
const QUANTIFIED_BENCHMARK: &str = r#"
    (set-logic UFLIA)
    (declare-fun f (Int) Int)
    (declare-const x Int)
    (declare-const y Int)
    (assert (or (= x 1) (= x 5)))
    (assert (or (= y 2) (= y 7)))
    (assert (= (f 1) 100))
    (assert (forall ((i Int)) (=> (= (f i) 100) (not (= x i)))))
"#;

/// How many `check-sat` calls the plateau is measured over.
const ROUNDS: usize = 10;

/// After a `Sat` that never backtracked, the theory solvers sit several scopes
/// deep; the rebase must bring them all the way back to the base scope.
///
/// Asserting on the counter rather than on a verdict is deliberate: it is the
/// invariant the fix rests on, and it fails loudly even for inputs where the
/// leaked facts happen not to contradict anything.
#[test]
fn rebase_returns_the_theory_solvers_to_the_base_scope() {
    let mut manager = TermManager::new();
    let mut solver = super::Solver::new();

    // `(or (= x 1) (= x 5))` plus `(or (= y 2) (= y 7))` forces two decisions,
    // so the `Sat` below leaves scopes open.
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let one = manager.mk_int(1);
    let five = manager.mk_int(5);
    let two = manager.mk_int(2);
    let seven = manager.mk_int(7);
    let x_is_1 = manager.mk_eq(x, one);
    let x_is_5 = manager.mk_eq(x, five);
    let y_is_2 = manager.mk_eq(y, two);
    let y_is_7 = manager.mk_eq(y, seven);
    let x_branch = manager.mk_or(vec![x_is_1, x_is_5]);
    let y_branch = manager.mk_or(vec![y_is_2, y_is_7]);
    solver.assert(x_branch, &mut manager);
    solver.assert(y_branch, &mut manager);

    assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    // The search ended `Sat` and therefore never backtracked.  Whatever depth it
    // left behind, the rebase must erase it.
    solver.rebase_theory_state();
    assert_eq!(
        solver.derived_reasons.depth(),
        0,
        "rebase_theory_state must leave the theory solvers at their base scope; \
         a non-zero absolute depth means scopes are open that no later \
         on_backtrack can reach"
    );
}

/// The rebase itself must add no clauses and no encode-memo entries.
///
/// This is the airtight half of the "re-derivation costs no clauses" claim: it
/// pins the mechanism directly rather than inferring it from a total.  A design
/// that recovered the kept lemmas by re-encoding them would grow both counters
/// here — once per MBQI round, and rounds are bounded only by
/// `max_mbqi_iterations`.
#[test]
fn the_rebase_adds_no_clauses_and_no_memo_entries() {
    let mut manager = TermManager::new();
    let mut solver = super::Solver::new();

    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let one = manager.mk_int(1);
    let five = manager.mk_int(5);
    let x_is_1 = manager.mk_eq(x, one);
    let x_is_5 = manager.mk_eq(x, five);
    let branch = manager.mk_or(vec![x_is_1, x_is_5]);
    solver.assert(branch, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Sat);

    let clauses_before = solver.sat.num_clauses();
    let memo_before = solver.encoded_terms.len();
    solver.rebase_theory_state();
    assert_eq!(
        solver.sat.num_clauses(),
        clauses_before,
        "rebase_theory_state must not emit a clause: the facts the next round \
         is entitled to are replayed from the SAT trail, never re-encoded"
    );
    assert_eq!(
        solver.encoded_terms.len(),
        memo_before,
        "rebase_theory_state must not touch the Tseitin memo"
    );
}

/// A quantified UFLIA goal whose *single* `check-sat` crosses **two** MBQI round
/// boundaries: the first model puts `x` on a branch the instantiation of
/// `(forall ((i Int)) (=> (= x i) (> (f i) 10)))` then refutes, and so does the
/// second, leaving the third branch as the model.  Used where the property under
/// test is about what a *round* costs, which needs more than one of them.
const MULTI_ROUND_BENCHMARK: &str = r#"
    (set-logic UFLIA)
    (declare-fun f (Int) Int)
    (declare-const x Int)
    (assert (or (= x 1) (= x 2) (= x 3)))
    (assert (forall ((i Int)) (=> (= x i) (> (f i) 10))))
    (assert (< (f 1) 5))
    (assert (< (f 2) 5))
"#;

/// Repeating `check-sat` on an unchanged quantified goal must cross the MBQI
/// round boundary **once, ever**, and cost no clauses after that.
///
/// # What the earlier version of this test got wrong
///
/// It measured only the clause total across repeated `(check-sat)` calls and
/// asserted a plateau, on the stated rationale that "the benchmark takes more
/// than one MBQI round — round 1 picks a branch, the instantiation lemma
/// retracts it, round 2 re-solves — so each call exercises the round-boundary
/// rebase".  That rationale is false, and the assertion could not detect that it
/// was false.  Measured here: the boundary counts per call are `[1, 0, 0, …]`.
/// Only the *first* call crosses a boundary at all; calls 2..N run no rebase and
/// encode no lemma, so their plateau was guaranteed by arithmetic and said
/// nothing whatsoever about the rebase.  The test passed loudest exactly where
/// it had stopped measuring anything.
///
/// # What is measured instead
///
/// The boundary is counted directly (`Solver::mbqi_round_clauses`, appended to
/// at the one site that encodes a round's lemmas, rebases and rebuilds the
/// theory manager), and the recorded shape `[n, 0, 0, …]` is asserted as such.
/// The zeros are the decisive part, not an inconvenience: a later call needs no
/// quantifier work *only if* the instantiation lemma the first call kept is
/// still in the clause database and the theory state re-derived after the rebase
/// is complete enough to satisfy the goal without re-instantiating.  A rebase
/// that dropped a fact would show up as `[n, n, n, …]` with a clause count
/// climbing per call — which is the actual failure mode, and which the old
/// assertion would also have caught only by accident.
///
/// # Why the goal is `MULTI_ROUND_BENCHMARK` and not `QUANTIFIED_BENCHMARK`
///
/// It used to be the latter, whose first call crossed exactly one boundary
/// (round 1 picked `x = 1`, the instantiation `(f 1) = 100 => x != 1` retracted
/// it, round 2 was satisfied).  Once every numeric `Eq` atom started carrying
/// its trichotomy `(a = b) OR (a < b) OR (a > b)` — see
/// `Solver::add_numeric_trichotomy` — the arithmetic solver settles `x` on the
/// first round and that goal crosses **no** boundary at all: measured
/// `[0, 0, …]`, with clause, memo and variable counts perfectly flat.
///
/// That is a strictly better outcome for the solver and a *fatal* one for this
/// test.  Re-pointing `EXPECTED_BOUNDARIES` at `[0, 0, …]` would have kept it
/// green while deleting its subject — the clause-plateau assertion below would
/// then be measuring a goal that never does quantifier work, which is precisely
/// the vacuity the `EXPECTED_BOUNDARIES[0] > 0` guard exists to reject.  So the
/// *goal* moved instead of the expectation: `MULTI_ROUND_BENCHMARK` still
/// crosses (two boundaries on the first call, none after) and still holds its
/// clause count flat, so both assertions keep their teeth.
#[test]
fn mbqi_round_boundaries_are_crossed_once_and_then_saturate() {
    // Measured on this benchmark: the first `check-sat` crosses two boundaries,
    // and every later call crosses none.
    const EXPECTED_BOUNDARIES: [usize; ROUNDS] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    let mut ctx = Context::new();
    ctx.execute_script(MULTI_ROUND_BENCHMARK)
        .expect("benchmark should parse and run");

    let mut boundaries = Vec::with_capacity(ROUNDS);
    let mut clauses = Vec::with_capacity(ROUNDS);
    let mut crossed_before = ctx.solver().mbqi_round_clauses.len();
    for round in 0..ROUNDS {
        let out = ctx
            .execute_script("(check-sat)")
            .expect("check-sat should run");
        assert!(
            out.iter().any(|line| line == "sat"),
            "round {round} must keep answering sat, got {out:?}"
        );
        let crossed_after = ctx.solver().mbqi_round_clauses.len();
        boundaries.push(crossed_after - crossed_before);
        crossed_before = crossed_after;
        clauses.push(ctx.solver().sat.num_clauses());
    }

    assert_eq!(
        boundaries.as_slice(),
        EXPECTED_BOUNDARIES.as_slice(),
        "MBQI round boundaries per check-sat changed shape; a non-zero entry \
         after the first means a later call had to redo quantifier work the \
         first call already did"
    );
    assert!(
        EXPECTED_BOUNDARIES[0] > 0,
        "the first call must cross a boundary, or the clause assertion below is \
         vacuous — that is precisely how the previous version of this test failed"
    );

    assert!(
        clauses.windows(2).all(|w| w[0] <= w[1]),
        "the clause database never shrinks between checks: {clauses:?}"
    );
    let plateau = clauses[0];
    assert!(
        clauses.iter().all(|&c| c == plateau),
        "the kept lemmas live in the clause database and are replayed from the \
         SAT trail, never re-encoded, so repeated checks must add no clause at \
         all; got {clauses:?}"
    );
}

/// The goal matrix the repeated-`check-sat` growth pins below run over
/// (task #28).
///
/// Chosen to cover every shape that was observed to grow, plus quiet controls
/// that must stay quiet:
///
/// * `two-quant-arith` — the goal from the original report.  Its growth was the
///   hyper-binary-resolution clauses that no ledger recorded.
/// * `mixed-arith`, `divmod` — `div` / `mod` / numeric-`ite` terms, whose
///   defining axioms are asserted *during* `check`.  These are what a `pop`
///   that dropped the Tseitin memo made the encoder emit all over again.
/// * `arith-heavy` — answers `unknown`, so every `check` used to re-run the
///   whole MBQI round budget; the goal where the late-onset (call ≥ 4) growth
///   showed up.
/// * `two-quant-diseq` — a goal whose *learned* clauses still legitimately move
///   at a late call (ordinary conflict-driven learning), which is why these pins
///   assert on the original count only.
/// * `one-quant`, `multi-round`, `two-quant-three-const`, `nested-impl`,
///   `uflra` — controls that never grew and must not start.
const REPEATED_CHECK_BENCHMARKS: &[(&str, &str)] = &[
    ("one-quant", QUANTIFIED_BENCHMARK),
    ("multi-round", MULTI_ROUND_BENCHMARK),
    (
        "two-quant-three-const",
        r#"
    (set-logic UFLIA)
    (declare-fun f (Int) Int)
    (declare-fun g (Int) Int)
    (declare-const x Int)
    (declare-const y Int)
    (declare-const z Int)
    (assert (or (= x 1) (= x 5)))
    (assert (or (= y 2) (= y 7)))
    (assert (= (f 1) 100))
    (assert (forall ((i Int)) (=> (= (f i) 100) (not (= x i)))))
    (assert (forall ((j Int)) (>= (g j) 0)))
    (assert (= z (g 3)))
"#,
    ),
    (
        "two-quant-arith",
        r#"
    (set-logic UFLIA)
    (declare-fun f (Int) Int)
    (declare-const a Int)
    (declare-const b Int)
    (declare-const c Int)
    (assert (or (= a 1) (= a 2) (= a 3)))
    (assert (forall ((i Int)) (=> (= a i) (> (f i) 10))))
    (assert (forall ((j Int)) (<= (f j) 100)))
    (assert (< (f 1) 5))
    (assert (= b (+ a 1)))
    (assert (= c (* b 2)))
"#,
    ),
    (
        "divmod",
        r#"
    (set-logic UFLIA)
    (declare-const x Int)
    (declare-const y Int)
    (declare-const z Int)
    (assert (> x 0))
    (assert (< x 20))
    (assert (= y (div x 3)))
    (assert (= z (mod x 3)))
    (assert (= z 1))
"#,
    ),
    (
        "mixed-arith",
        r#"
    (set-logic UFLIA)
    (declare-fun f (Int) Int)
    (declare-const a Int)
    (declare-const b Int)
    (declare-const c Int)
    (assert (or (= a 1) (= a 2) (= a 3)))
    (assert (= b (div a 2)))
    (assert (= c (ite (> a 1) (+ b 1) (- b 1))))
    (assert (forall ((i Int)) (=> (= a i) (> (f i) 10))))
    (assert (< (f 1) 5))
"#,
    ),
    (
        "two-quant-diseq",
        r#"
    (set-logic UFLIA)
    (declare-fun f (Int) Int)
    (declare-const p Int)
    (declare-const q Int)
    (assert (or (= p 1) (= p 2) (= p 3)))
    (assert (or (= q 1) (= q 2) (= q 3)))
    (assert (not (= p q)))
    (assert (forall ((i Int)) (=> (= p i) (> (f i) 10))))
    (assert (forall ((j Int)) (=> (= q j) (< (f j) 100))))
"#,
    ),
    (
        "arith-heavy",
        r#"
    (set-logic UFLIA)
    (declare-fun f (Int) Int)
    (declare-fun g (Int Int) Int)
    (declare-const a Int)
    (declare-const b Int)
    (assert (forall ((i Int)) (> (f i) (g i i))))
    (assert (forall ((j Int) (k Int)) (=> (< j k) (< (g j k) (g k j)))))
    (assert (= a (+ b 3)))
    (assert (> (f a) 100))
    (assert (< (g a b) 50))
"#,
    ),
    (
        "nested-impl",
        r#"
    (set-logic UFLIA)
    (declare-const p Bool)
    (declare-const q Bool)
    (declare-const r Bool)
    (assert (=> p (=> q r)))
    (assert (=> r (and p q)))
"#,
    ),
    (
        "uflra",
        r#"
    (set-logic UFLRA)
    (declare-fun h (Real) Real)
    (declare-const u Real)
    (assert (> (h u) 0.0))
"#,
    ),
];

/// How many `check-sat` calls the growth pins below sample.
///
/// Twelve rather than a handful because the growth this pins is *late-onset*:
/// on `arith-heavy` the original-clause count used to hold still for four calls
/// and only then jump, which is exactly why the bug read as non-deterministic
/// when it was first reported.  A three-call pin would have passed.
const REPEATED_CHECKS: usize = 12;

/// Whether the sampler below lets the verdict cache answer a repeated query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerdictCache {
    /// Leave it in force — this is what a real caller sees.
    InForce,
    /// Drop the cached verdict before every iteration, so each `(check-sat)`
    /// runs a real search.
    ///
    /// Without this the machinery *below* the cache is untested on a repeat:
    /// calls 2..12 return in nanoseconds and would keep passing a clause-growth
    /// pin even if `MBQIIntegration::restore_search_state` were deleted.  The
    /// cache is one of the three fixes for task #28; the other two have to be
    /// pinned underneath it.
    Bypassed,
}

/// Sample the original-clause count after each of [`REPEATED_CHECKS`] runs of
/// `script` on a fresh context seeded with `benchmark`.
///
/// Returns `(verdicts, original_clause_counts)`.  `script` is whatever is run
/// per iteration — `(check-sat)` on its own, or with something interleaved
/// before it.
fn sample_original_clauses_over_repeated_checks(
    benchmark: &str,
    script: &str,
    cache: VerdictCache,
) -> (Vec<String>, Vec<usize>) {
    sample_original_clauses(benchmark, script, cache, REPEATED_CHECKS)
}

/// [`sample_original_clauses_over_repeated_checks`] with an explicit number of
/// iterations.
fn sample_original_clauses(
    benchmark: &str,
    script: &str,
    cache: VerdictCache,
    rounds: usize,
) -> (Vec<String>, Vec<usize>) {
    let mut ctx = Context::new();
    ctx.execute_script(benchmark)
        .expect("benchmark should parse and run");

    let mut verdicts = Vec::with_capacity(rounds);
    let mut originals = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        if cache == VerdictCache::Bypassed {
            ctx.solver_mut().forget_cached_verdict();
        }
        let out = ctx.execute_script(script).expect("script should run");
        verdicts.push(out.last().cloned().unwrap_or_default());
        originals.push(ctx.solver().sat.num_original_clauses());
    }
    (verdicts, originals)
}

/// Asking the same question twice must cost nothing: repeated `(check-sat)`
/// calls on an untouched goal must not add a single original clause.
///
/// # The failure mode
///
/// A `check` is allowed to *learn* — conflict clauses, and quantifier
/// instantiation lemmas it keeps for later rounds.  It is not allowed to keep
/// **encoding**.  Every original clause after the first call is a definition or
/// a lemma the solver emitted for a goal that already had one, and it can never
/// be reclaimed: the assertion stack did not move, so no `pop` will ever retract
/// it.  A caller that polls `(check-sat)` in a loop — a portfolio driver, an
/// interactive session, a fuzzer — pays for that forever.
///
/// Three distinct mechanisms produced it (task #28), and all three are pinned
/// here because they are indistinguishable from this vantage point:
///
/// 1. **Unregistered learned clauses.**  On-the-fly hyper-binary resolution
///    (`oxiz_sat::Solver::check_hyper_binary_resolution`) added clauses to the
///    database as learned but recorded them in neither `learned_clause_ids` nor
///    the assertion level's clause list.  They were invisible to
///    `forget_learned_since`, un-retractable by `pop`, and — because
///    `learned_clause_count()` reports the *registry* — made every
///    "`num_clauses()` minus learned" computation misreport them as original.
///    That misreport is what the ticket originally described.  This pin measures
///    `ClauseDatabase::num_original` directly, so it cannot be fooled the same
///    way; `learned_clauses_are_all_registered_in_the_learned_clause_ledger`
///    pins the accounting itself.
/// 2. **Wholesale Tseitin-memo drops.**  See
///    `a_no_op_push_pop_between_checks_does_not_re_encode_the_goal`.
/// 3. **Quantifier-search residue surviving the check that produced it.**  The
///    MBQI candidate pool, instantiation dedup filter and round counter used to
///    outlive their search, so the *next* `check` on the same goal started
///    somewhere else and derived genuinely new lemmas — new terms, new SAT
///    variables, new clauses, several calls in.  `Solver::check` now restores
///    that state and, above it, answers a repeated query from the previous
///    verdict instead of re-running a search that is not idempotent (the SAT
///    solver keeps what it learned, so a re-run hands MBQI a different model).
///
/// # Why the assertion is on original clauses only
///
/// Learned clauses legitimately move between calls and are bounded by the
/// clause-database reduction policy; `two-quant-diseq` in the matrix grows one
/// at call 12 from ordinary conflict-driven learning.  Pinning them would pin
/// the SAT heuristics instead of the bug.
#[test]
fn repeated_checks_on_an_unchanged_goal_add_no_original_clauses() {
    for (name, benchmark) in REPEATED_CHECK_BENCHMARKS {
        let (verdicts, originals) = sample_original_clauses_over_repeated_checks(
            benchmark,
            "(check-sat)",
            VerdictCache::InForce,
        );

        let first_verdict = &verdicts[0];
        assert!(
            verdicts.iter().all(|v| v == first_verdict),
            "{name}: repeating check-sat on an unchanged goal must keep answering \
             the same thing; got {verdicts:?}"
        );

        let plateau = originals[0];
        assert!(
            originals.iter().all(|&c| c == plateau),
            "{name}: repeated check-sat on an unchanged goal must add no original \
             clause after the first call; counts were {originals:?}"
        );
    }
}

/// How many forced re-runs the plateau pin below samples.
///
/// Comfortably more than [`REPEATED_CHECKS`], because what it has to distinguish
/// is a *bounded* step from an *unbounded* trend, and a short window cannot tell
/// them apart.
///
/// Raised from 24 when every numeric `Eq` atom began carrying its trichotomy
/// (`Solver::add_numeric_trichotomy`).  The extra `Lt`/`Gt` atoms give the SAT
/// search more room to end on a different model, so MBQI keeps finding new
/// ground terms for longer before it saturates: `one-quant`'s single step moved
/// from call 0 to call 16 (`10 → 32`, then flat — measured out to call 160).
/// A 24-call window put that step *inside* the flat tail it asserts, which is a
/// mis-calibrated window rather than a regression, so the window moved.
const FORCED_RERUNS: usize = 40;

/// Goals excluded from the convergence pin below, with the reason.
///
/// `arith-heavy` converges, but far too slowly to pin with a fixed window: it
/// climbs `418 → 820` in bounded steps and reaches its plateau at **call 94**,
/// then holds flat through call 160 (measured).  Before the trichotomy it
/// settled at call 5.  It is the `unknown`, two-quantifier goal where MBQI has
/// the most room to diverge, and every extra `Lt`/`Gt` atom is another way for a
/// re-run to end on a different model and instantiate somewhere new; the space
/// is finite, so it does terminate, but a window long enough to show it costs
/// ~30 s of test time for this goal alone.
///
/// Excluding it loses nothing that is not covered exactly elsewhere:
/// `a_check_leaves_the_mbqi_search_state_where_it_found_it` runs over the whole
/// matrix, `arith-heavy` included, and asserts the residue this pin only infers
/// from a clause count is actually gone — non-statistically, at the root cause.
const CONVERGENCE_PIN_EXEMPT: &[&str] = &["arith-heavy"];

/// Repeatedly re-running the search on an unchanged goal must *converge*: the
/// original-clause count may diverge once and must then stop moving forever.
///
/// # Why this exists beside the pin above, and what each of them is for
///
/// The pin above measures what a caller sees, and a caller sees the verdict
/// cache: calls 2..12 there never enter the search at all.  That makes it a good
/// pin on the *fix* and no pin at all on the machinery *underneath* it — it
/// would go on passing if
/// [`MBQIIntegration::restore_search_state`](crate::mbqi::MBQIIntegration::restore_search_state)
/// were deleted tomorrow.  Here the cached verdict is dropped before every call,
/// so all 24 run a real search.
///
/// # Why the property is convergence and not equality
///
/// Re-running is deliberately not idempotent, and equality would be pinning the
/// SAT heuristics rather than the bug.  The SAT solver keeps what it learned, so
/// the second search takes a different route through the same goal, ends on a
/// different model, and hands a model-based instantiator different
/// counterexamples.  On `arith-heavy` — an `unknown` goal with two quantifiers,
/// where MBQI has the most room to diverge — that shows up as exactly one step
/// (474 → 531 original clauses, at the fifth call) which then holds flat through
/// call 24.  Every other goal in the matrix is flat from the first call.
///
/// A one-time step is a re-encoding cost the caller pays once.  What task #28 is
/// about is the *other* shape: a count that keeps climbing because each search
/// starts from the residue of the last one, so there is always somewhere new to
/// go.  That is what "the tail must be flat" rules out, and no fixed equality
/// could, because the two shapes are indistinguishable over three or four calls.
///
/// The exact, non-statistical pin on the same root cause is
/// `a_check_leaves_the_mbqi_search_state_where_it_found_it`, which asserts the
/// residue is gone rather than inferring it from a clause count.  This test is
/// the end-to-end backstop for the resource behaviour a long session actually
/// experiences.
///
/// Note what is deliberately *not* asserted here: verdict stability.  A forced
/// re-run may legitimately answer differently — on an `unknown` goal a second
/// search can get further — and it is the cached path above that owes a caller a
/// stable answer.
#[test]
fn re_running_the_search_on_an_unchanged_goal_converges() {
    /// The call from which the count must already have settled.  Half the
    /// window, so the flat tail is as long as the run-up that produced it.
    const SETTLED_BY: usize = FORCED_RERUNS / 2;

    for (name, benchmark) in REPEATED_CHECK_BENCHMARKS {
        if CONVERGENCE_PIN_EXEMPT.contains(name) {
            continue;
        }
        let (_, originals) = sample_original_clauses(
            benchmark,
            "(check-sat)",
            VerdictCache::Bypassed,
            FORCED_RERUNS,
        );

        assert!(
            originals.windows(2).all(|w| w[0] <= w[1]),
            "{name}: the original-clause count can only grow, so a decrease means \
             this pin is measuring something other than the encoder; counts were \
             {originals:?}"
        );

        let tail = &originals[SETTLED_BY..];
        let plateau = tail[0];
        assert!(
            tail.iter().all(|&c| c == plateau),
            "{name}: re-running the search on a goal the caller has not touched \
             must converge — a count still climbing halfway through the window is \
             a search feeding on its own residue, and every clause it adds is a \
             definition or lemma for a goal that already had one, which no `pop` \
             will ever retract; counts were {originals:?}"
        );
    }
}

/// A `check` hands the quantifier engine back exactly the state it borrowed.
///
/// The direct form of the property the clause-count pins observe indirectly.
/// `Solver::check` snapshots MBQI's per-search state on entry and restores it on
/// exit; what must come back unchanged is the *goal* state (the candidate pool
/// as the caller left it) and what must be reset is the *search* state (the
/// dedup filter, the round counter, the one-shot blind-instantiation guard).
///
/// Asserted per call rather than only at the end, because the two halves of the
/// old bug appear at different times: the residue that makes the next search
/// reach further shows up immediately, while the round counter creeping towards
/// `max_rounds` — after which the goal silently stops being instantiated at all
/// — takes several calls.
#[test]
fn a_check_leaves_the_mbqi_search_state_where_it_found_it() {
    for (name, benchmark) in REPEATED_CHECK_BENCHMARKS {
        let mut ctx = Context::new();
        ctx.execute_script(benchmark)
            .expect("benchmark should parse and run");

        let before = ctx.solver().mbqi.search_state_summary();
        let (candidates, deduped, round, blind) = before;
        assert_eq!(
            (deduped, round, blind),
            (0, 0, false),
            "{name}: no search has run yet, so the search state must be pristine"
        );

        for call in 0..REPEATED_CHECKS {
            ctx.solver_mut().forget_cached_verdict();
            ctx.execute_script("(check-sat)")
                .expect("check-sat should run");
            assert_eq!(
                ctx.solver().mbqi.search_state_summary(),
                before,
                "{name}, call {call}: a finished search must leave neither its \
                 dedup filter, nor its round counter, nor its blind-instantiation \
                 guard, nor the ground terms it harvested for itself behind — the \
                 goal still has exactly the {candidates} candidate(s) the caller \
                 registered"
            );
        }
    }
}

/// Goals excluded from the no-op-`push`/`pop` pin below, with the reason.
///
/// `arith-heavy` answers `unknown`.  `push` and `pop` both invalidate the cached
/// verdict — they are assertion-stack commands, and SMT-LIB puts the solver back
/// into assert mode — so each call there really does re-run the search.  A
/// re-run is not idempotent: the SAT solver keeps its learned clauses, takes a
/// different route, and hands MBQI a different model, which instantiates at
/// different ground terms.  On this goal that is a single bounded step (57
/// clauses at call 5, then flat through call 12 and, measured separately, to
/// call 40) rather than the per-call re-encoding this test is about.  Pinning
/// equality here would pin the SAT heuristics.
///
/// Nothing about `arith-heavy` goes unpinned as a result — the exemption is from
/// *this* test's equality assertion, not from the property:
///
/// * `repeated_checks_on_an_unchanged_goal_add_no_original_clauses` covers it
///   without the interleave (there the cache holds, and equality does apply);
/// * `re_running_the_search_on_an_unchanged_goal_converges` covers it *with*
///   every re-run forced, asserting the weaker property that does hold —
///   the step happens once and the count then stops moving;
/// * `a_check_leaves_the_mbqi_search_state_where_it_found_it` covers it exactly,
///   at the root cause, with no clause counting involved at all.
///
/// `mixed-arith` joined for the same reason, once every numeric `Eq` atom began
/// carrying its trichotomy (`Solver::add_numeric_trichotomy`).  The extra `Lt`
/// / `Gt` atoms enlarge the ground-term space MBQI can reach, so this goal
/// started showing the same re-run non-idempotence `arith-heavy` already did.
///
/// It was checked to be that, and not re-encoding, before being exempted:
///
/// * The steps are `41 → 47` (call 3) and `47 → 54` (call 6), and the count is
///   then flat through call 40 — a bounded, saturating step, not the unbounded
///   per-call copy this test exists to catch.
/// * At every step the *variable* count rises with the clause count
///   (`vars 54 → 58 → 65`, `memo 55 → 59 → 66`).  The failure mode here is
///   "literal-identical clauses over *identical* variables"; new variables mean
///   new content by definition.
/// * The new terms were dumped and inspected.  At call 3 they are
///   `Implies(…)`/`Lt`/`Gt` nodes of a quantifier instantiation at a ground term
///   no earlier call reached, and the trichotomy ledger does **not** grow there
///   — so they are MBQI lemmas, not new splits.  At call 6 one new `Eq` does
///   join the ledger, again at a newly reached ground term.
const RE_ENCODE_PIN_EXEMPT: &[&str] = &["arith-heavy", "mixed-arith"];

/// A `(push 1)(pop 1)` pair between two `(check-sat)` calls changes nothing, and
/// must therefore cost nothing.
///
/// # The failure mode
///
/// `Solver::pop` used to clear the whole Tseitin memo (`encoded_terms`), on the
/// premise that `sat.pop()` retracts the definitional clauses of everything in
/// it.  That premise holds only for terms first encoded *inside* the popped
/// scope.  Entries written at an outer level had their clauses left in place and
/// their memo entry dropped — and since `TrailOp::VarCreated` is journalled, the
/// terms kept their SAT variables too.  The next `check` therefore walked them
/// again and re-emitted literal-identical definitional clauses over identical
/// variables, which `oxiz_sat::Solver::add_clause` (no duplicate detection)
/// appended as new original clauses.
///
/// That is the one mechanism found that was genuinely *unbounded*: one full
/// extra copy of whatever the following check encodes, per `(push)(pop)` pair,
/// with no plateau.  Measured before the fix on this matrix: `mixed-arith` went
/// 28 → 37 → 46 → … → 127 over these twelve calls, and `arith-heavy` 474 → 3267.
/// The memo is now retracted entry by entry (`TrailOp::EncodedTermAdded`), which
/// keeps the outer entries whose clauses survive and restores — rather than
/// drops — an entry whose polarity coverage was widened inside the scope.
///
/// The goals with `div` / `mod` / numeric-`ite` terms carry this test: their
/// defining axioms are asserted from inside `check`, so they are the ones with
/// something left to re-encode after a `pop`.
#[test]
fn a_no_op_push_pop_between_checks_does_not_re_encode_the_goal() {
    for (name, benchmark) in REPEATED_CHECK_BENCHMARKS {
        if RE_ENCODE_PIN_EXEMPT.contains(name) {
            continue;
        }
        let (_, originals) = sample_original_clauses_over_repeated_checks(
            benchmark,
            "(push 1)(pop 1)(check-sat)",
            VerdictCache::InForce,
        );

        let plateau = originals[0];
        assert!(
            originals.iter().all(|&c| c == plateau),
            "{name}: a no-op push/pop leaves the goal identical, so the check that \
             follows must re-encode nothing; counts were {originals:?}"
        );
    }
}

/// Every clause the database counts as learned must be in the registry that
/// makes learned clauses retractable.
///
/// # The failure mode
///
/// `oxiz_sat::Solver` keeps two ledgers beside the clause database:
/// `learned_clause_ids` (what `forget_learned_since` can drop and
/// `learned_clause_count` reports) and the current assertion level's clause list
/// (what `pop` retracts).  `check_hyper_binary_resolution` added its on-the-fly
/// binary clauses to neither.  The consequences ran from cosmetic to unsound:
/// callers computing originals as `num_clauses() - learned_clause_count()`
/// counted them as original clauses (this is what task #28 was reported as); the
/// bit-vector theory's `forget_learned_since` safety net could not forget them;
/// and, since a hyper-binary clause is derived by discharging premises that are
/// false *at level 0*, and level-0 facts here last only as long as the assertion
/// scope that installed them, a clause could outlive the premises it rests on.
///
/// The invariant is checked after repeated checks rather than after one, because
/// the unregistered clauses only appeared on the *second* call: the first call's
/// hyper-binary clauses changed propagation order enough for the second to reach
/// implication pairs the first never did.
#[test]
fn learned_clauses_are_all_registered_in_the_learned_clause_ledger() {
    for (name, benchmark) in REPEATED_CHECK_BENCHMARKS {
        let mut ctx = Context::new();
        ctx.execute_script(benchmark)
            .expect("benchmark should parse and run");
        for round in 0..REPEATED_CHECKS {
            ctx.execute_script("(check-sat)")
                .expect("check-sat should run");
            let sat = &ctx.solver().sat;
            assert_eq!(
                sat.learned_clause_count(),
                sat.num_learned_clauses(),
                "{name}, round {round}: every clause the database flags as learned \
                 must be in `learned_clause_ids`, or it can be neither forgotten \
                 nor retracted — and callers subtracting the registry from the \
                 total will report it as an original clause"
            );
        }
    }
}

/// Crossing an MBQI round boundary costs a *bounded* number of clauses — the
/// round's own new lemmas — and never re-pays for the rounds before it.
///
/// This is the half the test above cannot reach: its benchmark crosses exactly
/// one boundary, so it has no second round to compare against.  Here the first
/// `check-sat` crosses two, and the clause count is sampled at each crossing.
///
/// The failure mode being measured is a rebase that recovered the kept lemmas by
/// re-encoding them through `Solver::encode` instead of replaying the SAT trail.
/// Round `k` would then re-emit every lemma kept from rounds `1..k`, so the
/// per-round increment would *grow with `k`* and the total would be
/// super-linear in the number of rounds — unbounded over a long MBQI loop, whose
/// only cap is `max_mbqi_iterations`.  The assertion is therefore on the shape
/// of the increments (no round costs more than the first), not on a total, which
/// is what makes it sensitive to the design difference rather than to the
/// benchmark's size.
#[test]
fn each_mbqi_round_costs_no_more_clauses_than_the_first() {
    // Measured on this benchmark: two boundaries in the first check.
    const EXPECTED_BOUNDARIES: usize = 2;

    let mut ctx = Context::new();
    ctx.execute_script(MULTI_ROUND_BENCHMARK)
        .expect("benchmark should parse and run");
    let out = ctx
        .execute_script("(check-sat)")
        .expect("check-sat should run");
    assert!(
        out.iter().any(|line| line == "sat"),
        "the multi-round benchmark is satisfiable, got {out:?}"
    );

    let samples = ctx.solver().mbqi_round_clauses.clone();
    assert_eq!(
        samples.len(),
        EXPECTED_BOUNDARIES,
        "this benchmark's whole purpose is to cross more than one round \
         boundary; sampled clause counts were {samples:?}"
    );

    // Clause counts at: round 1's boundary, round 2's boundary, and the end of
    // the search.  The increments between them are what each round cost.
    let mut marks = samples.clone();
    marks.push(ctx.solver().sat.num_clauses());
    let increments: Vec<usize> = core::iter::once(marks[0])
        .chain(marks.windows(2).map(|w| w[1] - w[0]))
        .collect();

    let first = increments[0];
    assert!(
        increments.iter().all(|&inc| inc <= first),
        "no MBQI round may cost more clauses than the first: an increment that \
         grows with the round index is the signature of re-encoding the kept \
         lemmas instead of replaying them; increments were {increments:?} over \
         marks {marks:?}"
    );
}
