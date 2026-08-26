//! Regression tests for [`super::skolem_candidates`]'s Skolem-candidate walk.
//!
//! Covers two changes to `collect_skolem_candidates_rec`:
//!
//! 1. The conversion from native recursion to an explicit worklist (so a
//!    pathologically deep term hits a graceful, bounded-native-stack loop
//!    instead of a fatal stack overflow no `Result` can report).
//! 2. Making the walk's `match` **exhaustive over every `TermKind`**, with no
//!    `_` catch-all. Before this change the walk covered only `Apply`,
//!    `Forall`/`Exists`, `And`/`Or`, `Not`/`Neg`,
//!    `Implies`/`Eq`/`Lt`/`Le`/`Gt`/`Ge`/`Sub`/`Div`/`Mod`, `Add`/`Mul`,
//!    `Ite`, and `Select`/`Store`; every BitVector operation, every
//!    FloatingPoint operation, every String operation, `Distinct`, `Let`,
//!    `Match`, `Xor`, and the datatype constructor/tester/selector forms fell
//!    into a `_ => {}` catch-all and were silently never descended into. A
//!    Skolem application nested under any of those was therefore never found
//!    as an MBQI candidate.
//!
//! Split out as a child module of `encode` (rather than appended inline) so
//! that `encode.rs` -- already close to the workspace's 2000-line-per-file
//! ceiling -- does not have to absorb the new test bulk; see
//! `euf/solver/tests.rs` for the identical precedent (file `X.rs` + child
//! module `X/tests.rs`).
//!
//! Two different inspection mechanisms are used below, deliberately:
//!
//! * The `visited: FxHashSet<TermId>` set the caller threads through records
//!   exactly which terms the walk reached, independent of whether any of
//!   them looked like a Skolem application. This is what the *preservation*
//!   tests use, since they are about the shape of the traversal itself.
//! * `MBQIIntegration::extra_candidates_snapshot` (a `#[cfg(test)]`-gated
//!   accessor added alongside this change, since `extra_candidates` is
//!   private to a different module tree -- `mbqi::integration` -- that
//!   plain module-descendant visibility does not reach) reports the actual
//!   MBQI candidate pool `self.mbqi.add_candidate` populated. This is what
//!   the *newly covered family* tests use below, since they are about the
//!   real, end-to-end, observable behaviour change: does a Skolem
//!   application under a bitvector/floating-point/string/etc. operation
//!   actually reach the candidate pool now.
use super::*;

/// Depth `n`: `wrap(wrap(wrap(...wrap(leaf)...)))`, built with a plain
/// iterative loop -- never a recursive helper, which would overflow before
/// the walk under test even ran.
///
/// `TermManager::mk_apply` performs no simplification (unlike, say, `mk_not`
/// or `mk_and`, which fold double negation and flatten nested conjunctions
/// respectively and so cannot be used to build genuine deep *nesting*): each
/// level's argument is the unique previous level's `TermId`, so every `wrap`
/// application is content-distinct and hash-consing cannot collapse the
/// chain.
fn build_apply_chain(manager: &mut TermManager, depth: usize) -> TermId {
    let int_sort = manager.sorts.int_sort;
    let mut term = manager.mk_var("leaf", int_sort);
    for _ in 0..depth {
        term = manager.mk_apply("wrap", [term], int_sort);
    }
    term
}

/// The point of the whole exercise: an embedder calling OxiZ from a worker
/// thread with a conventional ~1 MiB stack must get a normal return, not a
/// process abort.  The pinned stack here is an eighth of that, paired with an
/// eighth of the depth, which pins the same bytes-per-frame ratio.
///
/// A Rust stack overflow is not a panic -- it is a fatal runtime abort that
/// `catch_unwind` cannot intercept -- so the only way to assert on it is to
/// run on a thread whose stack size is pinned small and observe that the
/// thread returns at all. "Returned at all" is necessary but not sufficient:
/// asserting the exact `visited` size additionally rules out a silent
/// partial walk (the same "unhandled input silently dropped" failure mode
/// this whole audit targets), which a bare depth cap could have produced.
/// This exercises the `Apply` arm repeatedly; the exhaustive-match rewrite
/// does not change how any single arm behaves, only how many arms exist, so
/// this test's exact expected count is unaffected by that rewrite.
#[test]
fn skolem_candidate_walk_survives_a_small_stack() {
    // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
    // ~10 B-per-frame threshold is the pin, so never raise one alone.
    const STACK_SIZE: usize = 1 << 17; // 128 KiB
    const DEPTH: usize = 12_500;

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let mut solver = Solver::new();
            let mut manager = TermManager::new();
            let deepest = build_apply_chain(&mut manager, DEPTH);

            let mut visited = FxHashSet::default();
            solver.collect_skolem_candidates_rec(deepest, &manager, &mut visited);

            // DEPTH `wrap` applications plus the one `leaf` variable.
            assert_eq!(
                visited.len(),
                DEPTH + 1,
                "the walk must reach every level of a {DEPTH}-deep chain, not stop partway"
            );
        })
        .expect("spawning a 128 KiB-stack thread should succeed");

    handle
        .join()
        .expect("the skolem-candidate walk must return on 128 KiB instead of overflowing it");
}

// ---------------------------------------------------------------------
// Behaviour preservation: families already covered before this change.
// ---------------------------------------------------------------------

/// Shallow, hand-checkable formula exercising connectives already covered
/// before this change (`And`, `Eq`, `Forall`, `Apply`), pinning the exact
/// reached set to prove the exhaustive-match rewrite leaves their handling
/// unchanged.
///
/// Shape (all under one top-level `forall`):
/// ```text
/// forall z .
///   and(
///     eq(sk!0(x), y),        -- Eq is covered: descends into both sides
///     normal_fn(x),          -- Apply is covered, but name has no sk/skf prefix
///   )
/// ```
#[test]
fn skolem_candidate_walk_preserves_previously_covered_connectives() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);

    // Reached through `Eq`, itself reached through `And`, itself reached
    // through `Forall`'s body.
    let sk_apply = manager.mk_apply("sk!0", [x], int_sort);
    let eq_atom = manager.mk_eq(sk_apply, y);

    // Reached through `And` (Apply is covered), but its name does not start
    // with "sk"/"skf" and so must never be treated as a candidate.
    let normal_apply = manager.mk_apply("normal_fn", [x], int_sort);

    let and_term = manager.mk_and(vec![eq_atom, normal_apply]);
    let forall_term = manager.mk_forall(vec![("z", int_sort)], and_term);

    let mut visited = FxHashSet::default();
    solver.collect_skolem_candidates_rec(forall_term, &manager, &mut visited);

    // Reached: the quantifier, the conjunction, both operands of the
    // equality (including the Skolem application and its own argument `x`),
    // the plain equality-comparison target `y`, and the harmless `Apply`
    // together with *its* argument `x` (already visited, deduped by the
    // `visited` set -- not double-counted).
    for reached in [forall_term, and_term, eq_atom, sk_apply, x, y, normal_apply] {
        assert!(
            visited.contains(&reached),
            "expected the walk to reach {reached:?}"
        );
    }

    // Precisely which Apply terms in the reached set would be registered as
    // MBQI candidates (fname starting with "sk"/"skf"): exactly `sk_apply`,
    // not `normal_apply`.
    let sk_prefixed_applies: Vec<TermId> = visited
        .iter()
        .copied()
        .filter(|&id| {
            manager.get(id).is_some_and(|t| {
                matches!(&t.kind, TermKind::Apply { func, .. }
                    if { let n = manager.resolve_str(*func); n.starts_with("sk") || n.starts_with("skf") })
            })
        })
        .collect();
    assert_eq!(
        sk_prefixed_applies,
        vec![sk_apply],
        "exactly one sk-prefixed Apply must be reachable: sk_apply itself"
    );
}

/// Second behaviour-preservation test: exercises the previously-covered
/// connectives the test above does not touch (`Not`, `Or`, `Implies`, `Ite`,
/// `Lt`/`Le`/`Gt`/`Ge`, `Sub`/`Div`/`Mod`, `Add`/`Mul`, `Exists`), asserting
/// that a Skolem application nested deep beneath all of them is still found.
/// None of these arms' internal logic changed in the exhaustive-match
/// rewrite (they were copied over unchanged, just regrouped alongside the
/// newly added arms), and this test is the empirical check of that claim.
#[test]
fn skolem_candidate_walk_preserves_remaining_previously_covered_connectives() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let sk_apply = manager.mk_apply("sk!9", [x], int_sort);
    let one = manager.mk_int(1);

    let add_t = manager.mk_add(vec![sk_apply, one]); // Add
    let sub_t = manager.mk_sub(add_t, one); // Sub
    let mul_t = manager.mk_mul(vec![sub_t, one]); // Mul
    let div_t = manager.mk_div(mul_t, one); // Div
    let mod_t = manager.mk_mod(div_t, one); // Mod

    let lt_t = manager.mk_lt(mod_t, one); // Lt
    let le_t = manager.mk_le(mod_t, one); // Le
    let gt_t = manager.mk_gt(mod_t, one); // Gt
    let ge_t = manager.mk_ge(mod_t, one); // Ge

    let ite_t = manager.mk_ite(lt_t, le_t, gt_t); // Ite (le_t != gt_t, so this is not folded away)
    let not_t = manager.mk_not(ge_t); // Not
    let or_t = manager.mk_or(vec![ite_t, not_t]); // Or
    let implies_t = manager.mk_implies(or_t, ite_t); // Implies

    let exists_t = manager.mk_exists(vec![("w", int_sort)], implies_t); // Exists

    solver.collect_skolem_candidates(exists_t, &manager);

    let candidates = solver.mbqi.extra_candidates_snapshot(int_sort);
    assert!(
        candidates.contains(&sk_apply),
        "sk!9(x) nested under Add/Sub/Mul/Div/Mod/Lt/Le/Gt/Ge/Ite/Not/Or/Implies/Exists \
         must still be registered as an MBQI candidate after the exhaustive-match rewrite"
    );
}

// ---------------------------------------------------------------------
// Newly covered families: each was in the old `_ => {}` catch-all and so
// could never find a nested Skolem application before this change.
// ---------------------------------------------------------------------

/// Regression: every BitVector `TermKind` fell into the pre-fix `_ => {}`
/// catch-all -- this is the task's own motivating example, `(bvadd (sk!0 x)
/// y)`, generalized slightly to `BvUlt` so the Skolem application's operation
/// is directly the (already Bool-sorted) quantifier body, with no extra
/// wrapper connective needed.
#[test]
fn skolem_candidate_found_under_bitvector_operation() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);

    let x = manager.mk_var("x", bv8);
    let y = manager.mk_var("y", bv8);
    let sk_apply = manager.mk_apply("sk!0", [x], bv8);
    let body = manager.mk_bv_ult(sk_apply, y);
    let forall = manager.mk_forall(vec![("x", bv8)], body);

    solver.collect_skolem_candidates(forall, &manager);

    let candidates = solver.mbqi.extra_candidates_snapshot(bv8);
    assert!(
        candidates.contains(&sk_apply),
        "sk!0(x) nested under BvUlt must be registered as an MBQI candidate \
         (pre-fix, BvUlt fell into `_ => {{}}` and this would have been missed)"
    );
}

/// Regression: every FloatingPoint `TermKind` fell into the pre-fix `_ =>
/// {}` catch-all. `FpLeq` is Bool-sorted already and (comparisons being
/// exact) needs no rounding-mode argument, so it is used directly as the
/// quantifier body.
#[test]
fn skolem_candidate_found_under_floating_point_operation() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let fp_sort = manager.sorts.float_sort(11, 53); // IEEE 754 binary64

    let x = manager.mk_var("x", fp_sort);
    let y = manager.mk_var("y", fp_sort);
    let sk_apply = manager.mk_apply("sk!7", [x], fp_sort);
    let body = manager.mk_fp_leq(sk_apply, y);
    let forall = manager.mk_forall(vec![("x", fp_sort)], body);

    solver.collect_skolem_candidates(forall, &manager);

    let candidates = solver.mbqi.extra_candidates_snapshot(fp_sort);
    assert!(
        candidates.contains(&sk_apply),
        "sk!7(x) nested under FpLeq must be registered as an MBQI candidate \
         (pre-fix, FpLeq fell into `_ => {{}}` and this would have been missed)"
    );
}

/// Regression: every String `TermKind` fell into the pre-fix `_ => {}`
/// catch-all -- this is the task's own motivating example, `(str.++ (sk!1
/// x) y)`, generalized to `StrContains` so it is directly the (already
/// Bool-sorted) quantifier body.
#[test]
fn skolem_candidate_found_under_string_operation() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let string_sort = manager.sorts.string_sort();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", string_sort);
    let sk_apply = manager.mk_apply("sk!8", [x], string_sort);
    let body = manager.mk_str_contains(sk_apply, y);
    let forall = manager.mk_forall(vec![("x", int_sort)], body);

    solver.collect_skolem_candidates(forall, &manager);

    let candidates = solver.mbqi.extra_candidates_snapshot(string_sort);
    assert!(
        candidates.contains(&sk_apply),
        "sk!8(x) nested under StrContains must be registered as an MBQI candidate \
         (pre-fix, StrContains fell into `_ => {{}}` and this would have been missed)"
    );
}

/// Regression: `TermKind::Distinct` fell into the pre-fix `_ => {}`
/// catch-all -- this is exactly the task's own motivating example,
/// `(distinct (sk!1 x) y)`.
#[test]
fn skolem_candidate_found_under_distinct() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let sk_apply = manager.mk_apply("sk!1", [x], int_sort);
    let distinct_term = manager.mk_distinct(vec![sk_apply, y]);
    let forall = manager.mk_forall(vec![("x", int_sort)], distinct_term);

    solver.collect_skolem_candidates(forall, &manager);

    let candidates = solver.mbqi.extra_candidates_snapshot(int_sort);
    assert!(
        candidates.contains(&sk_apply),
        "sk!1(x) nested under Distinct must be registered as an MBQI candidate \
         (pre-fix, Distinct fell into `_ => {{}}` and this would have been missed)"
    );
}

/// Regression: `TermKind::Let` fell into the pre-fix `_ => {}` catch-all --
/// this is exactly the task's own motivating example, `(let ((a (sk!2 x)))
/// ...)`. The Skolem application sits in the *bound value* position; the
/// walk must descend into bound values, not just the body.
#[test]
fn skolem_candidate_found_under_let_binding() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let sk_apply = manager.mk_apply("sk!2", [x], int_sort);
    let let_body = manager.mk_true();
    let let_term = manager.mk_let(vec![("a", sk_apply)], let_body);
    let forall = manager.mk_forall(vec![("x", int_sort)], let_term);

    solver.collect_skolem_candidates(forall, &manager);

    let candidates = solver.mbqi.extra_candidates_snapshot(int_sort);
    assert!(
        candidates.contains(&sk_apply),
        "sk!2(x) bound in a Let must be registered as an MBQI candidate \
         (pre-fix, Let fell into `_ => {{}}` and this would have been missed)"
    );
}

/// Regression: `TermKind::Match` fell into the pre-fix `_ => {}` catch-all.
///
/// There is no `mk_match` smart constructor in `oxiz-core`, and `MatchCase`
/// is not part of its public API surface (it is not re-exported from
/// `oxiz_core::ast`, and the module that defines it is private) -- so from
/// this crate a `Match` term can only be built via the public `intern_term`
/// escape hatch, and only with an *empty* case list: `SmallVec::new()`
/// infers its element type from the field it is assigned to, so it never
/// needs to name `MatchCase` at all, unlike constructing an actual
/// `MatchCase` value would. This still genuinely exercises the arm's
/// traversal of `scrutinee` -- the position the Skolem application is placed
/// in below -- and is a real, executable proof for that half of the arm;
/// the other half (pushing each case's body) is exercised only by this
/// match having to compile exhaustively and by code review, not by a
/// runtime assertion here, because building a non-empty case list is not
/// possible from outside `oxiz-core` without changing that crate.
#[test]
fn skolem_candidate_found_under_match_scrutinee() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let bool_sort = manager.sorts.bool_sort;

    let x = manager.mk_var("x", int_sort);
    let sk_apply = manager.mk_apply("sk!5", [x], int_sort);
    let match_term = manager.intern_term(
        TermKind::Match {
            scrutinee: sk_apply,
            cases: SmallVec::new(),
        },
        bool_sort,
    );
    let forall = manager.mk_forall(vec![("x", int_sort)], match_term);

    solver.collect_skolem_candidates(forall, &manager);

    let candidates = solver.mbqi.extra_candidates_snapshot(int_sort);
    assert!(
        candidates.contains(&sk_apply),
        "sk!5(x) used as a Match scrutinee must be registered as an MBQI candidate \
         (pre-fix, Match fell into `_ => {{}}` and this would have been missed)"
    );
}

/// Regression: `TermKind::Xor` fell into the pre-fix `_ => {}` catch-all.
#[test]
fn skolem_candidate_found_under_xor() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;

    let x = manager.mk_var("x", bool_sort);
    let y = manager.mk_var("y", bool_sort);
    let sk_apply = manager.mk_apply("sk!6", [x], bool_sort);
    let xor_term = manager.mk_xor(sk_apply, y);
    let forall = manager.mk_forall(vec![("x", bool_sort)], xor_term);

    solver.collect_skolem_candidates(forall, &manager);

    let candidates = solver.mbqi.extra_candidates_snapshot(bool_sort);
    assert!(
        candidates.contains(&sk_apply),
        "sk!6(x) nested under Xor must be registered as an MBQI candidate \
         (pre-fix, Xor fell into `_ => {{}}` and this would have been missed)"
    );
}

// =====================================================================
// Tseitin-encoder hardening: memoisation (`Solver::encoded_terms`),
// iterative `extract_linear_terms`, iterative arith-split walks, and the
// depth-cap stack-budget measurement.
// =====================================================================

use super::super::{SolverConfig, SolverResult};
use num_rational::Rational64;
use num_traits::Zero;
use smallvec::SmallVec;

/// Depth `n` chain of implications `b_n => (b_{n-1} => (... => b_0))`, built
/// iteratively.  `mk_implies` folds only constant operands, so every level of
/// this chain is a distinct interned node and genuine nesting survives (the
/// n-ary `mk_and`/`mk_or` builders flatten nested conjunctions/disjunctions
/// and cannot be used here).
fn build_implies_chain(manager: &mut TermManager, depth: usize) -> TermId {
    let bool_sort = manager.sorts.bool_sort;
    let mut term = manager.mk_var("b0", bool_sort);
    for i in 1..=depth {
        let v = manager.mk_var(&format!("b{i}"), bool_sort);
        term = manager.mk_implies(v, term);
    }
    term
}

/// Doubling DAG: level `i+1` references level `i` **twice**, alternating
/// `And`/`Or` so the flattening `mk_and`/`mk_or` builders never merge a level
/// into its parent (they flatten only same-kind children; neither dedupes a
/// repeated argument).  `levels` levels give `levels` interned nodes but
/// `2^levels` root-to-leaf paths.
fn build_doubling_dag(manager: &mut TermManager, levels: usize) -> TermId {
    let bool_sort = manager.sorts.bool_sort;
    let mut term = manager.mk_var("leaf", bool_sort);
    for i in 0..levels {
        term = if i % 2 == 0 {
            manager.mk_and([term, term])
        } else {
            manager.mk_or([term, term])
        };
    }
    term
}

/// The memo must make the encoder visit each DAG *node* once, not each
/// *path*: 60 doubling levels are `2^60` paths, so this test finishing at all
/// (in milliseconds, with a clause count linear in the node count) is the
/// proof that shared sub-terms are no longer re-encoded.  Before the memo the
/// encoder hung at roughly depth 40 while emitting `2^n` duplicate clauses;
/// the assert-time `term_exceeds_encode_depth` pre-check cannot catch this
/// input because it measures *depth* (60 here) and prunes shared nodes.
#[test]
fn tseitin_memo_encodes_each_doubling_dag_node_once() {
    const LEVELS: usize = 60;

    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let root = build_doubling_dag(&mut manager, LEVELS);

    let lit = solver.encode(root, &mut manager);
    let clauses_after_first = solver.sat.num_clauses();

    // Every And/Or node emits at most 1 + arity clauses (both implication
    // directions at `Polarity::Both`); arity is 2 throughout, the leaf emits
    // none.  Anything superlinear in LEVELS means a node was re-encoded.
    assert!(
        clauses_after_first <= 4 * LEVELS + 8,
        "clause count must be linear in DAG nodes, got {clauses_after_first} \
         for {LEVELS} levels (exponential re-encoding?)"
    );

    // A second encode of the same root is a pure memo hit: the identical
    // literal comes back and no clause is re-emitted.
    let lit_again = solver.encode(root, &mut manager);
    assert_eq!(lit, lit_again, "memo hit must return the original literal");
    assert_eq!(
        solver.sat.num_clauses(),
        clauses_after_first,
        "a memo hit must not re-emit clauses"
    );
}

/// Soundness pin for the memo's polarity key.
///
/// `f = (or p q)` is first encoded under *negative* polarity (antecedent of
/// an implication): with `polarity_aware` that emits only the `arg => f`
/// clauses.  Asserting `f` itself afterwards widens `f`'s polarity to `Both`,
/// and the encoder must *re-encode* `f` to emit the missing `f => (p or q)`
/// direction.  A memo keyed on `TermId` alone would return the cached literal
/// and skip those clauses, leaving `f` under-constrained — this formula
/// (`(f => c) ∧ f ∧ ¬p ∧ ¬q`, unsatisfiable because `f = p ∨ q`) would then
/// come back `Sat`.
#[test]
fn tseitin_memo_polarity_widening_re_encodes_missing_direction() {
    let config = SolverConfig {
        // Pin the mechanism: keep the assertion terms exactly as built so the
        // widening path (not the simplifier) is what is under test.
        simplify: false,
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;

    let p = manager.mk_var("p", bool_sort);
    let q = manager.mk_var("q", bool_sort);
    let c = manager.mk_var("c", bool_sort);
    let f = manager.mk_or([p, q]);

    // 1. `f` occurs only negatively: encoded under `Polarity::Negative`.
    let impl_term = manager.mk_implies(f, c);
    solver.assert(impl_term, &mut manager);
    // 2. Asserting `f` widens it to `Both`: the memo entry from step 1 no
    //    longer covers it and the missing direction must be emitted.
    solver.assert(f, &mut manager);
    // 3. Refute both disjuncts.
    let not_p = manager.mk_not(p);
    let not_q = manager.mk_not(q);
    solver.assert(not_p, &mut manager);
    solver.assert(not_q, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Unsat,
        "(f => c) ∧ f ∧ ¬p ∧ ¬q with f = (p ∨ q) is unsat; `Sat` here means \
         the widened polarity was not re-encoded (memo key ignored polarity)"
    );
}

/// Companion satisfiable pin: sharing a sub-term across assertions (memo hits
/// on the second occurrence) must not change the verdict.
#[test]
fn tseitin_memo_shared_subterm_keeps_sat_verdict() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;

    let p = manager.mk_var("p", bool_sort);
    let q = manager.mk_var("q", bool_sort);
    let f = manager.mk_or([p, q]);
    let g = manager.mk_and([f, q]);

    solver.assert(f, &mut manager); // encodes f
    solver.assert(g, &mut manager); // re-uses f via the memo

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Sat,
        "(p ∨ q) ∧ ((p ∨ q) ∧ q) is satisfiable (q = true)"
    );
}

/// End-to-end honesty pin on a small worker-thread stack: a 12 500-deep
/// assertion (still 24x past [`super::super::ENCODE_DEPTH_LIMIT`]) must
/// produce a normal `Unknown`, never a process abort.  The assert-time
/// `term_exceeds_encode_depth` pre-check (explicit stack) flags the term
/// before any recursive pass sees it, and `check_core` consults
/// `encode_depth_exceeded` *first* — before the axiom instantiators and the
/// five theory collectors walk the over-deep assertion.  A stack overflow is
/// a fatal abort `catch_unwind` cannot intercept, so this thread returning at
/// all is the assertion.
#[test]
fn deep_assertion_answers_unknown_on_a_small_stack() {
    // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
    // ~10 B-per-frame threshold is the pin, so never raise one alone.
    const STACK_SIZE: usize = 1 << 17; // 128 KiB
    const DEPTH: usize = 12_500;

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let mut solver = Solver::new();
            let mut manager = TermManager::new();
            let deep = build_implies_chain(&mut manager, DEPTH);

            solver.assert(deep, &mut manager);
            assert!(
                solver.encode_depth_exceeded,
                "a {DEPTH}-deep assertion must trip the encode-depth guard"
            );
            assert_eq!(
                solver.check(&mut manager),
                SolverResult::Unknown,
                "a truncated encoding must answer Unknown, not guess"
            );
        })
        .expect("spawning a 128 KiB-stack thread should succeed");

    handle
        .join()
        .expect("deep assert+check must return on a 128 KiB stack instead of overflowing it");
}

/// Stack-budget measurement for the recursive Tseitin encoder (the one
/// recursive pass that legitimately keeps its depth cap, because the cap
/// reports through a real error channel: `encode_depth_exceeded` →
/// `Unknown`).
///
/// A chain exactly [`super::super::ENCODE_DEPTH_LIMIT`] deep passes the
/// pre-check, so `encode_depth` recurses to the full cap.  Running it on a
/// 1 MiB thread proves the cap fires *before* the native stack dies on the
/// smallest stack an embedder plausibly hands us — at the cap's value at the
/// time of writing, measured at `opt-level = 1` (this workspace's dev
/// profile), the whole at-cap descent fits with headroom; see the constant's
/// doc comment for the measured numbers.
///
/// This is the one deep-nesting test in the crate whose stack must **not** be
/// scaled down with the others: it exercises a *deliberately recursive* pass
/// at a fixed depth (512, set by the production constant, not by a stack
/// ratio), and the measured requirement is >= 384 KiB.  A 128 KiB stack would
/// turn a passing test into a process abort.
#[test]
fn encode_at_cap_depth_survives_a_one_mib_stack() {
    // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — deliberately
    // recursive pass at ENCODE_DEPTH_LIMIT (512), measured to need at
    // least 384 KiB. See TODO.md "v0.3.2 backlog".
    const STACK_SIZE: usize = 1 << 20; // 1 MiB

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let mut solver = Solver::new();
            let mut manager = TermManager::new();
            let depth = super::super::ENCODE_DEPTH_LIMIT as usize;
            let chain = build_implies_chain(&mut manager, depth);

            let lit = solver.encode(chain, &mut manager);
            let _ = lit;
            assert!(
                !solver.encode_depth_exceeded,
                "a chain exactly at the cap must encode completely (the guard \
                 is `depth > LIMIT`, and the pre-check already admitted it)"
            );
        })
        .expect("spawning a 1 MiB-stack thread should succeed");

    handle
        .join()
        .expect("an at-cap encode must return on a 1 MiB stack instead of overflowing it");
}

/// `extract_linear_terms` is reached with the *whole operand* of a shallow
/// comparison atom (`(< chain 0)` is depth 2 for the encoder but `DEPTH`
/// for the linear parser), and its frames used to stack on top of
/// `encode_depth`'s — so it must never recurse natively.  The deep `Sub`
/// chain doubles as a semantic pin: `((x - 1) - 1) - ...` must fold to
/// exactly `x - DEPTH`, proving the conversion did not perturb scales.
#[test]
fn extract_linear_terms_deep_sub_chain_on_a_small_stack() {
    // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
    // ~10 B-per-frame threshold is the pin, so never raise one alone.
    const STACK_SIZE: usize = 1 << 17; // 128 KiB
    const DEPTH: usize = 12_500;

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let solver = Solver::new();
            let mut manager = TermManager::new();
            let int_sort = manager.sorts.int_sort;
            let x = manager.mk_var("x", int_sort);
            let one = manager.mk_int(1);
            let mut chain = x;
            for _ in 0..DEPTH {
                chain = manager.mk_sub(chain, one);
            }

            let mut terms: SmallVec<[(TermId, Rational64); 4]> = SmallVec::new();
            let mut constant = Rational64::zero();
            let ok = solver.extract_linear_terms(
                chain,
                Rational64::from_integer(1),
                &mut terms,
                &mut constant,
                &manager,
            );

            assert_eq!(ok, Some(()), "a deep Sub chain is linear");
            assert_eq!(
                terms.as_slice(),
                &[(x, Rational64::from_integer(1))],
                "the chain contributes exactly x with coefficient 1"
            );
            assert_eq!(
                constant,
                Rational64::from_integer(-(DEPTH as i64)),
                "each of the {DEPTH} subtractions contributes -1"
            );
        })
        .expect("spawning a 128 KiB-stack thread should succeed");

    handle
        .join()
        .expect("extract_linear_terms must return on a 128 KiB stack instead of overflowing it");
}

/// Same, for the one arm that carries resume state: `Mul` factors are
/// evaluated into per-factor contexts, and `(...((x * 1) * 1)...) * 1`
/// nests one `MulFrame` per level.  `mk_mul` does not fold a unit factor and
/// does not flatten nested products, so the nesting is genuine.  Multiplying
/// by 1 keeps the pinned coefficient exact at any depth (a factor of 2 would
/// overflow `i64` long before the depth matters).
#[test]
fn extract_linear_terms_deep_nested_mul_on_a_small_stack() {
    // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
    // ~10 B-per-frame threshold is the pin, so never raise one alone.
    const STACK_SIZE: usize = 1 << 17; // 128 KiB
    const DEPTH: usize = 12_500;

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let solver = Solver::new();
            let mut manager = TermManager::new();
            let int_sort = manager.sorts.int_sort;
            let x = manager.mk_var("x", int_sort);
            let one = manager.mk_int(1);
            let mut chain = x;
            for _ in 0..DEPTH {
                chain = manager.mk_mul([chain, one]);
            }

            let mut terms: SmallVec<[(TermId, Rational64); 4]> = SmallVec::new();
            let mut constant = Rational64::zero();
            let ok = solver.extract_linear_terms(
                chain,
                Rational64::from_integer(1),
                &mut terms,
                &mut constant,
                &manager,
            );

            assert_eq!(ok, Some(()), "x multiplied by 1 at any depth is linear");
            assert_eq!(
                terms.as_slice(),
                &[(x, Rational64::from_integer(1))],
                "the nested unit products contribute exactly x"
            );
            assert!(constant.is_zero());
        })
        .expect("spawning a 128 KiB-stack thread should succeed");

    handle
        .join()
        .expect("nested-Mul extract_linear_terms must return on 128 KiB instead of overflowing it");
}

/// Behaviour pins for the iterative `extract_linear_terms`: exact
/// coefficient lists (including order — callers see the append order),
/// constant folding through `Mul`, and every nonlinear reject the recursive
/// version produced.
#[test]
fn extract_linear_terms_semantic_pins() {
    let solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let z = manager.mk_var("z", int_sort);

    let run = |manager: &TermManager, term: TermId| {
        let mut terms: SmallVec<[(TermId, Rational64); 4]> = SmallVec::new();
        let mut constant = Rational64::zero();
        let ok = solver.extract_linear_terms(
            term,
            Rational64::from_integer(1),
            &mut terms,
            &mut constant,
            manager,
        );
        (ok, terms, constant)
    };

    // 2*x + (y - 1) - (-z)  ==>  [(x,2), (y,1), (z,1)], constant -1,
    // in exactly that (left-to-right) append order.
    let two = manager.mk_int(2);
    let one = manager.mk_int(1);
    let two_x = manager.mk_mul([two, x]);
    let y_minus_1 = manager.mk_sub(y, one);
    let neg_z = manager.mk_neg(z);
    let sum = manager.mk_add([two_x, y_minus_1]);
    let expr = manager.mk_sub(sum, neg_z);
    let (ok, terms, constant) = run(&manager, expr);
    assert_eq!(ok, Some(()));
    assert_eq!(
        terms.as_slice(),
        &[
            (x, Rational64::from_integer(2)),
            (y, Rational64::from_integer(1)),
            (z, Rational64::from_integer(1)),
        ],
        "coefficients and append order must match the recursive version"
    );
    assert_eq!(constant, Rational64::from_integer(-1));

    // A pure-constant product folds into the constant: 3 * 2 => 6.
    let three = manager.mk_int(3);
    let const_prod = manager.mk_mul([three, two]);
    let (ok, terms, constant) = run(&manager, const_prod);
    assert_eq!(ok, Some(()));
    assert!(terms.is_empty());
    assert_eq!(constant, Rational64::from_integer(6));

    // Nonlinear rejects (all `None`, exactly as before):
    // two variable factors,
    let xy = manager.mk_mul([x, y]);
    assert_eq!(run(&manager, xy).0, None, "x*y is nonlinear");
    // a factor that is linear-with-offset,
    let one_plus_x = manager.mk_add([one, x]);
    let offset_prod = manager.mk_mul([one_plus_x, y]);
    assert_eq!(run(&manager, offset_prod).0, None, "(1+x)*y is nonlinear");
    // and a multi-variable factor.
    let x_plus_y = manager.mk_add([x, y]);
    let multi_prod = manager.mk_mul([x_plus_y, two]);
    assert_eq!(
        run(&manager, multi_prod).0,
        None,
        "(x+y)*2 has a multi-variable factor and was always rejected"
    );

    // Failure leaves the caller's buffers untouched (the recursive version
    // left partial writes; the only caller discards them on None, so the
    // cleaner behaviour is safe — pin it so it stays deliberate).
    let mut terms: SmallVec<[(TermId, Rational64); 4]> = SmallVec::new();
    let mut constant = Rational64::zero();
    let pre_seeded = (z, Rational64::from_integer(7));
    terms.push(pre_seeded);
    let ok = solver.extract_linear_terms(
        xy,
        Rational64::from_integer(1),
        &mut terms,
        &mut constant,
        &manager,
    );
    assert_eq!(ok, None);
    assert_eq!(terms.as_slice(), &[pre_seeded]);
    assert!(constant.is_zero());
}

/// Semantic pin for the **single owner** of arithmetic-disequality
/// enforcement, `Solver::add_numeric_trichotomy`.
///
/// This replaces two older tests that drove the four syntactic walks
/// (`add_arith_diseq_split`, `add_arith_trichotomy_clause`,
/// `add_arith_eq_trichotomy`, `add_arith_diseq_splits_for_sat_model`) directly.
/// Those walks are deleted: every one of their call sites called `encode` on
/// the same term one line earlier, so the encoder's `TermKind::Eq` arm had
/// already emitted the clause they duplicated — unguarded, and therefore again
/// on every MBQI round.
///
/// One of the deleted tests pinned that the walks return on a 128 KiB stack.
/// That property has no subject any more, and needs no replacement: the walks
/// were uncapped because their `()` return type had no channel to report a cap,
/// whereas `encode_depth` stops at `ENCODE_DEPTH_LIMIT` (512) and reports
/// through `encode_depth_exceeded`, which the tests below and
/// `check_core`'s gate already cover.
#[test]
fn numeric_eq_atoms_get_their_trichotomy_from_the_encoder() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let z = manager.mk_var("z", int_sort);

    // A bare comparison is not an `Eq` atom, so it gets no trichotomy. It does
    // still get its own definitional encoding, so pin the *ledger* rather than
    // the clause count.
    let lt = manager.mk_lt(x, y);
    solver.encode(lt, &mut manager);
    assert!(
        solver.numeric_trichotomy_atoms.is_empty(),
        "a bare comparison is not an equality and must not be split"
    );

    // `not (= x y)`: encoding descends to the `Eq` atom, which takes its
    // trichotomy. Assert clause *growth* rather than an exact count, to stay
    // robust to the strict atoms' own definitional clauses.
    let eq_xy = manager.mk_eq(x, y);
    let neq = manager.mk_not(eq_xy);
    let before = solver.sat.num_clauses();
    solver.encode(neq, &mut manager);
    let after_neq = solver.sat.num_clauses();
    assert!(
        after_neq > before,
        "not (= x y) must contribute a trichotomy clause"
    );
    assert!(
        solver.numeric_trichotomy_atoms.contains(&eq_xy),
        "the `Eq` atom under the `not` must be on the trichotomy ledger"
    );

    // Re-encoding the same atom must NOT emit a second, literal-identical
    // clause: `oxiz_sat::Solver::add_clause` does not deduplicate, and the
    // unguarded re-emission this ledger prevents is exactly what made the
    // deleted walks grow the clause database without bound.
    let after_reencode_start = solver.sat.num_clauses();
    solver.encode(neq, &mut manager);
    assert_eq!(
        solver.sat.num_clauses(),
        after_reencode_start,
        "re-encoding an already-split `Eq` atom must add no clauses"
    );

    // `distinct(x, y, z)` expands to three pairwise `Eq` atoms in the encoder,
    // so all three pairs land on the ledger through the same single owner.
    let distinct = manager.mk_distinct(vec![x, y, z]);
    solver.encode(distinct, &mut manager);
    assert!(
        solver.sat.num_clauses() > after_reencode_start,
        "distinct(x, y, z) must contribute its pairwise trichotomy clauses"
    );
    let eq_xz = manager.mk_eq(x, z);
    let eq_yz = manager.mk_eq(y, z);
    for (pair, name) in [(eq_xz, "(= x z)"), (eq_yz, "(= y z)")] {
        assert!(
            solver.numeric_trichotomy_atoms.contains(&pair),
            "{name} from the `distinct` must be on the trichotomy ledger"
        );
    }
}

// =====================================================================
// Depth-guard measurement coverage and the sticky-flag `Sat` gate.
// =====================================================================

/// The assert-time depth pre-check must measure nesting through *datatype
/// constructor* arguments: `succ(succ(...))` is one level per constructor.
/// `DtConstructor` (with `DtTester`/`DtSelector`, `Match`, `FpFma` and the
/// FP conversions) used to fall into `push_child_terms`' `_ => {}`
/// catch-all, so an over-deep constructor chain passed the guard unmeasured
/// and was handed to the native-recursive passes the guard exists to
/// protect.  The pin: a 12 500-deep chain (still 24x past the guard's 512)
/// must trip the guard at assert time, and `check` must answer an honest
/// `Unknown` — returning at all on a 128 KiB thread is the no-overflow half
/// of the proof.
#[test]
fn depth_guard_measures_datatype_constructor_chains() {
    // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
    // ~10 B-per-frame threshold is the pin, so never raise one alone.
    const STACK_SIZE: usize = 1 << 17; // 128 KiB
    const DEPTH: usize = 12_500;

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let mut solver = Solver::new();
            let mut manager = TermManager::new();
            let int_sort = manager.sorts.int_sort;
            // Nat-like chain: succ(succ(...zero...)).  `mk_dt_constructor`
            // interns without simplification, so the nesting is genuine; the
            // guard is purely structural, so the ascribed sort is irrelevant.
            let mut chain = manager.mk_dt_constructor("zero", [], int_sort);
            for _ in 0..DEPTH {
                chain = manager.mk_dt_constructor("succ", [chain], int_sort);
            }
            let x = manager.mk_var("x", int_sort);
            let eq = manager.mk_eq(x, chain);

            solver.assert(eq, &mut manager);
            assert!(
                solver.encode_depth_exceeded,
                "a {DEPTH}-deep constructor chain must be measured by the depth guard"
            );
            assert_eq!(
                solver.check(&mut manager),
                SolverResult::Unknown,
                "an assertion the encoder refused must answer Unknown, not guess"
            );
        })
        .expect("spawning a 128 KiB-stack thread should succeed");

    handle
        .join()
        .expect("a deep constructor chain must return on 128 KiB instead of overflowing it");
}

/// The `check()` wrapper must degrade `Sat` to `Unknown` whenever
/// `encode_depth_exceeded` is set — `check_core`'s top-of-function gate is
/// not enough on its own, because MBQI instantiation results and E-matching
/// lemmas are encoded *mid-loop*, after that gate has already been
/// consulted, and `encode_depth`'s cap can flip the flag between the gate
/// and a `Sat` exit.  Constructed here through the one deterministic path
/// that reaches a `Sat` verdict with the flag already set: an empty
/// assertion set (`check_core` answers `Sat` before its flag gate) after a
/// direct over-cap `encode` — exactly the call the MBQI loop makes for an
/// over-deep lemma.  A truncated encoding only ever drops clauses, so
/// `Unsat` would remain sound; `Sat` may rest on the missing constraints
/// and must not survive.
#[test]
fn sat_verdict_degrades_to_unknown_after_mid_search_truncation() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let over_cap = super::super::ENCODE_DEPTH_LIMIT as usize + 8;
    let deep = build_implies_chain(&mut manager, over_cap);

    // Encode without asserting, as the MBQI loop does for lemmas: the cap
    // fires and sets the flag, but no assertion exists for the assert-time
    // pre-check or the `check_core` top gate to catch.
    let _ = solver.encode(deep, &mut manager);
    assert!(
        solver.encode_depth_exceeded,
        "an over-cap encode must set the truncation flag"
    );

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Unknown,
        "a Sat verdict reached while the SAT core holds a truncated encoding \
         must be degraded to Unknown by the check() wrapper gate"
    );
    assert!(
        solver.model().is_none(),
        "no model may be surfaced alongside the degraded verdict"
    );
}

/// `check_sat_only` skips the theory layer but must still respect the two
/// facts `assert` records *outside* the SAT clause database: an asserted
/// `False` (no clause is emitted for it — only `has_false_assertion` is set)
/// and an encoder-refused over-deep assertion (no clauses at all — only
/// `encode_depth_exceeded` is set).  Solving the remaining clauses alone
/// answered `Sat` for both, a silently wrong verdict rather than a
/// documented limitation of the pure-SAT entry point.
#[test]
fn check_sat_only_respects_false_and_truncation_flags() {
    // Asserted `False` => Unsat, even though no clause carries it.
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let f = manager.mk_false();
    solver.assert(f, &mut manager);
    assert_eq!(
        solver.check_sat_only(&mut manager),
        SolverResult::Unsat,
        "check_sat_only must report an asserted false as Unsat"
    );

    // Encoder-refused deep assertion => Unknown, never a guessed Sat.  The
    // assert-time pre-check is an explicit-stack scan and the guard returns
    // before any recursive pass, so no small-stack thread is needed here.
    // Depth only needs to exceed ENCODE_DEPTH_LIMIT (512); lowered from
    // 100_000 to cut the crate's test memory floor.
    let mut solver = Solver::new();
    let deep = build_implies_chain(&mut manager, 4_096);
    solver.assert(deep, &mut manager);
    assert!(solver.encode_depth_exceeded);
    assert_eq!(
        solver.check_sat_only(&mut manager),
        SolverResult::Unknown,
        "check_sat_only must not guess over a truncated encoding"
    );
}

// =====================================================================
// The boolean-constant arms of the Tseitin encoder: `encode` must return a
// literal whose truth value *equals* the encoded term's.
// =====================================================================

/// Root-cause pin for the `TermKind::False` arm.  The arm pins its variable
/// with the unit clause `[neg(var)]`, so the literal that *stands for* the
/// constant is `pos(var)`; returning `neg(var)` (which evaluates to `true`)
/// inverted every nested occurrence of `false`.
///
/// `assert` short-circuits a top-level `False`/`True` (see the constant checks
/// in `Solver::assert`), so the arm is only reachable for a constant that
/// appears *inside* a larger term.  `mk_eq` folds constant operand pairs, which
/// makes `Distinct` over constants the natural route: `distinct(5, 2)` expands
/// to the pairwise `mk_eq(5, 2)`, which folds to the `False` term, and the
/// mis-polarised literal turned a satisfiable assertion into `Unsat`.
#[test]
fn distinct_over_constant_int_operands_is_sat() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();

    let five = manager.mk_int(5);
    let two = manager.mk_int(2);
    let distinct = manager.mk_distinct([five, two]);
    solver.assert(distinct, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Sat,
        "5 and 2 are distinct integers; `Unsat` means the folded `False` pair \
         encoded to a literal that evaluates to `true`"
    );
}

/// Same pin with more than one pair, so the `n`-ary expansion is covered too.
#[test]
fn distinct_over_three_constant_int_operands_is_sat() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();

    let one = manager.mk_int(1);
    let two = manager.mk_int(2);
    let three = manager.mk_int(3);
    let distinct = manager.mk_distinct([one, two, three]);
    solver.assert(distinct, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Sat,
        "1, 2 and 3 are pairwise distinct; all three folded pairs must encode \
         to literals that evaluate to `false`"
    );
}

/// Companion refutation pin: the `True` arm was already correct and must stay
/// correct.  `mk_eq` folds a syntactically identical pair to `True`, so
/// `distinct(5, 5)` is unsatisfiable.
#[test]
fn distinct_over_equal_int_constants_is_unsat() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();

    let five = manager.mk_int(5);
    let distinct = manager.mk_distinct([five, five]);
    solver.assert(distinct, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Unsat,
        "5 is not distinct from itself"
    );
}

/// Negating the assertion must flip the verdict: a bug that reports `Unsat`
/// for `distinct(5, 2)` reports `Sat` here, so this pins the other polarity.
#[test]
fn negated_distinct_over_constant_int_operands_is_unsat() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();

    let five = manager.mk_int(5);
    let two = manager.mk_int(2);
    let distinct = manager.mk_distinct([five, two]);
    let negated = manager.mk_not(distinct);
    solver.assert(negated, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Unsat,
        "`not (distinct 5 2)` claims 5 = 2 and must be refuted"
    );
}

/// The boolean-constant operands take the `(True, False)` fold branch of
/// `mk_eq` rather than the `IntConst` one, and reach the same arm.
#[test]
fn distinct_over_constant_bool_operands_is_sat() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();

    let t = manager.mk_true();
    let f = manager.mk_false();
    let distinct = manager.mk_distinct([t, f]);
    solver.assert(distinct, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Sat,
        "`true` and `false` are distinct booleans"
    );
}

/// The `BitVecConst` fold branch of `mk_eq`, reaching the same arm.
#[test]
fn distinct_over_constant_bitvec_operands_is_sat() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();

    let one = manager.mk_bitvec(1i64, 2);
    let two = manager.mk_bitvec(2i64, 2);
    let distinct = manager.mk_distinct([one, two]);
    solver.assert(distinct, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Sat,
        "#b01 and #b10 are distinct bitvectors"
    );
}

/// The exact shape the `QF_BV` differential-test seed produced: the bitvector
/// operands are *expressions*, but `mk_bv_neg`/`mk_bv_and` fold their constant
/// arguments, so `mk_eq` still sees a `BitVecConst` pair and still folds.
/// `bvneg #b1011` is `#b0101`, `bvand #b0100 #b0101` is `#b0100`, which is
/// distinct from `#b0010`.
#[test]
fn distinct_over_folded_bitvec_expression_operands_is_sat() {
    let mut solver = Solver::new();
    let mut manager = TermManager::new();

    let four = manager.mk_bitvec(4i64, 4);
    let eleven = manager.mk_bitvec(11i64, 4);
    let negated = manager.mk_bv_neg(eleven);
    let masked = manager.mk_bv_and(four, negated);
    let two = manager.mk_bitvec(2i64, 4);
    let distinct = manager.mk_distinct([masked, two]);
    solver.assert(distinct, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Sat,
        "(bvand #b0100 (bvneg #b1011)) is #b0100, which is distinct from #b0010"
    );
}

/// Guard the *non*-constant path: with a variable operand no pair folds, so
/// this exercises the ordinary theory encoding and must keep answering by the
/// variable's value, not by the constant arms.
#[test]
fn distinct_over_mixed_constant_and_variable_operands_stays_correct() {
    // x = 2 leaves `distinct(5, x)` satisfiable.
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let five = manager.mk_int(5);
    let two = manager.mk_int(2);
    let bind = manager.mk_eq(x, two);
    let distinct = manager.mk_distinct([five, x]);
    solver.assert(bind, &mut manager);
    solver.assert(distinct, &mut manager);
    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Sat,
        "x = 2 is distinct from 5"
    );

    // x = 5 refutes it.
    let mut solver = Solver::new();
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let five = manager.mk_int(5);
    let bind = manager.mk_eq(x, five);
    let distinct = manager.mk_distinct([five, x]);
    solver.assert(bind, &mut manager);
    solver.assert(distinct, &mut manager);
    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Unsat,
        "x = 5 is not distinct from 5"
    );
}
