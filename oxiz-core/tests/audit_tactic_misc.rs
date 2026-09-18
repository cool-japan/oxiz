//! Regression tests for audited soundness defects in the `tactic-misc`
//! package (`oxiz-core/src/tactic/solve_eqs.rs`'s Fourier-Motzkin tactic and
//! `oxiz-core/src/tactic/lia2card.rs`'s cardinality encodings).
//!
//! Findings fixed:
//!
//! 1. `FourierMotzkinTactic::apply_mut` used to mark *all* lower/upper
//!    constraints for a variable dead as soon as the pairwise-resolution
//!    inner loop hit `op_limit`, even though some pairs (potentially
//!    including a contradiction) were never resolved. This discarded
//!    constraint information and could turn an UNSAT input into a reported
//!    SAT/weaker subgoal -- a resource limit silently breaking soundness.
//!    Fixed by leaving that variable's original constraints alive (and
//!    stopping elimination entirely) when the op limit fires mid-resolution.
//!
//! 2. `Lia2CardTactic`'s sequential-counter (`__card_s_{i}_{j}`) and
//!    commander (`__card_cmd_{g}`) auxiliary variable names did not include
//!    the per-tactic `aux_var_counter`, so two independent encoding passes
//!    (e.g. the AtMost and negated-AtLeast passes used to encode a single
//!    `Exactly(k)` constraint) reused identical names for semantically
//!    unrelated auxiliary booleans. Since `TermManager::mk_var` hash-conses
//!    by name, this aliased distinct auxiliary variables onto the same
//!    `TermId`, injecting spurious equivalences that can flip SAT to UNSAT.
//!    Fixed by folding `aux_var_counter` into every generated name (as was
//!    already done for the totalizer's `__tot_{}_{}` variables).

use oxiz_core::ast::traversal::collect_subterms;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::tactic::{
    FourierMotzkinTactic, Goal, Lia2CardConfig, Lia2CardTactic, SolveResult, TacticResult,
};
use std::collections::HashSet;

/// Collect the set of distinct `TermId`s that are `Var` terms whose
/// (interned) name starts with `prefix`, across every assertion in `terms`.
fn distinct_vars_with_prefix(
    terms: &[TermId],
    manager: &TermManager,
    prefix: &str,
) -> HashSet<TermId> {
    let mut found = HashSet::new();
    for &t in terms {
        for sub in collect_subterms(t, manager) {
            if let Some(term) = manager.get(sub)
                && let TermKind::Var(name) = &term.kind
                && manager.resolve_str(*name).starts_with(prefix)
            {
                found.insert(sub);
            }
        }
    }
    found
}

// ============================================================================
// Finding 1: Fourier-Motzkin op_limit abort must not discard constraints.
// ============================================================================

/// Reproduces the audited bug: with an op_limit small enough to abort before
/// any pair is resolved, a genuinely UNSAT system of bound constraints on a
/// single variable used to have ALL of its constraints marked dead (since the
/// dead-marking loop ran unconditionally after the inner break), losing the
/// contradiction and yielding `Solved(Sat)` instead of preserving the
/// constraints for a later (sound) check.
#[test]
fn test_fm_op_limit_abort_preserves_constraints_unsat_system() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    // x >= 10, x >= 20 (lower bounds) and x <= 5, x <= 100 (upper bounds).
    // This is UNSAT: x >= 20 and x <= 5 cannot both hold.
    let x = manager.mk_var("x", int_sort);
    let ten = manager.mk_int(10);
    let twenty = manager.mk_int(20);
    let five = manager.mk_int(5);
    let hundred = manager.mk_int(100);

    let lower1 = manager.mk_ge(x, ten);
    let lower2 = manager.mk_ge(x, twenty);
    let upper1 = manager.mk_le(x, five);
    let upper2 = manager.mk_le(x, hundred);

    let goal = Goal::new(vec![lower1, lower2, upper1, upper2]);

    // op_limit = 1: the very first pairwise resolution increments op_count
    // to 1 and immediately aborts, before any pair (including the
    // contradictory 20 <= x <= 5 pair) is ever resolved.
    let mut tactic = FourierMotzkinTactic::new(&mut manager).with_op_limit(1);

    let result = tactic.apply_mut(&goal).expect("fm tactic should not error");

    // The bug: this used to return Solved(Sat), incorrectly claiming the
    // (unsat) system is satisfiable, because all four constraints were
    // marked dead without ever being resolved or checked for contradiction.
    assert!(
        !matches!(result, TacticResult::Solved(SolveResult::Sat)),
        "op_limit abort must not fabricate a Sat result for an unresolved, \
         potentially-unsat system: got {result:?}"
    );

    // Honest behavior: elimination for `x` is skipped entirely (since the
    // limit fired before any pair completed), so the tactic must either
    // leave all four original constraints fully intact (either by reporting
    // NotApplicable -- i.e. it made no change, so the caller's original goal
    // still holds all four assertions -- or by returning a SubGoal that
    // still carries all four) for a later, sound check to discover the
    // contradiction. It must never discard them.
    match result {
        TacticResult::NotApplicable => {
            // No change was made: the goal the caller already has (with all
            // four original assertions) remains valid and complete.
        }
        TacticResult::SubGoals(goals) => {
            assert_eq!(goals.len(), 1);
            assert_eq!(
                goals[0].assertions.len(),
                4,
                "all four original bound constraints on x must be preserved \
                 when the op limit aborts elimination, got: {:?}",
                goals[0].assertions
            );
        }
        TacticResult::Solved(SolveResult::Unsat) => {
            // Also acceptable: if the tactic manages to detect the
            // contradiction directly, that's a strictly better (still
            // sound) outcome.
        }
        other => panic!(
            "expected NotApplicable, SubGoals preserving constraints, or Unsat, got {other:?}"
        ),
    }
}

/// Sanity check that, away from any op_limit pressure, the tactic still
/// correctly detects the same UNSAT system as before (no regression to the
/// normal-path behavior).
#[test]
fn test_fm_detects_unsat_without_op_limit_pressure() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let ten = manager.mk_int(10);
    let five = manager.mk_int(5);

    let lower = manager.mk_ge(x, ten); // x >= 10
    let upper = manager.mk_le(x, five); // x <= 5

    let goal = Goal::new(vec![lower, upper]);
    let mut tactic = FourierMotzkinTactic::new(&mut manager); // default op_limit

    let result = tactic.apply_mut(&goal).expect("fm tactic should not error");
    assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
}

// ============================================================================
// Finding 2: lia2card auxiliary variable names must be globally unique.
// ============================================================================

/// Reproduces the audited bug: encoding a single `Exactly(k, vars)`
/// constraint via the sequential-counter internally performs two separate
/// `encode_at_most_sequential` calls (once for the AtMost pass, once for the
/// negated AtLeast pass), both iterating the same local `(i, j)` index
/// ranges. Without folding a per-tactic counter into the generated
/// `__card_s_{i}_{j}` names, both passes produced identical names, and
/// `TermManager::mk_var` hash-conses by name -- so the two passes'
/// "at least j of first i" counter variables aliased onto the same TermId
/// even though they range over entirely different (positive vs. negated)
/// variables. That forces spurious equivalences (e.g. s[0][1] <=> x0 AND
/// s[0][1] <=> !x0) that can flip a genuinely SAT input to UNSAT once the
/// clauses reach a SAT solver.
#[test]
fn test_lia2card_sequential_counter_aux_vars_unique_across_passes() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    // Four 0-1 bounded variables a, b, c, d.
    let vars: Vec<TermId> = ["a", "b", "c", "d"]
        .iter()
        .map(|n| manager.mk_var(n, int_sort))
        .collect();
    let zero = manager.mk_int(0);
    let one = manager.mk_int(1);

    let mut assertions = Vec::new();
    for &v in &vars {
        assertions.push(manager.mk_ge(v, zero));
        assertions.push(manager.mk_le(v, one));
    }

    // Exactly(2, [a,b,c,d]): AtMost(2, vars) forces an
    // encode_at_most_sequential(2, [a,b,c,d]) pass, and AtLeast(2, vars) is
    // encoded as NOT AtMost(4-2=2, negated vars), forcing a second
    // encode_at_most_sequential(2, [!a,!b,!c,!d]) pass with the identical
    // local (i, j) index range (i in 0..4, j in 0..=2).
    let sum = manager.mk_add(vars.clone());
    let two = manager.mk_int(2);
    let exactly_two = manager.mk_eq(sum, two);
    assertions.push(exactly_two);

    let goal = Goal::new(assertions);

    // Force sequential-counter encoding (not commander/totalizer) so both
    // passes definitely go through encode_at_most_sequential.
    let config = Lia2CardConfig {
        encoding: oxiz_core::tactic::CardinalityEncoding::SequentialCounter,
        sequential_counter_threshold: 100,
        use_commander_for_amo: false,
    };
    let mut tactic = Lia2CardTactic::with_config(&mut manager, config);

    let result = tactic
        .apply_mut(&goal)
        .expect("lia2card tactic should not error");

    let TacticResult::SubGoals(goals) = result else {
        panic!("expected SubGoals from lia2card encoding, got {result:?}");
    };
    assert_eq!(goals.len(), 1);

    // n=4, k=2 => each encode_at_most_sequential pass creates n * (k+1) = 12
    // fresh `__card_s_*` counter variables. Two independent passes (AtMost
    // pass over [a,b,c,d], AtLeast pass over [!a,!b,!c,!d]) must therefore
    // produce 24 *distinct* TermIds. Before the fix, the passes' identical
    // (i, j) names aliased pairwise, collapsing this down to 12.
    let card_s_vars = distinct_vars_with_prefix(
        &goals[0].assertions,
        &manager,
        &oxiz_core::smtlib::reserved_name("cards", ""),
    );
    assert_eq!(
        card_s_vars.len(),
        24,
        "sequential-counter aux vars from the AtMost and negated-AtLeast \
         passes of a single Exactly(k) constraint must not alias onto the \
         same TermId; found {} distinct vars (expected 24 -- 12 aliased \
         collisions would yield 12)",
        card_s_vars.len()
    );
}

/// Reproduces the same aliasing bug for the commander encoding's
/// `__card_cmd_{g}` group-commander variables, across two independent
/// `AtMost`-with-commander encoding calls in the same goal.
#[test]
fn test_lia2card_commander_aux_vars_unique_across_constraints() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    // Two independent groups of 6 variables (n=6 > 4 forces the group/
    // commander path rather than plain pairwise encoding), each with its
    // own AtMost(1, ...) constraint (k=1 with use_commander_for_amo=true
    // routes through encode_amo_commander).
    let group1: Vec<TermId> = (0..6)
        .map(|i| manager.mk_var(&format!("p{i}"), int_sort))
        .collect();
    let group2: Vec<TermId> = (0..6)
        .map(|i| manager.mk_var(&format!("q{i}"), int_sort))
        .collect();

    let zero = manager.mk_int(0);
    let one = manager.mk_int(1);
    let mut assertions = Vec::new();
    for &v in group1.iter().chain(group2.iter()) {
        assertions.push(manager.mk_ge(v, zero));
        assertions.push(manager.mk_le(v, one));
    }

    let sum1 = manager.mk_add(group1.clone());
    let at_most1 = manager.mk_le(sum1, one);
    let sum2 = manager.mk_add(group2.clone());
    let at_most2 = manager.mk_le(sum2, one);
    assertions.push(at_most1);
    assertions.push(at_most2);

    let goal = Goal::new(assertions);

    let config = Lia2CardConfig {
        encoding: oxiz_core::tactic::CardinalityEncoding::SequentialCounter,
        sequential_counter_threshold: 100,
        use_commander_for_amo: true,
    };
    let mut tactic = Lia2CardTactic::with_config(&mut manager, config);

    let result = tactic
        .apply_mut(&goal)
        .expect("lia2card tactic should not error");
    let TacticResult::SubGoals(goals) = result else {
        panic!("expected SubGoals from lia2card encoding, got {result:?}");
    };
    assert_eq!(goals.len(), 1);

    // n=6, group_size = ceil(sqrt(6)) = 3 => 2 groups => 2 commander vars
    // per AtMost(1, ...) call. Two independent AtMost constraints must
    // therefore yield 4 distinct commander TermIds; before the fix both
    // calls produced `__card_cmd_0` / `__card_cmd_1`, aliasing the two
    // constraints' commanders pairwise (collapsing to 2).
    let cmd_vars = distinct_vars_with_prefix(
        &goals[0].assertions,
        &manager,
        &oxiz_core::smtlib::reserved_name("cardcmd", ""),
    );
    assert_eq!(
        cmd_vars.len(),
        4,
        "commander aux vars from two independent AtMost(1, ...) constraints \
         must not alias onto the same TermId; found {} distinct vars \
         (expected 4 -- 2 aliased collisions would yield 2)",
        cmd_vars.len()
    );
}
