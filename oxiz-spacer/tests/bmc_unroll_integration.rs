//! Integration test: `BmcUnrollTactic` and the production `Bmc` solver
//! exercising the same simple transition system.
//!
//! The test verifies that:
//! 1. A minimal 3-assertion goal (init / trans / property) produces a `SubGoals`
//!    result from `BmcUnrollTactic`.
//! 2. A matching `ChcSystem` encoding of the same system passes through the
//!    production `Bmc::check()` without error, demonstrating that the tactic and
//!    the full solver agree on the problem domain.

use oxiz_core::TermManager;
use oxiz_core::tactic::Goal;
use oxiz_spacer::chc::{ChcSystem, PredicateApp};
use oxiz_spacer::bmc::{Bmc, BmcConfig, BmcResult};
use oxiz_spacer::{BmcUnrollTactic};

/// Build a minimal "x starts at 0 and increments" transition system as both
/// a flat `Goal` (for the tactic) and a `ChcSystem` (for the Bmc solver).
///
/// Transition system:
///   init:     x = 0
///   trans:    x_next = x + 1
///   property: x >= 0
#[test]
fn test_bmc_unroll_feeds_bmc_engine() {
    let mut terms = TermManager::new();

    // --- Build the flat Goal for BmcUnrollTactic ---
    let int_sort = terms.sorts.int_sort;
    let x = terms.mk_var("x", int_sort);
    let x_next = terms.mk_var("x_next", int_sort);
    let zero = terms.mk_int(0);
    let one = terms.mk_int(1);

    let init_term = terms.mk_eq(x, zero);
    let x_plus_one_tactic = terms.mk_add([x, one]);
    let trans_term = terms.mk_eq(x_next, x_plus_one_tactic);
    let property_term = terms.mk_ge(x, zero);

    let goal = Goal::new(vec![init_term, trans_term, property_term]);

    // Apply the tactic
    let mut tactic = BmcUnrollTactic::with_depth(&mut terms, 3);
    let result = tactic
        .apply_mut(&goal)
        .expect("BmcUnrollTactic::apply_mut should not fail on a 3-assertion goal");

    // The tactic must return SubGoals (not NotApplicable) for ≥3 assertions.
    match result {
        oxiz_core::tactic::TacticResult::SubGoals(ref subgoals) => {
            assert_eq!(subgoals.len(), 1, "expected exactly one subgoal");
            assert!(
                !subgoals[0].assertions.is_empty(),
                "unrolled subgoal must have at least one assertion"
            );
        }
        oxiz_core::tactic::TacticResult::NotApplicable => {
            panic!(
                "BmcUnrollTactic returned NotApplicable for a goal with ≥3 assertions; \
                 this path must not arise for well-formed transition systems"
            );
        }
        other => panic!("unexpected TacticResult variant: {other:?}"),
    }

    // --- Build the ChcSystem for the production Bmc solver ---
    // The Bmc solver requires a ChcSystem with at least one init rule and one query.
    let mut system = ChcSystem::new();

    // Declare invariant predicate Inv(x: Int)
    let inv = system.declare_predicate("Inv", [int_sort]);

    // Fresh variables (the ChcSystem encodes them via rule bodies)
    let xv = terms.mk_var("xv", int_sort);
    let xv_next = terms.mk_var("xv_next", int_sort);
    let zero2 = terms.mk_int(0);
    let one2 = terms.mk_int(1);
    let neg_one = terms.mk_int(-1);

    // Init rule: xv = 0 => Inv(xv)
    let init_constraint = terms.mk_eq(xv, zero2);
    system.add_init_rule(
        [("xv".to_string(), int_sort)],
        init_constraint,
        inv,
        [xv],
    );

    // Transition rule: Inv(xv) /\ xv_next = xv + 1 => Inv(xv_next)
    let xv_plus_one = terms.mk_add([xv, one2]);
    let trans_constraint = terms.mk_eq(xv_next, xv_plus_one);
    system.add_transition_rule(
        [("xv".to_string(), int_sort), ("xv_next".to_string(), int_sort)],
        [PredicateApp::new(inv, [xv])],
        trans_constraint,
        inv,
        [xv_next],
    );

    // Query: Inv(xv) /\ xv < 0 => false  (checking safety: xv >= 0 always)
    let neg_constraint = terms.mk_lt(xv, neg_one);
    system.add_query(
        [("xv".to_string(), int_sort)],
        [PredicateApp::new(inv, [xv])],
        neg_constraint,
    );

    // Run the production BMC solver — just assert no error (Ok(_)).
    let config = BmcConfig {
        max_depth: 3,
        use_kinduction: false,
        verbosity: 0,
    };
    let bmc_result = Bmc::with_config(&mut terms, &system, config).check();
    assert!(
        bmc_result.is_ok(),
        "Bmc::check() must not return an error on a valid CHC system; got: {bmc_result:?}"
    );

    // The system is safe (xv >= 0 invariant holds), so we expect Safe(3).
    assert!(
        matches!(bmc_result, Ok(BmcResult::Safe(_))),
        "expected Bmc to report Safe for x>=0 counter system; got: {bmc_result:?}"
    );
}
