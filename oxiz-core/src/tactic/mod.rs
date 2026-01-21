//! Tactic framework for OxiZ
//!
//! Tactics are transformers that take a Goal and produce sub-Goals.
//! They form the basis of the strategy layer in the solver.

// Existing submodules
pub mod lia2card;
pub mod mbp;
pub mod nla2bv;
pub mod probe;
pub mod quantifier;

// Refactored modules
mod core;
mod simplify;
mod propagate;
mod bitblast;
mod ackermann;
mod ctx_simplify;
mod combinators;
mod split;
mod eliminate;
mod solve_eqs;

// Re-export core types
pub use core::{Goal, Precision, SolveResult, Tactic, TacticResult};

// Re-export tactics
pub use simplify::{SimplifyTactic, StatelessSimplifyTactic};
pub use propagate::{PropagateValuesTactic, StatelessPropagateValuesTactic};
pub use bitblast::{BitBlastTactic, StatelessBitBlastTactic};
pub use ackermann::{AckermannizeTactic, StatelessAckermannizeTactic};
pub use ctx_simplify::{CtxSolverSimplifyTactic, StatelessCtxSolverSimplifyTactic};
pub use combinators::{OrElseTactic, ParallelTactic, RepeatTactic, ThenTactic, TimeoutTactic};
pub use split::{SplitTactic, StatelessSplitTactic};
pub use eliminate::{EliminateUnconstrainedTactic, StatelessEliminateUnconstrainedTactic};
pub use solve_eqs::{
    CondTactic, FailIfTactic, FourierMotzkinTactic, NnfTactic, Pb2BvTactic, ScriptableTactic,
    SolveEqsTactic, StatelessCnfTactic, StatelessFourierMotzkinTactic, StatelessNnfTactic,
    StatelessPb2BvTactic, StatelessSolveEqsTactic, TseitinCnfTactic, WhenTactic,
};

// Re-export probe types
pub use probe::{
    AddProbe, AndProbe, ConstProbe, DepthProbe, HasArrayProbe, HasBitVectorProbe,
    HasFloatingPointProbe, HasQuantifierProbe, IsLinearProbe, LtProbe, NegProbe, NodeCountProbe,
    NotProbe, NumConnectivesProbe, NumVarsProbe, OrProbe, Probe, SizeProbe,
};

// Re-export quantifier tactics
pub use quantifier::{
    Binding, GroundTermCollector, Pattern, PatternMatcher, QuantifierInstantiationTactic,
    SkolemizationTactic, UniversalEliminationTactic, contains_quantifier, goal_has_quantifiers,
};

// Re-export DER (Destructive Equality Resolution) types
pub use quantifier::{DerConfig, DerTactic, StatelessDerTactic};

// Re-export MBP types
pub use mbp::{MbpConfig, MbpEngine, MbpResult, MbpTactic, Model as MbpModel, ProjectorKind};

// Re-export NLA2BV types
pub use nla2bv::{Nla2BvConfig, Nla2BvTactic, StatelessNla2BvTactic};

// Re-export LIA2Card types
pub use lia2card::{
    CardinalityConstraint, CardinalityEncoding, Lia2CardConfig, Lia2CardTactic,
    StatelessLia2CardTactic,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{TermId, TermKind, TermManager};
    use crate::error::Result;

    #[test]
    fn test_empty_goal() {
        let goal = Goal::empty();
        assert!(goal.is_empty());
        assert_eq!(goal.len(), 0);
    }

    #[test]
    fn test_simplify_tactic() {
        let goal = Goal::empty();
        let tactic = StatelessSimplifyTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_simplify_tactic_with_manager() {
        let mut manager = TermManager::new();

        // Create (+ 1 2) which should simplify to 3
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let sum = manager.mk_add([one, two]);

        // Create (= (+ 1 2) 3) which should simplify to true
        let three = manager.mk_int(3);
        let eq = manager.mk_eq(sum, three);

        let goal = Goal::new(vec![eq]);
        let mut tactic = SimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        // The goal should be solved as SAT (all assertions simplified to true)
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_simplify_tactic_unsat() {
        let mut manager = TermManager::new();

        // Create (< 5 3) which should simplify to false
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);
        let lt = manager.mk_lt(five, three);

        let goal = Goal::new(vec![lt]);
        let mut tactic = SimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        // The goal should be solved as UNSAT (an assertion simplified to false)
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_simplify_constant_folding() {
        let mut manager = TermManager::new();

        // Create (+ 1 2 3) which should simplify to 6
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let three = manager.mk_int(3);
        let sum = manager.mk_add([one, two, three]);

        let simplified = manager.simplify(sum);
        // Check it's an integer constant with value 6
        if let Some(term) = manager.get(simplified) {
            if let crate::ast::TermKind::IntConst(n) = &term.kind {
                assert_eq!(*n, num_bigint::BigInt::from(6));
            } else {
                panic!("Expected IntConst");
            }
        } else {
            panic!("Term not found");
        }
    }

    #[test]
    fn test_simplify_mul_by_zero() {
        let mut manager = TermManager::new();

        // Create (* x 0) which should simplify to 0
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let product = manager.mk_mul([x, zero]);

        let simplified = manager.simplify(product);
        assert_eq!(simplified, zero);
    }

    #[test]
    fn test_propagate_values_basic() {
        let mut manager = TermManager::new();

        // Create goal: (= x 5), (< x 10)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After propagation: (= 5 5) -> true, (< 5 10) -> true
        // Both simplify to true, so goal should be SAT
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_propagate_values_unsat() {
        let mut manager = TermManager::new();

        // Create goal: (= x 5), (< x 3)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, three);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After propagation: (= 5 5) -> true, (< 5 3) -> false
        // Contains false, so goal should be UNSAT
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_propagate_values_partial() {
        let mut manager = TermManager::new();

        // Create goal: (= x 5), (< y 10)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(y, ten);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After propagation: (= 5 5) -> true (removed), (< y 10) stays
        // Should produce a subgoal with just (< y 10)
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            assert_eq!(goals[0].assertions.len(), 1);
            // The remaining assertion should be (< y 10)
            assert_eq!(goals[0].assertions[0], lt);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_propagate_values_not_applicable() {
        let mut manager = TermManager::new();

        // Create goal: (< x 10) - no equalities to propagate
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let lt = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![lt]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // No equalities of form (= var const), so not applicable
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_propagate_values_chained() {
        let mut manager = TermManager::new();

        // Create goal: (= x 5), (= y x), (< y 10)
        // Note: This tests that x=5 propagates, and whether we handle (= y x)
        // Currently, we only handle (= var const), so y won't be substituted in first pass
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq_x_5 = manager.mk_eq(x, five);
        let eq_y_x = manager.mk_eq(y, x);
        let lt_y_10 = manager.mk_lt(y, ten);

        let goal = Goal::new(vec![eq_x_5, eq_y_x, lt_y_10]);
        let mut tactic = PropagateValuesTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // x=5 is propagated, (= y x) becomes (= y 5)
        // We should get a reduced goal
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Should have 2 assertions: (= y 5) and (< y 10)
            assert_eq!(goals[0].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result, got {:?}", result);
        }
    }

    #[test]
    fn test_bit_blast_not_applicable() {
        let manager = TermManager::new();

        // Create a goal with no BV terms
        let goal = Goal::empty();
        let tactic = BitBlastTactic::new(&manager);

        let result = tactic.apply_check(&goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_bit_blast_with_bv_constant() {
        let mut manager = TermManager::new();

        // Create a BV constant term
        let bv = manager.mk_bitvec(42u64, 8);
        let goal = Goal::new(vec![bv]);

        let tactic = BitBlastTactic::new(&manager);
        let result = tactic.apply_check(&goal).unwrap();

        // Should detect BV term and return SubGoals (not NotApplicable)
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_bit_blast_with_bv_operation() {
        let mut manager = TermManager::new();

        // Create BV variables and an operation
        let bv8_sort = manager.sorts.bitvec(8);
        let a = manager.mk_var("a", bv8_sort);
        let b = manager.mk_var("b", bv8_sort);
        let sum = manager.mk_bv_add(a, b);
        let hundred = manager.mk_bitvec(100u64, 8);
        let eq = manager.mk_eq(sum, hundred);

        let goal = Goal::new(vec![eq]);
        let tactic = BitBlastTactic::new(&manager);
        let result = tactic.apply_check(&goal).unwrap();

        // Should detect BV terms
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_bit_blast_mixed_goal() {
        let mut manager = TermManager::new();

        // Create a mixed goal with both Int and BV terms
        let int_sort = manager.sorts.int_sort;
        let bv8_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", int_sort);
        let a = manager.mk_var("a", bv8_sort);
        let ten = manager.mk_int(10);
        let int_constraint = manager.mk_lt(x, ten);
        let hundred = manager.mk_bitvec(100u64, 8);
        let bv_constraint = manager.mk_bv_ult(a, hundred);

        let goal = Goal::new(vec![int_constraint, bv_constraint]);
        let tactic = BitBlastTactic::new(&manager);
        let result = tactic.apply_check(&goal).unwrap();

        // Should detect BV terms in the goal
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_stateless_bit_blast() {
        let goal = Goal::empty();
        let tactic = StatelessBitBlastTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "bit-blast");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_ackermannize_not_applicable() {
        let mut manager = TermManager::new();

        // Create a goal with no function applications
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let constraint = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = AckermannizeTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_ackermannize_single_function() {
        let mut manager = TermManager::new();

        // Create f(x) = 5
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let five = manager.mk_int(5);
        let constraint = manager.mk_eq(fx, five);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = AckermannizeTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce a subgoal with the function replaced by a variable
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Should have 1 assertion (no consistency constraints needed for single occurrence)
            assert_eq!(goals[0].assertions.len(), 1);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_ackermannize_two_applications() {
        let mut manager = TermManager::new();

        // Create f(x) = f(y)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let fy = manager.mk_apply("f", [y], int_sort);
        let constraint = manager.mk_eq(fx, fy);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = AckermannizeTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce a subgoal with:
        // 1. The original constraint with f(x) and f(y) replaced by fresh vars
        // 2. A consistency constraint: (x = y) => (ack_0 = ack_1)
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Should have 2 assertions: the substituted constraint + consistency constraint
            assert_eq!(goals[0].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_ackermannize_nested_functions() {
        let mut manager = TermManager::new();

        // Create f(f(x)) = y
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let ffx = manager.mk_apply("f", [fx], int_sort);
        let constraint = manager.mk_eq(ffx, y);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = AckermannizeTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce subgoals with both function applications replaced
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Should have 2 assertions: substituted constraint + consistency for f(x) and f(f(x))
            assert_eq!(goals[0].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_stateless_ackermannize() {
        let goal = Goal::empty();
        let tactic = StatelessAckermannizeTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "ackermannize");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_ctx_solver_simplify_not_applicable() {
        let mut manager = TermManager::new();

        // Empty goal - not applicable
        let goal = Goal::empty();
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_ctx_solver_simplify_single_assertion() {
        let mut manager = TermManager::new();

        // Single assertion with no context to use
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let constraint = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![constraint]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        // No context to use, so not applicable
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_ctx_solver_simplify_propagate() {
        let mut manager = TermManager::new();

        // x = 5, x < 10 should simplify to x = 5, 5 < 10 -> x = 5, true
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // The (< x 10) should simplify to (< 5 10) -> true and be filtered out
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // Only the equality should remain (the < simplified to true)
            assert_eq!(goals[0].assertions.len(), 1);
        } else {
            panic!("Expected SubGoals result, got {:?}", result);
        }
    }

    #[test]
    fn test_ctx_solver_simplify_unsat() {
        let mut manager = TermManager::new();

        // x = 5, x < 3 should simplify to x = 5, false -> UNSAT
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, three);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_ctx_solver_simplify_multiple_equalities() {
        let mut manager = TermManager::new();

        // x = 5, y = x, y < 10
        // Should propagate x -> 5, then y -> 5, resulting in all true
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq_x = manager.mk_eq(x, five);
        let eq_y = manager.mk_eq(y, x);
        let lt = manager.mk_lt(y, ten);

        let goal = Goal::new(vec![eq_x, eq_y, lt]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should reduce the goal significantly
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // All assertions simplified to true - OK
            }
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Some assertions may remain
                assert!(goals[0].assertions.len() <= 3);
            }
            _ => panic!("Expected SubGoals or Sat result"),
        }
    }

    #[test]
    fn test_stateless_ctx_solver_simplify() {
        let goal = Goal::empty();
        let tactic = StatelessCtxSolverSimplifyTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "ctx-solver-simplify");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_split_tactic_not_applicable() {
        let mut manager = TermManager::new();

        // Goal with only constants - nothing to split on
        let goal = Goal::new(vec![manager.mk_true()]);
        let mut tactic = SplitTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_split_tactic_simple() {
        let mut manager = TermManager::new();

        // Goal: x (where x is a boolean variable)
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let goal = Goal::new(vec![x]);

        let mut tactic = SplitTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce two sub-goals: (x, x) and (x, ¬x)
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 2);

            // First goal should have (x, x) -> simplified to just x
            assert_eq!(goals[0].assertions.len(), 2);

            // Second goal should have (x, ¬x) - contradiction
            assert_eq!(goals[1].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_split_tactic_with_constraint() {
        let mut manager = TermManager::new();

        // Goal: (x ∨ y)
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let or_xy = manager.mk_or([x, y]);

        let goal = Goal::new(vec![or_xy]);
        let mut tactic = SplitTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should produce two sub-goals
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 2);

            // Each goal should have 2 assertions (the original plus the split condition)
            assert_eq!(goals[0].assertions.len(), 2);
            assert_eq!(goals[1].assertions.len(), 2);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_split_prefers_variables() {
        let mut manager = TermManager::new();

        // Goal: (x ∧ (y ∨ z))
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let z = manager.mk_var("z", manager.sorts.bool_sort);
        let or_yz = manager.mk_or([y, z]);
        let and_expr = manager.mk_and([x, or_yz]);

        let goal = Goal::new(vec![and_expr]);
        let mut tactic = SplitTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should split on a variable (x, y, or z) rather than the complex term (y ∨ z)
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 2);

            // The split term should be a variable
            let split_term_true = goals[0].assertions.last().unwrap();
            let split_term_false = goals[1].assertions.last().unwrap();

            // One should be a variable, the other should be its negation
            if let Some(term_false) = manager.get(*split_term_false) {
                if let TermKind::Not(inner) = term_false.kind {
                    assert_eq!(inner, *split_term_true);
                }
            }
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_stateless_split() {
        let goal = Goal::empty();
        let tactic = StatelessSplitTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "split");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_eliminate_unconstrained_basic() {
        let mut manager = TermManager::new();
        let sort_int = manager.sorts.int_sort;

        // Create two variables: x and y
        let x = manager.mk_var("x", sort_int);
        let y = manager.mk_var("y", sort_int);

        // Create equation: x = 5 (x is eliminable since it appears once)
        let five = manager.mk_int(5);
        let eq = manager.mk_eq(x, five);

        // Create another constraint that doesn't involve x: y > 0
        let zero = manager.mk_int(0);
        let gt = manager.mk_gt(y, zero);

        let goal = Goal::new(vec![eq, gt]);

        let mut tactic = EliminateUnconstrainedTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // x should be eliminated, leaving only y > 0
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            assert_eq!(goals[0].assertions.len(), 1);
            assert_eq!(goals[0].assertions[0], gt);
        } else {
            panic!("Expected SubGoals result");
        }
    }

    #[test]
    fn test_eliminate_unconstrained_or() {
        let mut manager = TermManager::new();
        let sort_bool = manager.sorts.bool_sort;

        // Create variables x and y
        let x = manager.mk_var("x", sort_bool);
        let y = manager.mk_var("y", sort_bool);

        // Create disjunction: x | y
        // x appears once and can be eliminated from the disjunction
        let or = manager.mk_or(vec![x, y]);

        let goal = Goal::new(vec![or]);

        let mut tactic = EliminateUnconstrainedTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // After eliminating x, the disjunction becomes true (unconstrained)
        if let TacticResult::Solved(SolveResult::Sat) = result {
            // Expected outcome
        } else {
            panic!("Expected Sat result, got {:?}", result);
        }
    }

    #[test]
    fn test_eliminate_unconstrained_not_applicable() {
        let mut manager = TermManager::new();
        let sort_int = manager.sorts.int_sort;

        // Create variable x that appears multiple times
        let x = manager.mk_var("x", sort_int);

        // Create constraints where x appears twice
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);
        let eq1 = manager.mk_eq(x, five);
        let eq2 = manager.mk_eq(x, ten);

        let goal = Goal::new(vec![eq1, eq2]);

        let mut tactic = EliminateUnconstrainedTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // x appears twice, so it's not eliminable
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_eliminate_unconstrained_circular_dependency() {
        let mut manager = TermManager::new();
        let sort_int = manager.sorts.int_sort;

        // Create variables x and y
        let x = manager.mk_var("x", sort_int);
        let y = manager.mk_var("y", sort_int);

        // Create equation where x depends on itself: x = x + y
        // This shouldn't be eliminable
        let sum = manager.mk_add(vec![x, y]);
        let eq = manager.mk_eq(x, sum);

        let goal = Goal::new(vec![eq]);

        let mut tactic = EliminateUnconstrainedTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // x appears on both sides, so it shouldn't be eliminated
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_stateless_eliminate_unconstrained() {
        let goal = Goal::empty();
        let tactic = StatelessEliminateUnconstrainedTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "elim-uncnstr");
        assert!(!tactic.description().is_empty());
    }

    // ==================== Solve-Eqs Tactic Tests ====================

    #[test]
    fn test_solve_eqs_simple() {
        let mut manager = TermManager::new();

        // Create: x = 5, y = x + 3
        // After solving: x = 5, y = 8, then y = 8 simplifies to Sat
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);

        let eq_x = manager.mk_eq(x, five);
        let x_plus_3 = manager.mk_add([x, three]);
        let eq_y = manager.mk_eq(y, x_plus_3);

        let goal = Goal::new(vec![eq_x, eq_y]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should solve x = 5, substitute into y = x + 3 -> y = 8
        // Then solve y = 8, which results in all equations being solved (SAT)
        // or SubGoals with reduced assertions
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // Both equations got solved completely
            }
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Should have at most one assertion left
                assert!(goals[0].assertions.len() <= 1);
            }
            _ => panic!("Expected SubGoals or Sat result, got {:?}", result),
        }
    }

    #[test]
    fn test_solve_eqs_chain() {
        let mut manager = TermManager::new();

        // Create: x = 5, y = x, z = y
        // After solving: all equalities are resolved
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let z = manager.mk_var("z", int_sort);
        let five = manager.mk_int(5);

        let eq_x = manager.mk_eq(x, five);
        let eq_y = manager.mk_eq(y, x);
        let eq_z = manager.mk_eq(z, y);

        let goal = Goal::new(vec![eq_x, eq_y, eq_z]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should solve the chain
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // All equations simplified to true
            }
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Should have fewer assertions
                assert!(goals[0].assertions.len() < 3);
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_solve_eqs_not_applicable() {
        let mut manager = TermManager::new();

        // Create: x + y = 10 (no simple var = expr form)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let ten = manager.mk_int(10);

        let sum = manager.mk_add([x, y]);
        let eq = manager.mk_eq(sum, ten);
        let lt = manager.mk_lt(x, y); // Another constraint

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should solve (x + y) = 10 => x = 10 - y
        // Then substitute x in the second constraint
        if let TacticResult::SubGoals(goals) = result {
            assert_eq!(goals.len(), 1);
            // The goal should be simplified
            assert!(goals[0].assertions.len() <= 2);
        } else if let TacticResult::NotApplicable = result {
            // If we can't solve linear add yet, this is acceptable
        } else {
            panic!("Unexpected result: {:?}", result);
        }
    }

    #[test]
    fn test_solve_eqs_unsat() {
        let mut manager = TermManager::new();

        // Create: x = 5, x = 10 (contradiction)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let eq1 = manager.mk_eq(x, five);
        let eq2 = manager.mk_eq(x, ten);

        let goal = Goal::new(vec![eq1, eq2]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After substituting x = 5 into x = 10, we get 5 = 10 which is false
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_solve_eqs_sat() {
        let mut manager = TermManager::new();

        // Create: x = 5, x = 5 (trivially true)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let eq1 = manager.mk_eq(x, five);
        let eq2 = manager.mk_eq(x, five);

        let goal = Goal::new(vec![eq1, eq2]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Both equations should simplify to true
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_solve_eqs_with_constraints() {
        let mut manager = TermManager::new();

        // Create: x = 5, x < 10
        // After solving: 5 < 10 => true => SAT
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, ten);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After substituting x = 5 into x < 10, we get 5 < 10 which is true
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_solve_eqs_cyclic() {
        let mut manager = TermManager::new();

        // Create: x = y, y = x (cyclic, should still work)
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);

        let eq1 = manager.mk_eq(x, y);
        let eq2 = manager.mk_eq(y, x);

        let goal = Goal::new(vec![eq1, eq2]);
        let mut tactic = SolveEqsTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Should be able to solve at least one equation
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // Both simplified to true (x=y, y=y -> true, true)
            }
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert!(goals[0].assertions.len() <= 2);
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_stateless_solve_eqs() {
        let goal = Goal::empty();
        let tactic = StatelessSolveEqsTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "solve-eqs");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_timeout_tactic_completes() {
        let goal = Goal::empty();
        let tactic = std::sync::Arc::new(StatelessSimplifyTactic);
        let timeout_tactic = TimeoutTactic::new(tactic, 1000); // 1 second timeout

        let result = timeout_tactic.apply(&goal).unwrap();
        // Should complete quickly (stateless simplify returns SubGoals)
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(timeout_tactic.name(), "timeout");
    }

    #[test]
    fn test_timeout_tactic_exceeds() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        // Create a custom slow tactic for testing
        struct SlowTactic {
            delay_ms: u64,
        }

        impl Tactic for SlowTactic {
            fn name(&self) -> &str {
                "slow"
            }

            fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
                thread::sleep(Duration::from_millis(self.delay_ms));
                Ok(TacticResult::NotApplicable)
            }

            fn description(&self) -> &str {
                "A slow tactic for testing timeouts"
            }
        }

        let goal = Goal::empty();
        let slow_tactic = Arc::new(SlowTactic { delay_ms: 500 });
        let timeout_tactic = TimeoutTactic::new(slow_tactic, 100); // 100ms timeout, but tactic takes 500ms

        let result = timeout_tactic.apply(&goal).unwrap();
        // Should timeout and return Failed
        match result {
            TacticResult::Failed(msg) => {
                assert!(msg.contains("timed out"));
            }
            _ => panic!("Expected timeout failure, got: {:?}", result),
        }
    }

    #[test]
    fn test_timeout_from_box() {
        let goal = Goal::empty();
        let tactic = Box::new(StatelessSimplifyTactic);
        let timeout_tactic = TimeoutTactic::from_box(tactic, 1000);

        let result = timeout_tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    // ========================================================================
    // ScriptableTactic tests
    // ========================================================================

    #[test]
    fn test_scriptable_tactic_basic() {
        let script = r#"
            #{result: "unchanged", assertions: assertions}
        "#
        .to_string();

        let tactic =
            ScriptableTactic::new("test-script".to_string(), script, "Test script".to_string())
                .unwrap();

        let goal = Goal::new(vec![TermId(1), TermId(2)]);
        let result = tactic.apply(&goal).unwrap();

        assert!(matches!(result, TacticResult::NotApplicable));
        assert_eq!(tactic.name(), "test-script");
        assert_eq!(tactic.description(), "Test script");
    }

    #[test]
    fn test_scriptable_tactic_return_sat() {
        let script = r#"
            #{result: "sat"}
        "#
        .to_string();

        let tactic =
            ScriptableTactic::new("sat-tactic".to_string(), script, "Returns SAT".to_string())
                .unwrap();

        let goal = Goal::empty();
        let result = tactic.apply(&goal).unwrap();

        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));
    }

    #[test]
    fn test_scriptable_tactic_return_unsat() {
        let script = r#"
            #{result: "unsat"}
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "unsat-tactic".to_string(),
            script,
            "Returns UNSAT".to_string(),
        )
        .unwrap();

        let goal = Goal::empty();
        let result = tactic.apply(&goal).unwrap();

        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_scriptable_tactic_return_unknown() {
        let script = r#"
            #{result: "unknown"}
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "unknown-tactic".to_string(),
            script,
            "Returns Unknown".to_string(),
        )
        .unwrap();

        let goal = Goal::empty();
        let result = tactic.apply(&goal).unwrap();

        assert!(matches!(result, TacticResult::Solved(SolveResult::Unknown)));
    }

    #[test]
    fn test_scriptable_tactic_modify_assertions() {
        let script = r#"
            // Filter out assertion with ID 2
            let new_assertions = [];
            let i = 0;
            while i < assertions.len {
                let a = assertions[i];
                if a != 2 {
                    new_assertions.push(a);
                }
                i += 1;
            }
            #{result: "modified", assertions: new_assertions}
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "filter-tactic".to_string(),
            script,
            "Filters assertions".to_string(),
        )
        .unwrap();

        let goal = Goal::new(vec![TermId(1), TermId(2), TermId(3)]);
        let result = tactic.apply(&goal).unwrap();

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert_eq!(goals[0].assertions, vec![TermId(1), TermId(3)]);
            }
            _ => panic!("Expected SubGoals result"),
        }
    }

    #[test]
    fn test_scriptable_tactic_compilation_error() {
        let script = r#"
            // Invalid Rhai syntax
            this is not valid rhai code !@#$
        "#
        .to_string();

        let result = ScriptableTactic::new(
            "invalid-tactic".to_string(),
            script,
            "Invalid script".to_string(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_scriptable_tactic_access_assertions() {
        let script = r#"
            // Check assertion count
            if assertions.len == 0 {
                #{result: "sat"}
            } else {
                #{result: "unchanged", assertions: assertions}
            }
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "check-empty".to_string(),
            script,
            "Check if goal is empty".to_string(),
        )
        .unwrap();

        // Test with empty goal
        let empty_goal = Goal::empty();
        let result = tactic.apply(&empty_goal).unwrap();
        assert!(matches!(result, TacticResult::Solved(SolveResult::Sat)));

        // Test with non-empty goal
        let non_empty_goal = Goal::new(vec![TermId(1)]);
        let result = tactic.apply(&non_empty_goal).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_scriptable_tactic_complex_logic() {
        let script = r#"
            // Remove duplicate assertion IDs
            let seen = [];
            let unique = [];
            let i = 0;
            while i < assertions.len {
                let a = assertions[i];
                if !seen.contains(a) {
                    seen.push(a);
                    unique.push(a);
                }
                i += 1;
            }
            if unique.len != assertions.len {
                #{result: "modified", assertions: unique}
            } else {
                #{result: "unchanged"}
            }
        "#
        .to_string();

        let tactic = ScriptableTactic::new(
            "deduplicate".to_string(),
            script,
            "Remove duplicate assertions".to_string(),
        )
        .unwrap();

        // Goal with duplicates
        let goal_with_dupes =
            Goal::new(vec![TermId(1), TermId(2), TermId(1), TermId(3), TermId(2)]);
        let result = tactic.apply(&goal_with_dupes).unwrap();

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert_eq!(goals[0].assertions, vec![TermId(1), TermId(2), TermId(3)]);
            }
            _ => panic!("Expected SubGoals result"),
        }

        // Goal without duplicates
        let goal_no_dupes = Goal::new(vec![TermId(1), TermId(2), TermId(3)]);
        let result = tactic.apply(&goal_no_dupes).unwrap();
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    // ========================================================================
    // Fourier-Motzkin Tactic Tests
    // ========================================================================

    #[test]
    fn test_fm_not_applicable_non_linear() {
        let mut manager = TermManager::new();

        // Create a goal with non-linear constraints (not applicable)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let eq = manager.mk_eq(x, y);

        let goal = Goal::new(vec![eq]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();
        // Equality constraints are not directly handled by FM
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_fm_simple_inequality() {
        let mut manager = TermManager::new();

        // x < 5
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let five = manager.mk_int(5);
        let lt = manager.mk_lt(x, five);

        let goal = Goal::new(vec![lt]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Single bound with only one variable - should still be applicable
        // since the variable has only upper bound but no lower bound
        match result {
            TacticResult::SubGoals(goals) => {
                // Variable eliminated (no lower bound)
                assert!(goals[0].assertions.is_empty() || goals[0].assertions.len() <= 1);
            }
            TacticResult::Solved(SolveResult::Sat) => {
                // All constraints eliminated
            }
            TacticResult::NotApplicable => {
                // No elimination happened (acceptable since only one constraint)
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_fm_bounded_variable() {
        let mut manager = TermManager::new();

        // x >= 0 and x <= 10 (lower and upper bounds)
        // Rewritten as: -x <= 0 and x <= 10
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let ten = manager.mk_int(10);
        let lower = manager.mk_ge(x, zero); // x >= 0
        let upper = manager.mk_le(x, ten); // x <= 10

        let goal = Goal::new(vec![lower, upper]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After FM elimination of x: 0 <= 10 (tautology)
        match result {
            TacticResult::Solved(SolveResult::Sat) => {
                // All constraints resolved to tautologies
            }
            TacticResult::SubGoals(goals) => {
                // Reduced goal
                assert!(goals.len() == 1);
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_fm_unsat_bounds() {
        let mut manager = TermManager::new();

        // x >= 10 and x <= 5 (infeasible)
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ten = manager.mk_int(10);
        let five = manager.mk_int(5);
        let lower = manager.mk_ge(x, ten); // x >= 10
        let upper = manager.mk_le(x, five); // x <= 5

        let goal = Goal::new(vec![lower, upper]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // After FM: 10 <= 5 (contradiction)
        assert!(matches!(result, TacticResult::Solved(SolveResult::Unsat)));
    }

    #[test]
    fn test_fm_two_variables() {
        let mut manager = TermManager::new();

        // x + y <= 10 and x >= 0 and y >= 0
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let ten = manager.mk_int(10);

        let sum = manager.mk_add([x, y]);
        let c1 = manager.mk_le(sum, ten); // x + y <= 10
        let c2 = manager.mk_ge(x, zero); // x >= 0
        let c3 = manager.mk_ge(y, zero); // y >= 0

        let goal = Goal::new(vec![c1, c2, c3]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // FM should eliminate some variables
        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Some constraints may remain
            }
            TacticResult::Solved(SolveResult::Sat) => {
                // All constraints eliminated (system is feasible)
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_fm_coefficient_scaling() {
        let mut manager = TermManager::new();

        // 2x <= 10 and x >= 1
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let two = manager.mk_int(2);
        let one = manager.mk_int(1);
        let ten = manager.mk_int(10);

        let two_x = manager.mk_mul([two, x]);
        let c1 = manager.mk_le(two_x, ten); // 2x <= 10, i.e., x <= 5

        let c2 = manager.mk_ge(x, one); // x >= 1

        let goal = Goal::new(vec![c1, c2]);
        let mut tactic = FourierMotzkinTactic::new(&mut manager);

        let result = tactic.apply_mut(&goal).unwrap();

        // Feasible: 1 <= x <= 5
        match result {
            TacticResult::Solved(SolveResult::Sat) | TacticResult::SubGoals(_) => {
                // Expected
            }
            TacticResult::Solved(SolveResult::Unsat) => {
                panic!("System should be SAT");
            }
            _ => panic!("Unexpected result: {:?}", result),
        }
    }

    #[test]
    fn test_stateless_fm() {
        let goal = Goal::empty();
        let tactic = StatelessFourierMotzkinTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "fm");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_stateless_pb2bv() {
        let goal = Goal::empty();
        let tactic = StatelessPb2BvTactic;

        let result = tactic.apply(&goal).unwrap();
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert_eq!(tactic.name(), "pb2bv");
        assert!(!tactic.description().is_empty());
    }

    #[test]
    fn test_pb2bv_not_applicable() {
        let mut manager = TermManager::new();

        // A non-PB constraint (regular boolean)
        let p = manager.mk_var("p", manager.sorts.bool_sort);
        let goal = Goal::new(vec![p]);

        let mut tactic = Pb2BvTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).unwrap();

        // Should be not applicable since it's just a boolean variable
        assert!(matches!(result, TacticResult::NotApplicable));
    }
}
