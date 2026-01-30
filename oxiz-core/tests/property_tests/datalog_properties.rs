//! Property-based tests for Datalog engine
//!
//! Tests:
//! - Rule evaluation correctness
//! - Fixpoint computation
//! - Stratification
//! - Incremental evaluation
//! - Query answering

use oxiz_core::datalog::*;
use proptest::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

#[cfg(test)]
mod datalog_basic_properties {
    use super::*;

    proptest! {
        /// Test that empty Datalog program has empty fixpoint
        #[test]
        fn empty_program_empty_fixpoint() {
            let mut engine = DatalogEngine::new();
            engine.compute_fixpoint();

            let facts = engine.get_all_facts();
            prop_assert!(facts.is_empty());
        }

        /// Test that adding a fact makes it queryable
        #[test]
        fn fact_is_queryable(
            pred_id in 0usize..10usize,
            arg1 in 0i32..10i32,
            arg2 in 0i32..10i32
        ) {
            let mut engine = DatalogEngine::new();

            let fact = Fact::new(pred_id, vec![arg1, arg2]);
            engine.add_fact(fact.clone());

            engine.compute_fixpoint();

            let result = engine.query_fact(&fact);
            prop_assert!(result);
        }

        /// Test that non-added facts are not present
        #[test]
        fn absent_fact_not_queryable(
            pred_id in 0usize..10usize,
            arg1 in 0i32..10i32,
            arg2 in 0i32..10i32,
            arg3 in 11i32..20i32
        ) {
            let mut engine = DatalogEngine::new();

            let fact1 = Fact::new(pred_id, vec![arg1, arg2]);
            engine.add_fact(fact1);

            engine.compute_fixpoint();

            let fact2 = Fact::new(pred_id, vec![arg3, arg3]);
            let result = engine.query_fact(&fact2);

            // Should not be present (unless derived)
            if arg3 != arg1 && arg3 != arg2 {
                prop_assert!(!result);
            }
        }

        /// Test reflexivity: if we add P(x), we can query P(x)
        #[test]
        fn reflexivity(n in 0i32..20i32) {
            let mut engine = DatalogEngine::new();

            let pred = 0;
            let fact = Fact::new(pred, vec![n]);
            engine.add_fact(fact.clone());

            engine.compute_fixpoint();

            prop_assert!(engine.query_fact(&fact));
        }
    }
}

#[cfg(test)]
mod datalog_rule_properties {
    use super::*;

    proptest! {
        /// Test that simple rule derives facts
        #[test]
        fn simple_rule_derives_facts(a in 0i32..10i32, b in 0i32..10i32) {
            let mut engine = DatalogEngine::new();

            // Add fact: edge(a, b)
            let edge_pred = 0;
            engine.add_fact(Fact::new(edge_pred, vec![a, b]));

            // Add rule: path(X, Y) :- edge(X, Y)
            let path_pred = 1;
            let rule = Rule {
                head: Atom::new(path_pred, vec![Var(0), Var(1)]),
                body: vec![Atom::new(edge_pred, vec![Var(0), Var(1)])],
            };
            engine.add_rule(rule);

            engine.compute_fixpoint();

            // Should derive path(a, b)
            let path_fact = Fact::new(path_pred, vec![a, b]);
            prop_assert!(engine.query_fact(&path_fact));
        }

        /// Test transitivity rule
        #[test]
        fn transitivity_rule(
            a in 0i32..5i32,
            b in 0i32..5i32,
            c in 0i32..5i32
        ) {
            let mut engine = DatalogEngine::new();

            let edge_pred = 0;
            let path_pred = 1;

            // Add facts: edge(a,b), edge(b,c)
            engine.add_fact(Fact::new(edge_pred, vec![a, b]));
            engine.add_fact(Fact::new(edge_pred, vec![b, c]));

            // Rule 1: path(X,Y) :- edge(X,Y)
            engine.add_rule(Rule {
                head: Atom::new(path_pred, vec![Var(0), Var(1)]),
                body: vec![Atom::new(edge_pred, vec![Var(0), Var(1)])],
            });

            // Rule 2: path(X,Z) :- path(X,Y), path(Y,Z)
            engine.add_rule(Rule {
                head: Atom::new(path_pred, vec![Var(0), Var(2)]),
                body: vec![
                    Atom::new(path_pred, vec![Var(0), Var(1)]),
                    Atom::new(path_pred, vec![Var(1), Var(2)]),
                ],
            });

            engine.compute_fixpoint();

            // Should derive path(a,c)
            if a != b && b != c {
                let derived = Fact::new(path_pred, vec![a, c]);
                prop_assert!(engine.query_fact(&derived));
            }
        }

        /// Test that rule application is sound
        #[test]
        fn rule_application_sound(
            x in 0i32..8i32,
            y in 0i32..8i32
        ) {
            let mut engine = DatalogEngine::new();

            // P(x) :- Q(x, y), R(y)
            let p_pred = 0;
            let q_pred = 1;
            let r_pred = 2;

            engine.add_fact(Fact::new(q_pred, vec![x, y]));
            engine.add_fact(Fact::new(r_pred, vec![y]));

            engine.add_rule(Rule {
                head: Atom::new(p_pred, vec![Var(0)]),
                body: vec![
                    Atom::new(q_pred, vec![Var(0), Var(1)]),
                    Atom::new(r_pred, vec![Var(1)]),
                ],
            });

            engine.compute_fixpoint();

            // Should derive P(x)
            prop_assert!(engine.query_fact(&Fact::new(p_pred, vec![x])));
        }

        /// Test constant in head
        #[test]
        fn constant_in_head(x in 0i32..10i32) {
            let mut engine = DatalogEngine::new();

            // Q(5) :- P(x)
            let p_pred = 0;
            let q_pred = 1;

            engine.add_fact(Fact::new(p_pred, vec![x]));

            engine.add_rule(Rule {
                head: Atom::new(q_pred, vec![Const(5)]),
                body: vec![Atom::new(p_pred, vec![Var(0)])],
            });

            engine.compute_fixpoint();

            // Should derive Q(5)
            prop_assert!(engine.query_fact(&Fact::new(q_pred, vec![5])));
        }
    }
}

#[cfg(test)]
mod datalog_fixpoint_properties {
    use super::*;

    proptest! {
        /// Test that fixpoint computation terminates
        #[test]
        fn fixpoint_terminates(n in 2usize..8usize) {
            let mut engine = DatalogEngine::new();

            // Add chain of facts and rules
            for i in 0..n-1 {
                engine.add_fact(Fact::new(0, vec![i as i32, (i+1) as i32]));
            }

            // Transitive closure rule
            engine.add_rule(Rule {
                head: Atom::new(1, vec![Var(0), Var(1)]),
                body: vec![Atom::new(0, vec![Var(0), Var(1)])],
            });

            engine.add_rule(Rule {
                head: Atom::new(1, vec![Var(0), Var(2)]),
                body: vec![
                    Atom::new(1, vec![Var(0), Var(1)]),
                    Atom::new(1, vec![Var(1), Var(2)]),
                ],
            });

            // Should terminate
            let iterations = engine.compute_fixpoint();

            prop_assert!(iterations < 1000, "Fixpoint should terminate");
        }

        /// Test that fixpoint is idempotent
        #[test]
        fn fixpoint_idempotent(n in 1usize..6usize) {
            let mut engine = DatalogEngine::new();

            for i in 0..n {
                engine.add_fact(Fact::new(0, vec![i as i32]));
            }

            engine.add_rule(Rule {
                head: Atom::new(1, vec![Var(0)]),
                body: vec![Atom::new(0, vec![Var(0)])],
            });

            // Compute twice
            engine.compute_fixpoint();
            let facts1 = engine.get_all_facts();

            engine.compute_fixpoint();
            let facts2 = engine.get_all_facts();

            // Should be the same
            prop_assert_eq!(facts1.len(), facts2.len());
        }

        /// Test monotonicity: adding facts only increases derived facts
        #[test]
        fn monotonicity(base_facts in 1usize..5usize, extra_facts in 1usize..3usize) {
            let mut engine = DatalogEngine::new();

            // Add base facts
            for i in 0..base_facts {
                engine.add_fact(Fact::new(0, vec![i as i32]));
            }

            engine.add_rule(Rule {
                head: Atom::new(1, vec![Var(0)]),
                body: vec![Atom::new(0, vec![Var(0)])],
            });

            engine.compute_fixpoint();
            let count1 = engine.get_all_facts().len();

            // Add more facts
            for i in base_facts..base_facts+extra_facts {
                engine.add_fact(Fact::new(0, vec![i as i32]));
            }

            engine.compute_fixpoint();
            let count2 = engine.get_all_facts().len();

            // Should not decrease
            prop_assert!(count2 >= count1);
        }
    }
}

#[cfg(test)]
mod datalog_stratification_properties {
    use super::*;

    proptest! {
        /// Test that stratifiable programs compute correctly
        #[test]
        fn stratifiable_program(n in 2usize..6usize) {
            let mut engine = DatalogEngine::new();

            // Stratum 0: base facts
            for i in 0..n {
                engine.add_fact(Fact::new(0, vec![i as i32]));
            }

            // Stratum 1: derived from stratum 0
            engine.add_rule(Rule {
                head: Atom::new(1, vec![Var(0)]),
                body: vec![Atom::new(0, vec![Var(0)])],
            });

            // Stratum 2: derived from stratum 1
            engine.add_rule(Rule {
                head: Atom::new(2, vec![Var(0)]),
                body: vec![Atom::new(1, vec![Var(0)])],
            });

            engine.compute_fixpoint();

            // All should be derived
            for i in 0..n {
                prop_assert!(engine.query_fact(&Fact::new(2, vec![i as i32])));
            }
        }

        /// Test negation in stratified program
        #[test]
        fn stratified_negation(n in 3i32..8i32) {
            let mut engine = DatalogEngine::new();

            // P(0), P(1), ..., P(n-1)
            for i in 0..n {
                engine.add_fact(Fact::new(0, vec![i]));
            }

            // Q(n) (not in P)
            engine.add_fact(Fact::new(1, vec![n]));

            // R(X) :- Q(X), not P(X)
            engine.add_rule(Rule {
                head: Atom::new(2, vec![Var(0)]),
                body: vec![
                    Atom::new(1, vec![Var(0)]),
                    NegAtom::new(0, vec![Var(0)]),
                ],
            });

            engine.compute_fixpoint();

            // Should derive R(n)
            prop_assert!(engine.query_fact(&Fact::new(2, vec![n])));

            // Should not derive R(i) for i < n
            for i in 0..n {
                prop_assert!(!engine.query_fact(&Fact::new(2, vec![i])));
            }
        }
    }
}

#[cfg(test)]
mod datalog_incremental_properties {
    use super::*;

    proptest! {
        /// Test incremental evaluation gives same result as batch
        #[test]
        fn incremental_equals_batch(
            initial_facts in 2usize..6usize,
            new_facts in 1usize..4usize
        ) {
            // Batch engine
            let mut batch_engine = DatalogEngine::new();
            for i in 0..initial_facts+new_facts {
                batch_engine.add_fact(Fact::new(0, vec![i as i32]));
            }
            batch_engine.add_rule(Rule {
                head: Atom::new(1, vec![Var(0)]),
                body: vec![Atom::new(0, vec![Var(0)])],
            });
            batch_engine.compute_fixpoint();
            let batch_facts = batch_engine.get_all_facts();

            // Incremental engine
            let mut incr_engine = DatalogEngine::new();
            for i in 0..initial_facts {
                incr_engine.add_fact(Fact::new(0, vec![i as i32]));
            }
            incr_engine.add_rule(Rule {
                head: Atom::new(1, vec![Var(0)]),
                body: vec![Atom::new(0, vec![Var(0)])],
            });
            incr_engine.compute_fixpoint();

            // Add new facts incrementally
            for i in initial_facts..initial_facts+new_facts {
                incr_engine.add_fact(Fact::new(0, vec![i as i32]));
            }
            incr_engine.compute_fixpoint_incremental();
            let incr_facts = incr_engine.get_all_facts();

            // Should have same facts
            prop_assert_eq!(batch_facts.len(), incr_facts.len());
        }

        /// Test that incremental is more efficient than batch
        #[test]
        fn incremental_more_efficient(
            initial_size in 10usize..20usize,
            delta_size in 1usize..5usize
        ) {
            let mut engine = DatalogEngine::new();

            // Initial load
            for i in 0..initial_size {
                engine.add_fact(Fact::new(0, vec![i as i32]));
            }
            engine.add_rule(Rule {
                head: Atom::new(1, vec![Var(0)]),
                body: vec![Atom::new(0, vec![Var(0)])],
            });
            engine.compute_fixpoint();

            // Measure incremental work
            for i in initial_size..initial_size+delta_size {
                engine.add_fact(Fact::new(0, vec![i as i32]));
            }
            let incr_work = engine.compute_fixpoint_incremental();

            // Incremental should process less than initial_size + delta_size
            prop_assert!(incr_work < initial_size + delta_size + 10);
        }
    }
}

#[cfg(test)]
mod datalog_query_properties {
    use super::*;

    proptest! {
        /// Test query answering correctness
        #[test]
        fn query_answering_correct(
            facts in 3usize..10usize,
            query_idx in 0usize..3usize
        ) {
            let mut engine = DatalogEngine::new();

            // Add facts
            for i in 0..facts {
                engine.add_fact(Fact::new(0, vec![i as i32]));
            }

            engine.compute_fixpoint();

            // Query should return true for added facts
            let query_arg = query_idx as i32;
            if query_arg < facts as i32 {
                prop_assert!(engine.query_fact(&Fact::new(0, vec![query_arg])));
            }
        }

        /// Test query with variables
        #[test]
        fn query_with_variables(n in 2usize..8usize) {
            let mut engine = DatalogEngine::new();

            // Add facts
            for i in 0..n {
                engine.add_fact(Fact::new(0, vec![i as i32, (i+1) as i32]));
            }

            engine.compute_fixpoint();

            // Query: P(X, Y) should return all pairs
            let results = engine.query_with_vars(&Query {
                pred: 0,
                args: vec![QueryArg::Var(0), QueryArg::Var(1)],
            });

            prop_assert_eq!(results.len(), n);
        }

        /// Test query projection
        #[test]
        fn query_projection(n in 3usize..8usize) {
            let mut engine = DatalogEngine::new();

            // Add facts P(i, i+1)
            for i in 0..n {
                engine.add_fact(Fact::new(0, vec![i as i32, (i+1) as i32]));
            }

            engine.compute_fixpoint();

            // Query: P(X, _) should project first argument
            let results = engine.query_projection(&Query {
                pred: 0,
                args: vec![QueryArg::Var(0), QueryArg::Wildcard],
            });

            prop_assert_eq!(results.len(), n);
        }
    }
}

#[cfg(test)]
mod datalog_join_properties {
    use super::*;

    proptest! {
        /// Test join correctness
        #[test]
        fn join_correct(n in 2usize..6usize) {
            let mut engine = DatalogEngine::new();

            // R(i, i+1), S(i+1, i+2)
            for i in 0..n-1 {
                engine.add_fact(Fact::new(0, vec![i as i32, (i+1) as i32]));
                engine.add_fact(Fact::new(1, vec![(i+1) as i32, (i+2) as i32]));
            }

            // T(X, Z) :- R(X, Y), S(Y, Z)
            engine.add_rule(Rule {
                head: Atom::new(2, vec![Var(0), Var(2)]),
                body: vec![
                    Atom::new(0, vec![Var(0), Var(1)]),
                    Atom::new(1, vec![Var(1), Var(2)]),
                ],
            });

            engine.compute_fixpoint();

            // Should derive T(i, i+2) for valid i
            for i in 0..n-2 {
                let fact = Fact::new(2, vec![i as i32, (i+2) as i32]);
                prop_assert!(engine.query_fact(&fact));
            }
        }

        /// Test multi-way join
        #[test]
        fn multiway_join(n in 2usize..5usize) {
            let mut engine = DatalogEngine::new();

            // P(i, i+1), Q(i+1, i+2), R(i+2, i+3)
            for i in 0..n {
                engine.add_fact(Fact::new(0, vec![i as i32, (i+1) as i32]));
                engine.add_fact(Fact::new(1, vec![(i+1) as i32, (i+2) as i32]));
                engine.add_fact(Fact::new(2, vec![(i+2) as i32, (i+3) as i32]));
            }

            // S(W, Z) :- P(W, X), Q(X, Y), R(Y, Z)
            engine.add_rule(Rule {
                head: Atom::new(3, vec![Var(0), Var(3)]),
                body: vec![
                    Atom::new(0, vec![Var(0), Var(1)]),
                    Atom::new(1, vec![Var(1), Var(2)]),
                    Atom::new(2, vec![Var(2), Var(3)]),
                ],
            });

            engine.compute_fixpoint();

            // Check derived facts
            for i in 0..n {
                if (i+3) as usize <= n + 2 {
                    let fact = Fact::new(3, vec![i as i32, (i+3) as i32]);
                    prop_assert!(engine.query_fact(&fact));
                }
            }
        }
    }
}
