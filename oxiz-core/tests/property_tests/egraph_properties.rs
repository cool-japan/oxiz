//! Property-based tests for E-graph (Equality Graph) operations
//!
//! Tests:
//! - E-graph construction
//! - Term addition
//! - E-class operations
//! - Basic congruence properties

use num_bigint::BigInt;
use oxiz_core::ast::{EGraph, TermManager};
use proptest::prelude::*;

/// Test that creating an EGraph works
#[test]
fn egraph_creation() {
    let egraph = EGraph::new();
    let stats = egraph.statistics();
    assert_eq!(stats.num_eclasses, 0);
}

proptest! {
    /// Test that we can add terms to an EGraph
    #[test]
    fn add_term_to_egraph(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let mut egraph = EGraph::new();

        let t = tm.mk_int(BigInt::from(n));
        let eclass = egraph.add_term(t, &tm);

        // Adding a term should return an EClassId
        prop_assert!(eclass.is_some());
    }

    /// Test that adding the same term twice gives the same EClass
    #[test]
    fn same_term_same_eclass(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let mut egraph = EGraph::new();

        let t = tm.mk_int(BigInt::from(n));
        let eclass1 = egraph.add_term(t, &tm);
        let eclass2 = egraph.add_term(t, &tm);

        // Same term should give same EClassId
        prop_assert_eq!(eclass1, eclass2);
    }

    /// Test that different terms give different EClasses initially
    #[test]
    fn different_terms_different_eclasses(a in -50i64..50i64, b in -50i64..50i64) {
        prop_assume!(a != b);

        let mut tm = TermManager::new();
        let mut egraph = EGraph::new();

        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        let eclass_a = egraph.add_term(ta, &tm);
        let eclass_b = egraph.add_term(tb, &tm);

        // Different terms should have different EClassIds
        prop_assert_ne!(eclass_a, eclass_b);
    }

    /// Test that EGraph stats are consistent
    #[test]
    fn egraph_stats_consistent(n in 1usize..10usize) {
        let mut tm = TermManager::new();
        let mut egraph = EGraph::new();

        // Add n distinct terms
        for i in 0..n {
            let t = tm.mk_int(BigInt::from(i as i64));
            let _ = egraph.add_term(t, &tm);
        }

        let stats = egraph.statistics();
        // Each distinct term should be in its own EClass initially
        prop_assert!(stats.num_eclasses >= 1);
        prop_assert!(stats.num_enodes >= 1);
    }

    /// Test that merging EClasses works
    #[test]
    fn merge_eclasses(a in -50i64..50i64, b in -50i64..50i64) {
        prop_assume!(a != b);

        let mut tm = TermManager::new();
        let mut egraph = EGraph::new();

        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        let eclass_a = egraph.add_term(ta, &tm);
        let eclass_b = egraph.add_term(tb, &tm);

        // Assert equality
        egraph.assert_eq(ta, tb, &tm);

        // After merge, both should have the same representative
        if let (Some(ec_a), Some(ec_b)) = (eclass_a, eclass_b) {
            let rep_a = egraph.find(ec_a);
            let rep_b = egraph.find(ec_b);

            prop_assert_eq!(rep_a, rep_b);
        }
    }

    /// Test that merge is transitive
    #[test]
    fn merge_transitive(a in -30i64..30i64, b in -30i64..30i64, c in -30i64..30i64) {
        prop_assume!(a != b && b != c && a != c);

        let mut tm = TermManager::new();
        let mut egraph = EGraph::new();

        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));
        let tc = tm.mk_int(BigInt::from(c));

        let ec_a = egraph.add_term(ta, &tm);
        let ec_b = egraph.add_term(tb, &tm);
        let ec_c = egraph.add_term(tc, &tm);

        // Merge a with b, and b with c
        egraph.assert_eq(ta, tb, &tm);
        egraph.assert_eq(tb, tc, &tm);

        // All three should have the same representative
        if let (Some(ea), Some(eb), Some(ec)) = (ec_a, ec_b, ec_c) {
            let rep_a = egraph.find(ea);
            let rep_b = egraph.find(eb);
            let rep_c = egraph.find(ec);

            prop_assert_eq!(rep_a, rep_b);
            prop_assert_eq!(rep_b, rep_c);
        }
    }

    /// Test adding compound terms
    #[test]
    fn add_compound_terms(a in -30i64..30i64, b in -30i64..30i64) {
        let mut tm = TermManager::new();
        let mut egraph = EGraph::new();

        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));
        let sum = tm.mk_add(vec![ta, tb]);

        egraph.add_term(ta, &tm);
        egraph.add_term(tb, &tm);
        let eclass_sum = egraph.add_term(sum, &tm);

        // Compound term should be added successfully
        prop_assert!(eclass_sum.is_some());
    }
}
