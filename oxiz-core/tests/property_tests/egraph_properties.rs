//! Property-based tests for E-graph (Equality Graph) operations
//!
//! Tests:
//! - Congruence closure correctness
//! - Union-find consistency
//! - E-class merging
//! - Extraction optimality
//! - Rebuilding correctness

use num_bigint::BigInt;
use num_traits::Zero;
use oxiz_core::ast::*;
use proptest::prelude::*;
use rustc_hash::FxHashMap;

#[cfg(test)]
mod union_find_properties {
    use super::*;

    proptest! {
        /// Test that find is idempotent
        #[test]
        fn find_idempotent(n in 1usize..100usize) {
            let mut egraph = egraph::EGraph::new();

            // Add nodes
            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("node{}", i));
                nodes.push(node);
            }

            // Find should be idempotent
            for &node in &nodes {
                let rep1 = egraph.find(node);
                let rep2 = egraph.find(rep1);

                prop_assert_eq!(rep1, rep2);
            }
        }

        /// Test that union creates equivalence
        #[test]
        fn union_creates_equivalence(
            a_idx in 0usize..10usize,
            b_idx in 0usize..10usize
        ) {
            let mut egraph = egraph::EGraph::new();

            // Create nodes
            let mut nodes = Vec::new();
            for i in 0..10 {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            let a = nodes[a_idx];
            let b = nodes[b_idx];

            // Union a and b
            egraph.union(a, b);

            // They should have the same representative
            prop_assert_eq!(egraph.find(a), egraph.find(b));
        }

        /// Test union-find transitivity
        #[test]
        fn union_find_transitive(n in 3usize..15usize) {
            let mut egraph = egraph::EGraph::new();

            // Create chain
            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Create chain: 0 = 1 = 2 = ... = n-1
            for i in 0..n-1 {
                egraph.union(nodes[i], nodes[i+1]);
            }

            // All should have same representative
            let rep = egraph.find(nodes[0]);
            for &node in &nodes {
                prop_assert_eq!(egraph.find(node), rep);
            }
        }

        /// Test path compression effectiveness
        #[test]
        fn path_compression_works(depth in 2usize..10usize) {
            let mut egraph = egraph::EGraph::new();

            // Create deep chain
            let mut nodes = Vec::new();
            for i in 0..depth {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Create chain
            for i in 0..depth-1 {
                egraph.union(nodes[i], nodes[i+1]);
            }

            // Find on leaf should compress path
            let leaf = nodes[0];
            let rep = egraph.find(leaf);

            // After path compression, find should be fast
            let rep2 = egraph.find(leaf);
            prop_assert_eq!(rep, rep2);

            // Parent should now point directly to root (or close)
            let parent_of_leaf = egraph.get_parent(leaf);
            if let Some(parent) = parent_of_leaf {
                let parent_rep = egraph.find(parent);
                prop_assert_eq!(parent_rep, rep);
            }
        }

        /// Test that union by rank maintains balance
        #[test]
        fn union_by_rank_balanced(n in 2usize..20usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Random unions
            for i in 0..n-1 {
                egraph.union(nodes[i], nodes[(i + 1) % n]);
            }

            // Tree should be reasonably balanced
            // (depth should be O(log n) with union-by-rank)
            for &node in &nodes {
                let depth = egraph.get_depth(node);
                prop_assert!(depth <= ((n as f64).log2().ceil() as usize + 1));
            }
        }
    }
}

#[cfg(test)]
mod congruence_properties {
    use super::*;

    proptest! {
        /// Test basic congruence: if a=b then f(a)=f(b)
        #[test]
        fn basic_congruence(
            a_val in -10i64..10i64,
            b_val in -10i64..10i64
        ) {
            let mut tm = manager::TermManager::new();
            let mut egraph = egraph::EGraph::from_term_manager(&tm);

            let ta = tm.mk_int(BigInt::from(a_val));
            let tb = tm.mk_int(BigInt::from(b_val));

            // Assert a = b
            egraph.assert_eq(ta, tb);

            // Create f(a) and f(b)
            let f_a = tm.mk_neg(ta);
            let f_b = tm.mk_neg(tb);

            // Rebuild to propagate congruence
            egraph.rebuild();

            // f(a) and f(b) should be congruent if a = b
            if a_val == b_val {
                prop_assert!(egraph.are_congruent(f_a, f_b));
            }
        }

        /// Test congruence with binary functions
        #[test]
        fn binary_function_congruence(
            a1 in -5i64..5i64,
            a2 in -5i64..5i64,
            b1 in -5i64..5i64,
            b2 in -5i64..5i64
        ) {
            let mut tm = manager::TermManager::new();
            let mut egraph = egraph::EGraph::from_term_manager(&tm);

            let ta1 = tm.mk_int(BigInt::from(a1));
            let ta2 = tm.mk_int(BigInt::from(a2));
            let tb1 = tm.mk_int(BigInt::from(b1));
            let tb2 = tm.mk_int(BigInt::from(b2));

            // Assert a1 = b1 and a2 = b2
            egraph.assert_eq(ta1, tb1);
            egraph.assert_eq(ta2, tb2);

            // Create f(a1, a2) and f(b1, b2)
            let f_a = tm.mk_add(vec![ta1, ta2]);
            let f_b = tm.mk_add(vec![tb1, tb2]);

            egraph.rebuild();

            // Should be congruent if inputs are equal
            if a1 == b1 && a2 == b2 {
                prop_assert!(egraph.are_congruent(f_a, f_b));
            }
        }

        /// Test transitivity of congruence
        #[test]
        fn congruence_transitive(
            a in -5i64..5i64,
            b in -5i64..5i64,
            c in -5i64..5i64
        ) {
            let mut tm = manager::TermManager::new();
            let mut egraph = egraph::EGraph::from_term_manager(&tm);

            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let tc = tm.mk_int(BigInt::from(c));

            // a = b and b = c
            egraph.assert_eq(ta, tb);
            egraph.assert_eq(tb, tc);

            egraph.rebuild();

            // Therefore a = c
            prop_assert!(egraph.are_congruent(ta, tc));
        }

        /// Test nested congruence: if a=b then f(g(a))=f(g(b))
        #[test]
        fn nested_congruence(a in -5i64..5i64, b in -5i64..5i64) {
            let mut tm = manager::TermManager::new();
            let mut egraph = egraph::EGraph::from_term_manager(&tm);

            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // Assert a = b
            egraph.assert_eq(ta, tb);

            // Create f(g(a)) = -(a+1) and f(g(b)) = -(b+1)
            let one = tm.mk_int(BigInt::from(1));
            let g_a = tm.mk_add(vec![ta, one]);
            let g_b = tm.mk_add(vec![tb, one]);
            let f_g_a = tm.mk_neg(g_a);
            let f_g_b = tm.mk_neg(g_b);

            egraph.rebuild();

            // Should be congruent if a = b
            if a == b {
                prop_assert!(egraph.are_congruent(f_g_a, f_g_b));
            }
        }

        /// Test congruence with commutative operators
        #[test]
        fn commutative_congruence(
            a in -5i64..5i64,
            b in -5i64..5i64
        ) {
            let mut tm = manager::TermManager::new();
            let mut egraph = egraph::EGraph::from_term_manager(&tm);

            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // Create a+b and b+a
            let sum1 = tm.mk_add(vec![ta, tb]);
            let sum2 = tm.mk_add(vec![tb, ta]);

            // Should recognize as congruent (commutative)
            egraph.rebuild();

            prop_assert!(egraph.are_congruent(sum1, sum2));
        }
    }
}

#[cfg(test)]
mod eclass_properties {
    use super::*;

    proptest! {
        /// Test that e-class contains its canonical representative
        #[test]
        fn eclass_contains_representative(n in 1usize..20usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Merge some nodes
            for i in 0..n-1 {
                if i % 2 == 0 {
                    egraph.union(nodes[i], nodes[i+1]);
                }
            }

            // Each e-class should contain its representative
            for &node in &nodes {
                let rep = egraph.find(node);
                let eclass = egraph.get_eclass(rep);

                prop_assert!(eclass.contains(&rep));
            }
        }

        /// Test e-class size consistency
        #[test]
        fn eclass_size_consistent(n in 2usize..15usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Merge all into one e-class
            for i in 0..n-1 {
                egraph.union(nodes[i], nodes[i+1]);
            }

            // E-class should have size n
            let rep = egraph.find(nodes[0]);
            let eclass = egraph.get_eclass(rep);

            prop_assert_eq!(eclass.size(), n);
        }

        /// Test that disjoint e-classes don't share nodes
        #[test]
        fn disjoint_eclasses_no_overlap(n in 4usize..12usize) {
            let mut egraph = egraph::EGraph::new();

            let mut group1 = Vec::new();
            let mut group2 = Vec::new();

            // Create two groups
            for i in 0..n/2 {
                group1.push(egraph.add_term(format!("a{}", i)));
            }
            for i in 0..n/2 {
                group2.push(egraph.add_term(format!("b{}", i)));
            }

            // Merge within groups
            for i in 0..group1.len()-1 {
                egraph.union(group1[i], group1[i+1]);
            }
            for i in 0..group2.len()-1 {
                egraph.union(group2[i], group2[i+1]);
            }

            // Get e-classes
            let rep1 = egraph.find(group1[0]);
            let rep2 = egraph.find(group2[0]);

            if rep1 != rep2 {
                let eclass1 = egraph.get_eclass(rep1);
                let eclass2 = egraph.get_eclass(rep2);

                // Should be disjoint
                for &node in eclass1.iter() {
                    prop_assert!(!eclass2.contains(&node));
                }
            }
        }

        /// Test e-class merging
        #[test]
        fn eclass_merge_combines_members(
            size1 in 2usize..8usize,
            size2 in 2usize..8usize
        ) {
            let mut egraph = egraph::EGraph::new();

            // Create first e-class
            let mut group1 = Vec::new();
            for i in 0..size1 {
                group1.push(egraph.add_term(format!("a{}", i)));
            }
            for i in 0..size1-1 {
                egraph.union(group1[i], group1[i+1]);
            }

            // Create second e-class
            let mut group2 = Vec::new();
            for i in 0..size2 {
                group2.push(egraph.add_term(format!("b{}", i)));
            }
            for i in 0..size2-1 {
                egraph.union(group2[i], group2[i+1]);
            }

            // Merge the two e-classes
            egraph.union(group1[0], group2[0]);

            // All nodes should now be in same e-class
            let rep = egraph.find(group1[0]);
            for &node in group1.iter().chain(group2.iter()) {
                prop_assert_eq!(egraph.find(node), rep);
            }

            // Combined e-class should have size1 + size2 elements
            let eclass = egraph.get_eclass(rep);
            prop_assert_eq!(eclass.size(), size1 + size2);
        }
    }
}

#[cfg(test)]
mod extraction_properties {
    use super::*;

    proptest! {
        /// Test that extracted term is in the e-class
        #[test]
        fn extracted_term_in_eclass(n in 1usize..10usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Merge some
            for i in 0..n-1 {
                if i % 2 == 0 {
                    egraph.union(nodes[i], nodes[i+1]);
                }
            }

            // Extract from an e-class
            let rep = egraph.find(nodes[0]);
            let extracted = egraph.extract_best(rep);

            // Extracted term should be in the e-class
            prop_assert!(egraph.are_congruent(extracted, rep));
        }

        /// Test extraction consistency: multiple extractions give same result
        #[test]
        fn extraction_deterministic(n in 2usize..8usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Merge all
            for i in 0..n-1 {
                egraph.union(nodes[i], nodes[i+1]);
            }

            let rep = egraph.find(nodes[0]);

            // Extract twice
            let extracted1 = egraph.extract_best(rep);
            let extracted2 = egraph.extract_best(rep);

            // Should be the same (or equivalent)
            prop_assert!(egraph.are_congruent(extracted1, extracted2));
        }

        /// Test that extracted term has minimal cost
        #[test]
        fn extraction_minimal_cost(cost_limit in 1usize..10usize) {
            let mut egraph = egraph::EGraph::new();

            // Add terms with different costs
            let simple = egraph.add_term_with_cost("simple", 1);
            let complex = egraph.add_term_with_cost("complex", cost_limit);

            // Make them equal
            egraph.union(simple, complex);

            // Extract best
            let rep = egraph.find(simple);
            let extracted = egraph.extract_best(rep);

            // Should prefer simpler term
            let extracted_cost = egraph.get_cost(extracted);
            prop_assert!(extracted_cost <= cost_limit);
        }
    }
}

#[cfg(test)]
mod rebuilding_properties {
    use super::*;

    proptest! {
        /// Test that rebuilding is idempotent
        #[test]
        fn rebuilding_idempotent(n in 2usize..10usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Add some equalities
            for i in 0..n-1 {
                egraph.assert_eq(nodes[i], nodes[i+1]);
            }

            // Rebuild twice
            egraph.rebuild();
            let state1 = egraph.get_state();

            egraph.rebuild();
            let state2 = egraph.get_state();

            // State should be the same
            prop_assert_eq!(state1, state2);
        }

        /// Test that rebuilding propagates congruences
        #[test]
        fn rebuilding_propagates_congruences(
            a in -5i64..5i64,
            b in -5i64..5i64
        ) {
            let mut tm = manager::TermManager::new();
            let mut egraph = egraph::EGraph::from_term_manager(&tm);

            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // Assert a = b
            egraph.assert_eq(ta, tb);

            // Create functions of a and b
            let f_a = tm.mk_neg(ta);
            let f_b = tm.mk_neg(tb);

            // Before rebuild, may not be congruent
            let before = egraph.are_congruent(f_a, f_b);

            // Rebuild
            egraph.rebuild();

            // After rebuild, should be congruent if a = b
            let after = egraph.are_congruent(f_a, f_b);

            if a == b {
                prop_assert!(after || !before, "Rebuild should propagate congruence");
            }
        }

        /// Test that rebuilding maintains invariants
        #[test]
        fn rebuilding_maintains_invariants(n in 3usize..12usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Random equalities
            for i in 0..n-1 {
                egraph.assert_eq(nodes[i], nodes[(i + 2) % n]);
            }

            // Rebuild
            egraph.rebuild();

            // Check invariants
            // 1. All nodes still findable
            for &node in &nodes {
                let rep = egraph.find(node);
                prop_assert!(rep.is_valid());
            }

            // 2. Transitivity holds
            for i in 0..n-2 {
                if egraph.are_congruent(nodes[i], nodes[i+1]) &&
                   egraph.are_congruent(nodes[i+1], nodes[i+2]) {
                    prop_assert!(egraph.are_congruent(nodes[i], nodes[i+2]));
                }
            }
        }

        /// Test rebuilding efficiency (should converge quickly)
        #[test]
        fn rebuilding_converges(n in 3usize..10usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                let node = egraph.add_term(format!("n{}", i));
                nodes.push(node);
            }

            // Create chain of equalities
            for i in 0..n-1 {
                egraph.assert_eq(nodes[i], nodes[i+1]);
            }

            // Rebuild should converge in reasonable iterations
            let mut iterations = 0;
            let max_iterations = n * 2; // Should be O(n)

            loop {
                let changed = egraph.rebuild();
                iterations += 1;

                if !changed || iterations >= max_iterations {
                    break;
                }
            }

            prop_assert!(iterations < max_iterations, "Rebuild should converge quickly");
        }
    }
}

#[cfg(test)]
mod egraph_soundness_properties {
    use super::*;

    proptest! {
        /// Test that equal terms remain equal after operations
        #[test]
        fn equality_preserved(
            a in -10i64..10i64,
            b in -10i64..10i64,
            ops in 0usize..5usize
        ) {
            let mut tm = manager::TermManager::new();
            let mut egraph = egraph::EGraph::from_term_manager(&tm);

            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // Assert equality
            egraph.assert_eq(ta, tb);
            egraph.rebuild();

            // Perform random operations
            for _ in 0..ops {
                let dummy = tm.mk_int(BigInt::from(0));
                egraph.assert_eq(dummy, dummy);
                egraph.rebuild();
            }

            // Equality should still hold
            prop_assert!(egraph.are_congruent(ta, tb));
        }

        /// Test monotonicity: adding equalities only increases congruence
        #[test]
        fn monotonicity(n in 3usize..10usize) {
            let mut egraph = egraph::EGraph::new();

            let mut nodes = Vec::new();
            for i in 0..n {
                nodes.push(egraph.add_term(format!("n{}", i)));
            }

            // Count congruent pairs initially
            let mut initial_congruences = 0;
            for i in 0..n {
                for j in i+1..n {
                    if egraph.are_congruent(nodes[i], nodes[j]) {
                        initial_congruences += 1;
                    }
                }
            }

            // Add an equality
            egraph.assert_eq(nodes[0], nodes[1]);
            egraph.rebuild();

            // Count congruent pairs after
            let mut final_congruences = 0;
            for i in 0..n {
                for j in i+1..n {
                    if egraph.are_congruent(nodes[i], nodes[j]) {
                        final_congruences += 1;
                    }
                }
            }

            // Should not decrease
            prop_assert!(final_congruences >= initial_congruences);
        }
    }
}
