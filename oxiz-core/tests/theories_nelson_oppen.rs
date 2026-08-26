//! End-to-end Nelson-Oppen runs over `oxiz_core::theories`.
//!
//! These drive [`TheoryCombiner`] with real theory objects and check that a
//! fact deduced inside one theory reaches the other one, which is the whole
//! point of the combination: the datatype theory learns `x = #x01` from
//! injectivity, and the bit-vector theory — which is where `x = #x00` was
//! asserted — is the one that turns the pair into a conflict.

use oxiz_core::ast::{TermId, TermManager};
use oxiz_core::sort::{DataTypeConstructor, SortId};
use oxiz_core::theories::{
    ArrayTheory, BitVectorTheory, CombinerOutcome, DatatypeTheory, TheoryCombiner,
};

/// Declare `Pair` with `mk(first: BV8, second: BV8)` and a nullary `none`.
fn declare_pair(manager: &mut TermManager) -> (SortId, SortId) {
    let bv_sort = manager.sorts.bitvec(8);
    let pair_sort = manager.sorts.mk_datatype_sort("Pair");

    let first = manager.sorts.intern_str("first");
    let second = manager.sorts.intern_str("second");
    let mk = manager.sorts.intern_str("mk");
    let none = manager.sorts.intern_str("none");

    manager.sorts.declare_datatype(
        "Pair",
        vec![
            DataTypeConstructor {
                name: mk,
                selectors: vec![(first, bv_sort), (second, bv_sort)].into(),
            },
            DataTypeConstructor {
                name: none,
                selectors: Vec::new().into(),
            },
        ],
    );

    (pair_sort, bv_sort)
}

#[test]
fn datatype_injectivity_reaches_the_bitvector_theory() {
    let mut manager = TermManager::new();
    let (pair_sort, bv_sort) = declare_pair(&mut manager);

    let x = manager.mk_var("x", bv_sort);
    let y = manager.mk_var("y", bv_sort);
    let zero = manager.mk_bitvec(0, 8);
    let one = manager.mk_bitvec(1, 8);

    // mk(x, y) and mk(#x01, y): asserting them equal forces x = #x01.
    let left = manager.mk_dt_constructor("mk", [x, y], pair_sort);
    let right = manager.mk_dt_constructor("mk", [one, y], pair_sort);

    let mut combiner = TheoryCombiner::new(0);
    let datatype_index = combiner.add_theory(Box::new(DatatypeTheory::new()));
    let bitvector_index = combiner.add_theory(Box::new(BitVectorTheory::new()));
    assert_eq!(datatype_index, 0);
    assert_eq!(bitvector_index, 1);

    for term in [x, y, zero, one, left, right] {
        combiner.add_term(term, &manager);
    }

    // x is a bit-vector term that also appears inside a datatype term, so it
    // is exactly the kind of shared term the exchange runs on.
    assert!(
        combiner.shared_variables().contains(&x),
        "x should be shared between the datatype and bit-vector theories"
    );
    assert_eq!(
        combiner.theories_of(left),
        Some([datatype_index].as_slice())
    );

    // The bit-vector theory is told x = #x00; the datatype theory is told
    // mk(x, y) = mk(#x01, y). Neither is a conflict on its own.
    combiner.assert_equality(x, zero);
    combiner.assert_equality(left, right);

    match combiner.run(&mut manager) {
        CombinerOutcome::Conflict {
            theory,
            explanation,
        } => {
            assert_eq!(
                theory, bitvector_index,
                "the bit-vector theory is the one holding both constants"
            );
            assert!(
                explanation.contains(&zero) && explanation.contains(&one),
                "the explanation should name both constants, got {explanation:?}"
            );
        }
        CombinerOutcome::NoConflict => {
            panic!("x = #x00 and x = #x01 should be a conflict");
        }
    }
}

/// Control for the test above: without the datatype theory there is nothing to
/// deduce `x = #x01` from, so the same assertions produce no conflict. The
/// conflict in that test is therefore the exchange doing its job, not the
/// bit-vector theory seeing both constants by itself.
#[test]
fn the_bitvector_theory_alone_finds_no_conflict() {
    let mut manager = TermManager::new();
    let (pair_sort, bv_sort) = declare_pair(&mut manager);

    let x = manager.mk_var("x", bv_sort);
    let y = manager.mk_var("y", bv_sort);
    let zero = manager.mk_bitvec(0, 8);
    let one = manager.mk_bitvec(1, 8);
    let left = manager.mk_dt_constructor("mk", [x, y], pair_sort);
    let right = manager.mk_dt_constructor("mk", [one, y], pair_sort);

    let mut combiner = TheoryCombiner::new(0);
    combiner.add_theory(Box::new(BitVectorTheory::new()));

    for term in [x, y, zero, one, left, right] {
        combiner.add_term(term, &manager);
    }
    combiner.assert_equality(x, zero);
    combiner.assert_equality(left, right);

    assert_eq!(combiner.run(&mut manager), CombinerOutcome::NoConflict);
}

#[test]
fn a_consistent_combination_reaches_a_fixpoint() {
    let mut manager = TermManager::new();
    let (pair_sort, bv_sort) = declare_pair(&mut manager);

    let x = manager.mk_var("x", bv_sort);
    let y = manager.mk_var("y", bv_sort);
    let one = manager.mk_bitvec(1, 8);

    let left = manager.mk_dt_constructor("mk", [x, y], pair_sort);
    let right = manager.mk_dt_constructor("mk", [one, y], pair_sort);

    let mut combiner = TheoryCombiner::new(0);
    combiner.add_theory(Box::new(DatatypeTheory::new()));
    combiner.add_theory(Box::new(BitVectorTheory::new()));

    for term in [x, y, one, left, right] {
        combiner.add_term(term, &manager);
    }
    combiner.assert_equality(left, right);

    // x = #x01 follows, and nothing contradicts it.
    assert_eq!(combiner.run(&mut manager), CombinerOutcome::NoConflict);
}

#[test]
fn constructor_distinctness_is_reported_by_the_datatype_theory() {
    let mut manager = TermManager::new();
    let (pair_sort, bv_sort) = declare_pair(&mut manager);

    let x = manager.mk_var("x", bv_sort);
    let y = manager.mk_var("y", bv_sort);
    let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
    let empty = manager.mk_dt_constructor("none", [], pair_sort);

    let mut combiner = TheoryCombiner::new(0);
    let datatype_index = combiner.add_theory(Box::new(DatatypeTheory::new()));
    combiner.add_theory(Box::new(BitVectorTheory::new()));

    for term in [x, y, pair, empty] {
        combiner.add_term(term, &manager);
    }
    combiner.assert_equality(pair, empty);

    match combiner.run(&mut manager) {
        CombinerOutcome::Conflict {
            theory,
            explanation,
        } => {
            assert_eq!(theory, datatype_index);
            assert!(explanation.contains(&pair) && explanation.contains(&empty));
        }
        CombinerOutcome::NoConflict => panic!("mk(x, y) = none should be a conflict"),
    }
}

#[test]
fn selector_application_travels_from_datatypes_to_bitvectors() {
    let mut manager = TermManager::new();
    let (pair_sort, bv_sort) = declare_pair(&mut manager);

    let x = manager.mk_var("x", bv_sort);
    let y = manager.mk_var("y", bv_sort);
    let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
    let first = manager.mk_dt_selector("first", pair, bv_sort);
    let zero = manager.mk_bitvec(0, 8);
    let one = manager.mk_bitvec(1, 8);

    let mut combiner = TheoryCombiner::new(0);
    combiner.add_theory(Box::new(DatatypeTheory::new()));
    let bitvector_index = combiner.add_theory(Box::new(BitVectorTheory::new()));

    for term in [x, y, pair, first, zero, one] {
        combiner.add_term(term, &manager);
    }

    // first(mk(x, y)) = x is the datatype theory's job; the clash between the
    // two constants is the bit-vector theory's.
    combiner.assert_equality(x, zero);
    combiner.assert_equality(first, one);

    match combiner.run(&mut manager) {
        CombinerOutcome::Conflict { theory, .. } => assert_eq!(theory, bitvector_index),
        CombinerOutcome::NoConflict => {
            panic!("first(mk(x, y)) = #x01 with x = #x00 should be a conflict");
        }
    }
}

#[test]
fn the_array_theory_contributes_lemmas_to_the_combiner() {
    let mut manager = TermManager::new();

    let int_sort = manager.sorts.int_sort;
    let array_sort = manager.sorts.array(int_sort, int_sort);
    let a = manager.mk_var("a", array_sort);
    let i = manager.mk_var("i", int_sort);
    let v = manager.mk_int(7);
    let store = manager.mk_store(a, i, v);
    let select = manager.mk_select(store, i);

    let mut combiner = TheoryCombiner::new(0);
    combiner.add_theory(Box::new(ArrayTheory::new()));

    for term in [a, i, store, select] {
        combiner.add_term(term, &manager);
    }

    assert_eq!(combiner.run(&mut manager), CombinerOutcome::NoConflict);

    let lemmas: Vec<TermId> = combiner.take_lemmas();
    assert!(
        !lemmas.is_empty(),
        "select(store(a, i, v), i) should produce a read-over-write lemma"
    );
    assert!(combiner.lemmas().is_empty(), "taking should drain them");
}

#[test]
fn resetting_the_combiner_resets_its_theories() {
    let mut manager = TermManager::new();
    let (pair_sort, bv_sort) = declare_pair(&mut manager);

    let x = manager.mk_var("x", bv_sort);
    let y = manager.mk_var("y", bv_sort);
    let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
    let empty = manager.mk_dt_constructor("none", [], pair_sort);

    let mut combiner = TheoryCombiner::new(0);
    combiner.add_theory(Box::new(DatatypeTheory::new()));

    for term in [pair, empty] {
        combiner.add_term(term, &manager);
    }
    combiner.assert_equality(pair, empty);
    assert!(matches!(
        combiner.run(&mut manager),
        CombinerOutcome::Conflict { .. }
    ));

    combiner.reset();

    // The conflicting equality is gone from every theory, so the same terms
    // no longer clash.
    for term in [pair, empty] {
        combiner.add_term(term, &manager);
    }
    assert_eq!(combiner.run(&mut manager), CombinerOutcome::NoConflict);
    assert!(combiner.shared_variables().is_empty() || !combiner.shared_variables().contains(&x));
}
