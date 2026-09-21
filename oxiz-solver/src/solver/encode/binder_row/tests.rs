//! Unit tests for the under-a-binder read-over-write expansion.

use super::*;
use oxiz_core::ast::traversal::collect_subterms;

/// How many `(select (store …) …)` terms survive in `term`.
fn reads_over_writes(term: TermId, manager: &TermManager) -> usize {
    collect_subterms(term, manager)
        .into_iter()
        .filter(|&subterm| {
            let Some(TermKind::Select(array, _)) = manager.get(subterm).map(|t| t.kind.clone())
            else {
                return false;
            };
            matches!(
                manager.get(array).map(|t| &t.kind),
                Some(TermKind::Store(..))
            )
        })
        .count()
}

/// How many `ite` terms `term` carries.
fn ites(term: TermId, manager: &TermManager) -> usize {
    collect_subterms(term, manager)
        .into_iter()
        .filter(|&subterm| {
            matches!(
                manager.get(subterm).map(|t| &t.kind),
                Some(TermKind::Ite(..))
            )
        })
        .count()
}

/// The defect's own shape: `∀i. (select (store a i v) k) = v` keeps no
/// read-over-write at all once the pass has run.
#[test]
fn a_read_over_write_under_a_forall_is_expanded() {
    let mut manager = TermManager::new();
    let index_sort = manager.sorts.bitvec(7);
    let elem_sort = manager.sorts.bitvec(7);
    let array_sort = manager.sorts.array(index_sort, elem_sort);

    let array = manager.mk_var("a", array_sort);
    let bound = manager.mk_var("i", index_sort);
    let five = manager.mk_bitvec(5u32, 7);
    let one = manager.mk_bitvec(1u32, 7);
    let store = manager.mk_store(array, bound, five);
    let read = manager.mk_select(store, one);
    let body = manager.mk_eq(read, five);
    let quantifier = manager.mk_forall([("i", index_sort)], body);

    assert_eq!(reads_over_writes(quantifier, &manager), 1);
    let rewritten = expand_reads_over_writes_under_binders(quantifier, &mut manager)
        .expect("the read-over-write under the binder is expandable");
    assert_eq!(
        reads_over_writes(rewritten, &manager),
        0,
        "every read-over-write under the binder is gone"
    );
    assert_eq!(
        ites(rewritten, &manager),
        1,
        "one case split took its place"
    );
}

/// A ground read-over-write is the lazy axiom's business and must be left
/// exactly as it is: expanding it eagerly would change the cost model of every
/// ground array benchmark.
#[test]
fn a_ground_read_over_write_is_left_alone() {
    let mut manager = TermManager::new();
    let index_sort = manager.sorts.bitvec(3);
    let array_sort = manager.sorts.array(index_sort, index_sort);

    let array = manager.mk_var("a", array_sort);
    let zero = manager.mk_bitvec(0u32, 3);
    let one = manager.mk_bitvec(1u32, 3);
    let store = manager.mk_store(array, zero, one);
    let read = manager.mk_select(store, one);
    let body = manager.mk_eq(read, one);

    assert!(
        expand_reads_over_writes_under_binders(body, &mut manager).is_none(),
        "no binder, no rewrite"
    );
}

/// A chain of stores becomes a chain of `ite`s, innermost store last, and the
/// base array keeps exactly one plain `select`.
#[test]
fn a_store_chain_under_a_binder_is_peeled_to_its_base() {
    let mut manager = TermManager::new();
    let index_sort = manager.sorts.bitvec(4);
    let array_sort = manager.sorts.array(index_sort, index_sort);

    let array = manager.mk_var("a", array_sort);
    let bound = manager.mk_var("i", index_sort);
    let two = manager.mk_bitvec(2u32, 4);
    let three = manager.mk_bitvec(3u32, 4);
    let seven = manager.mk_bitvec(7u32, 4);
    let free_index = manager.mk_var("k", index_sort);
    let inner = manager.mk_store(array, bound, two);
    let outer = manager.mk_store(inner, three, seven);
    let read = manager.mk_select(outer, free_index);
    let body = manager.mk_eq(read, seven);
    let quantifier = manager.mk_forall([("i", index_sort)], body);

    let rewritten = expand_reads_over_writes_under_binders(quantifier, &mut manager)
        .expect("a two-store chain is expandable");
    assert_eq!(reads_over_writes(rewritten, &manager), 0);
    assert_eq!(
        ites(rewritten, &manager),
        2,
        "one case split per store in the chain"
    );
}

/// The `exists` direction rewrites too: the axiom is polarity-independent.
#[test]
fn a_read_over_write_under_an_exists_is_expanded() {
    let mut manager = TermManager::new();
    let index_sort = manager.sorts.bitvec(5);
    let array_sort = manager.sorts.array(index_sort, index_sort);

    let array = manager.mk_var("a", array_sort);
    let bound = manager.mk_var("j", index_sort);
    let nine = manager.mk_bitvec(9u32, 5);
    let store = manager.mk_store(array, bound, nine);
    let read = manager.mk_select(store, nine);
    let body = manager.mk_eq(read, nine);
    let quantifier = manager.mk_exists([("j", index_sort)], body);

    let rewritten = expand_reads_over_writes_under_binders(quantifier, &mut manager)
        .expect("an exists body is rewritten exactly like a forall body");
    assert_eq!(reads_over_writes(rewritten, &manager), 0);
    assert!(matches!(
        manager.get(rewritten).map(|t| &t.kind),
        Some(TermKind::Exists { .. })
    ));
}

/// A trigger mentioning the term this pass would rewrite declines the whole
/// quantifier: rewriting the body while leaving the trigger alone would
/// silently disable e-matching for it, and rewriting the trigger would emit an
/// `ite`, which is not a legal pattern.
#[test]
fn a_trigger_mentioning_the_read_declines_the_rewrite() {
    let mut manager = TermManager::new();
    let index_sort = manager.sorts.bitvec(3);
    let array_sort = manager.sorts.array(index_sort, index_sort);

    let array = manager.mk_var("a", array_sort);
    let bound = manager.mk_var("i", index_sort);
    let one = manager.mk_bitvec(1u32, 3);
    let store = manager.mk_store(array, bound, one);
    let read = manager.mk_select(store, one);
    let body = manager.mk_eq(read, one);
    let quantifier = manager.mk_forall_with_patterns([("i", index_sort)], body, [[read]]);

    assert!(
        expand_reads_over_writes_under_binders(quantifier, &mut manager).is_none(),
        "a quantifier whose trigger names the read is declined"
    );
}

/// A trigger that names something else does not stand in the way.
#[test]
fn a_trigger_naming_another_term_still_allows_the_rewrite() {
    let mut manager = TermManager::new();
    let index_sort = manager.sorts.bitvec(3);
    let array_sort = manager.sorts.array(index_sort, index_sort);

    let array = manager.mk_var("a", array_sort);
    let bound = manager.mk_var("i", index_sort);
    let one = manager.mk_bitvec(1u32, 3);
    let store = manager.mk_store(array, bound, one);
    let read = manager.mk_select(store, one);
    let plain = manager.mk_select(array, bound);
    let body = manager.mk_eq(read, one);
    let quantifier = manager.mk_forall_with_patterns([("i", index_sort)], body, [[plain]]);

    let rewritten = expand_reads_over_writes_under_binders(quantifier, &mut manager)
        .expect("an unrelated trigger does not block the rewrite");
    assert_eq!(reads_over_writes(rewritten, &manager), 0);
    match manager.get(rewritten).map(|t| t.kind.clone()) {
        Some(TermKind::Forall { patterns, .. }) => {
            assert_eq!(patterns.len(), 1, "the trigger survives the rewrite");
        }
        other => panic!("expected a forall, got {other:?}"),
    }
}

/// The rewrite keeps the binder's own variable: the replacement mentions it,
/// and a capture-avoiding `substitute` over the *quantifier* would have
/// alpha-renamed it away.  This is the guard that makes the whole pass work
/// rather than silently do nothing.
#[test]
fn the_bound_variable_survives_in_the_expansion() {
    let mut manager = TermManager::new();
    let index_sort = manager.sorts.bitvec(6);
    let array_sort = manager.sorts.array(index_sort, index_sort);

    let array = manager.mk_var("a", array_sort);
    let bound = manager.mk_var("i", index_sort);
    let four = manager.mk_bitvec(4u32, 6);
    let store = manager.mk_store(array, bound, four);
    let read = manager.mk_select(store, four);
    let body = manager.mk_eq(read, four);
    let quantifier = manager.mk_forall([("i", index_sort)], body);

    let rewritten =
        expand_reads_over_writes_under_binders(quantifier, &mut manager).expect("expandable");
    let Some(TermKind::Forall { vars, body, .. }) = manager.get(rewritten).map(|t| t.kind.clone())
    else {
        panic!("expected a forall");
    };
    assert_eq!(vars.len(), 1);
    let name = manager.resolve_str(vars[0].0).to_string();
    assert_eq!(name, "i", "the binder is not alpha-renamed");
    assert!(
        collect_subterms(body, &manager).contains(&bound),
        "the expansion still reads the bound variable"
    );
}
