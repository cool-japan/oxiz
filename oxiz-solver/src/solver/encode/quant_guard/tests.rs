//! Unit tests for the polarity-complete quantifier guard.
//!
//! Each one drives the pass directly and inspects the obligations it produced,
//! so a mutation that keeps the end-to-end verdict by accident (the finite
//! expansion deciding a small width, say) still reddens here.

use super::{Pol, guard_conditional_quantifiers};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::smtlib::{RESERVED_PREFIX, is_reserved_tag};

/// `(forall ((x Int)) (= (f x) x))` and the symbols it needs.
///
/// The body is deliberately *not* a tautology: `mk_eq` folds `(= t t)` to
/// `True` on construction, and `mk_implies(g, True)` folds again, so a
/// tautological body would leave nothing for these tests to look at.
fn simple_forall(manager: &mut TermManager) -> TermId {
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let fx = manager.mk_apply("f", vec![x], int_sort);
    let body = manager.mk_eq(fx, x);
    manager.mk_forall([("x", int_sort)], body)
}

/// Every `Var` name introduced by `term` that carries the reserved prefix.
fn reserved_names(term: TermId, manager: &TermManager) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut stack = vec![term];
    let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        if let TermKind::Var(name) = &kind {
            let resolved = manager.resolve_str(*name).to_string();
            if resolved.starts_with(RESERVED_PREFIX) {
                out.push(resolved);
            }
        }
        stack.extend(oxiz_core::ast::traversal::get_children(&kind));
    }
    out.sort();
    out
}

/// A quantifier the assertion states unconditionally is left alone: it is
/// already registered as a fact, and guarding it twice would be a second,
/// weaker copy of the same obligation.
#[test]
fn an_unconditional_quantifier_is_not_guarded() {
    let mut manager = TermManager::new();
    let quantifier = simple_forall(&mut manager);
    let mut next_skolem_id = 0u64;
    assert!(
        guard_conditional_quantifiers(quantifier, &mut manager, &mut next_skolem_id).is_none(),
        "a top-level `forall` is on the asserted spine and must keep its \
         unconditional MBQI registration"
    );
    assert_eq!(next_skolem_id, 0, "declining must mint nothing");
}

/// The same quantifier as a conjunct is still unconditional.
#[test]
fn a_conjunct_is_not_guarded() {
    let mut manager = TermManager::new();
    let quantifier = simple_forall(&mut manager);
    let bool_sort = manager.sorts.bool_sort;
    let p = manager.mk_var("p", bool_sort);
    let assertion = manager.mk_and(vec![p, quantifier]);
    let mut next_skolem_id = 0u64;
    assert!(
        guard_conditional_quantifiers(assertion, &mut manager, &mut next_skolem_id).is_none(),
        "`(and p Q)` entails `Q`"
    );
}

/// `(=> p Q)`: `Q` is at positive polarity but conditional, so it earns the
/// *universal* obligation `∀x. (g → φ)` and nothing else.
#[test]
fn a_positive_conditional_forall_earns_one_guarded_universal() {
    let mut manager = TermManager::new();
    let quantifier = simple_forall(&mut manager);
    let bool_sort = manager.sorts.bool_sort;
    let p = manager.mk_var("p", bool_sort);
    let assertion = manager.mk_implies(p, quantifier);
    let mut next_skolem_id = 0u64;
    let guarded = guard_conditional_quantifiers(assertion, &mut manager, &mut next_skolem_id)
        .expect("a `forall` under `=>` is conditional");

    assert_eq!(
        guarded.obligations.len(),
        1,
        "a purely positive occurrence needs the universal half only"
    );
    let obligation = guarded.obligations[0];
    let TermKind::Forall { body, .. } = manager
        .get(obligation)
        .map(|t| t.kind.clone())
        .expect("term")
    else {
        panic!("the universal obligation must still be a `forall`");
    };
    assert!(
        matches!(
            manager.get(body).map(|t| &t.kind),
            Some(TermKind::Implies(_, _))
        ),
        "the guard must be pushed *inside* the binder, so every instance of it \
         is `g => instance`"
    );
    let names = reserved_names(obligation, &manager);
    assert!(
        names.iter().any(|name| is_reserved_tag(name, "qg")),
        "the guard constant must be minted through `reserved_name`, got {names:?}"
    );
    assert!(
        !names.iter().any(|name| is_reserved_tag(name, "sk")),
        "a positive occurrence needs no witness, got {names:?}"
    );
    // The quantifier itself is gone from the encoded term: its literal is the
    // guard constant's now.
    assert!(
        !reserved_names(guarded.term, &manager).is_empty(),
        "the assertion must mention the guard constant"
    );
}

/// `(not Q)`: an existential obligation, discharged with a Skolem constant.
#[test]
fn a_negated_forall_earns_a_skolem_witness() {
    let mut manager = TermManager::new();
    let quantifier = simple_forall(&mut manager);
    let assertion = manager.mk_not(quantifier);
    let mut next_skolem_id = 0u64;
    let guarded = guard_conditional_quantifiers(assertion, &mut manager, &mut next_skolem_id)
        .expect("`(not Q)` puts `Q` at negative polarity");

    assert_eq!(
        guarded.obligations.len(),
        1,
        "a purely negative occurrence needs the witness half only"
    );
    let obligation = guarded.obligations[0];
    assert!(
        matches!(
            manager.get(obligation).map(|t| &t.kind),
            Some(TermKind::Or(_))
        ),
        "the witness obligation is `g OR NOT phi(sk)`"
    );
    let names = reserved_names(obligation, &manager);
    assert!(
        names.iter().any(|name| is_reserved_tag(name, "sk")),
        "the witness must be minted through `reserved_name`, got {names:?}"
    );
    assert!(
        !manager
            .get(obligation)
            .map(|t| t.kind.clone())
            .into_iter()
            .any(|kind| matches!(kind, TermKind::Forall { .. })),
        "a negative occurrence must not be handed back to MBQI as a universal"
    );
}

/// A Boolean `=` is a polarity boundary, so the occurrence is `Both` and the
/// guard constant is given the *full* definition — the shape that is sound in
/// any position whatsoever.
#[test]
fn a_boolean_equality_earns_both_halves() {
    let mut manager = TermManager::new();
    let quantifier = simple_forall(&mut manager);
    let bool_sort = manager.sorts.bool_sort;
    let p = manager.mk_var("p", bool_sort);
    let assertion = manager.mk_eq(p, quantifier);
    let mut next_skolem_id = 0u64;
    let guarded = guard_conditional_quantifiers(assertion, &mut manager, &mut next_skolem_id)
        .expect("a `forall` inside a Boolean `=` is conditional");
    assert_eq!(
        guarded.obligations.len(),
        2,
        "an occurrence whose polarity cannot be pinned down needs `g <-> Q`"
    );
    let names: Vec<String> = guarded
        .obligations
        .iter()
        .flat_map(|&o| reserved_names(o, &manager))
        .collect();
    assert!(names.iter().any(|name| is_reserved_tag(name, "qg")));
    assert!(names.iter().any(|name| is_reserved_tag(name, "sk")));
}

/// A quantifier under another binder is declined: it may mention the enclosing
/// binder's variables, so a closed guard constant cannot stand for it.  The
/// enclosing quantifier's *instances* are ground, and the pass runs on those.
#[test]
fn a_quantifier_under_a_binder_is_declined() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let bool_sort = manager.sorts.bool_sort;
    let j = manager.mk_var("j", int_sort);
    let i = manager.mk_var("i", int_sort);
    let inner_body = manager.mk_eq(i, j);
    let inner = manager.mk_forall([("i", int_sort)], inner_body);
    let p = manager.mk_var("p", bool_sort);
    let outer_body = manager.mk_implies(p, inner);
    let outer = manager.mk_forall([("j", int_sort)], outer_body);
    let mut next_skolem_id = 0u64;
    assert!(
        guard_conditional_quantifiers(outer, &mut manager, &mut next_skolem_id).is_none(),
        "the inner quantifier mentions `j`, so no closed constant can stand \
         for it and the pass must decline rather than guess"
    );
}

/// The counter is threaded, not restarted: two guards in one assertion must
/// not share a symbol, which would be a strengthening.
#[test]
fn two_guards_in_one_assertion_get_distinct_symbols() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let bool_sort = manager.sorts.bool_sort;
    let x = manager.mk_var("x", int_sort);
    let fx = manager.mk_apply("f", vec![x], int_sort);
    let gx = manager.mk_apply("g", vec![x], int_sort);
    let body_one = manager.mk_eq(fx, x);
    let body_two = manager.mk_eq(gx, x);
    let first = manager.mk_forall([("x", int_sort)], body_one);
    let second = manager.mk_forall([("x", int_sort)], body_two);
    let p = manager.mk_var("p", bool_sort);
    let left = manager.mk_implies(p, first);
    let right = manager.mk_implies(p, second);
    let assertion = manager.mk_or(vec![left, right]);
    let mut next_skolem_id = 7u64;
    let guarded = guard_conditional_quantifiers(assertion, &mut manager, &mut next_skolem_id)
        .expect("both quantifiers are conditional");
    assert_eq!(guarded.obligations.len(), 2);
    let mut names: Vec<String> = guarded
        .obligations
        .iter()
        .flat_map(|&o| reserved_names(o, &manager))
        .collect();
    names.sort();
    names.dedup();
    assert_eq!(
        names.len(),
        2,
        "each quantifier needs its own guard constant, got {names:?}"
    );
    assert!(next_skolem_id > 7, "the caller's counter must be advanced");
}

/// The polarity lattice, stated directly: `Both` absorbs, `flip` is an
/// involution on the two definite values.
#[test]
fn the_polarity_lattice_is_what_the_table_says() {
    assert_eq!(Pol::Pos.flip(), Pol::Neg);
    assert_eq!(Pol::Neg.flip(), Pol::Pos);
    assert_eq!(Pol::Both.flip(), Pol::Both);
    assert_eq!(Pol::Pos.join(Pol::Neg), Pol::Both);
    assert_eq!(Pol::Pos.join(Pol::Pos), Pol::Pos);
    assert_eq!(Pol::Both.join(Pol::Pos), Pol::Both);
    assert!(Pol::Pos.has_pos() && !Pol::Pos.has_neg());
    assert!(Pol::Neg.has_neg() && !Pol::Neg.has_pos());
    assert!(Pol::Both.has_pos() && Pol::Both.has_neg());
}
