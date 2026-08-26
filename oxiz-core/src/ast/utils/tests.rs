//! Tests for `ast::utils`.
//!
//! The first section is the pre-existing behavioral test suite, moved here
//! verbatim (see `git log` on `ast/utils.rs` for its history) when this
//! module was split up. The second section is new: regression tests for the
//! recursive -> iterative conversion of every walker in this module
//! (`structural_hash`, `structurally_equal`, `is_ground`, `term_complexity`,
//! `alpha_equivalent`, `flatten_associative`).

use super::*;
use crate::ast::term::MatchCase;
use crate::ast::{RoundingMode, TermId, TermKind, TermManager};
use num_bigint::BigInt;
use smallvec::{SmallVec, smallvec};

// =====================================================================
// Pre-existing behavioral tests (moved verbatim from `ast/utils.rs`).
// =====================================================================

#[test]
fn test_alpha_equivalent_simple() {
    let mut manager = TermManager::new();

    // Create two identical terms: x + 1
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let one = manager.mk_int(1);
    let expr1 = manager.mk_add([x, one]);

    // Create y + 1 (should not be alpha-equivalent to x + 1 for free variables)
    let y = manager.mk_var("y", manager.sorts.int_sort);
    let expr2 = manager.mk_add([y, one]);

    // For free variables, different names mean not alpha-equivalent
    assert!(!alpha_equivalent(expr1, expr2, &manager));

    // Same term should be alpha-equivalent to itself
    assert!(alpha_equivalent(expr1, expr1, &manager));
}

#[test]
fn test_alpha_equivalent_constants() {
    let mut manager = TermManager::new();

    let five1 = manager.mk_int(5);
    let five2 = manager.mk_int(5);
    let six = manager.mk_int(6);

    assert!(alpha_equivalent(five1, five2, &manager));
    assert!(!alpha_equivalent(five1, six, &manager));

    let true1 = manager.mk_true();
    let true2 = manager.mk_true();
    let false1 = manager.mk_false();

    assert!(alpha_equivalent(true1, true2, &manager));
    assert!(!alpha_equivalent(true1, false1, &manager));
}

#[test]
fn test_alpha_equivalent_compound() {
    let mut manager = TermManager::new();

    // (and true false)
    let t = manager.mk_true();
    let f = manager.mk_false();
    let expr1 = manager.mk_and([t, f]);

    // (and true false) - same structure
    let expr2 = manager.mk_and([t, f]);

    assert!(alpha_equivalent(expr1, expr2, &manager));

    // (or true false) - different operation
    let expr3 = manager.mk_or([t, f]);
    assert!(!alpha_equivalent(expr1, expr3, &manager));
}

#[test]
fn test_structural_hash_consistency() {
    let mut manager = TermManager::new();

    let x = manager.mk_var("x", manager.sorts.int_sort);
    let y = manager.mk_var("y", manager.sorts.int_sort);
    let one = manager.mk_int(1);

    let expr1 = manager.mk_add([x, one]);
    let expr2 = manager.mk_add([y, one]);

    // Different variables should produce different hashes
    let hash1 = structural_hash(expr1, &manager);
    let hash2 = structural_hash(expr2, &manager);

    // Note: This test could theoretically fail with hash collisions,
    // but it's extremely unlikely
    assert_ne!(hash1, hash2);
}

#[test]
fn test_structurally_equal_basic() {
    let mut manager = TermManager::new();

    let x = manager.mk_var("x", manager.sorts.int_sort);
    let one = manager.mk_int(1);
    let two = manager.mk_int(2);

    let expr1 = manager.mk_add([x, one]);
    let expr2 = manager.mk_add([x, one]);
    let expr3 = manager.mk_add([x, two]);

    assert!(structurally_equal(expr1, expr2, &manager));
    assert!(!structurally_equal(expr1, expr3, &manager));
}

#[test]
fn test_is_ground() {
    let mut manager = TermManager::new();

    let five = manager.mk_int(5);
    assert!(is_ground(five, &manager));

    let x = manager.mk_var("x", manager.sorts.int_sort);
    assert!(!is_ground(x, &manager));

    let expr = manager.mk_add([x, five]);
    assert!(!is_ground(expr, &manager));

    let ten = manager.mk_int(10);
    let ground_expr = manager.mk_add([five, ten]);
    assert!(is_ground(ground_expr, &manager));
}

#[test]
fn test_term_complexity() {
    let mut manager = TermManager::new();

    // Constant: complexity 1
    let five = manager.mk_int(5);
    assert_eq!(term_complexity(five, &manager), 1);

    // Variable: complexity 1
    let x = manager.mk_var("x", manager.sorts.int_sort);
    assert_eq!(term_complexity(x, &manager), 1);

    // Unary operation: 2 + arg complexity
    let neg_x = manager.mk_neg(x);
    assert_eq!(term_complexity(neg_x, &manager), 3); // 2 + 1

    // Binary operation: 3 + args complexity
    let add = manager.mk_add([x, five]);
    assert_eq!(term_complexity(add, &manager), 5); // 2 + 1 + 1 + 1 (len + base)
}

#[test]
fn test_compute_statistics() {
    let mut manager = TermManager::new();

    let x = manager.mk_var("x", manager.sorts.int_sort);
    let y = manager.mk_var("y", manager.sorts.int_sort);
    let five = manager.mk_int(5);

    // (x + y) + 5
    let add1 = manager.mk_add([x, y]);
    let expr = manager.mk_add([add1, five]);

    let stats = compute_statistics(expr, &manager);

    assert!(stats.num_variables >= 2); // x and y
    assert!(stats.num_constants >= 1); // 5
    assert!(stats.unique_subterms > 0);
    assert!(stats.depth > 0);
}

#[test]
fn test_find_terms() {
    let mut manager = TermManager::new();

    let x = manager.mk_var("x", manager.sorts.int_sort);
    let five = manager.mk_int(5);
    let ten = manager.mk_int(10);

    let mul = manager.mk_mul([five, ten]);
    let expr = manager.mk_add([x, mul]);

    // Find all integer constants
    let constants = find_terms(expr, &manager, |id, mgr| {
        mgr.get(id)
            .map(|t| matches!(t.kind, TermKind::IntConst(_)))
            .unwrap_or(false)
    });

    assert!(constants.len() >= 2); // Should find 5 and 10
}

#[test]
fn test_count_operations() {
    let mut manager = TermManager::new();

    let x = manager.mk_var("x", manager.sorts.int_sort);
    let y = manager.mk_var("y", manager.sorts.int_sort);

    // (x + y) * (x + y)
    let add = manager.mk_add([x, y]);
    let expr = manager.mk_mul([add, add]);

    // Count additions
    let add_count = count_operations(expr, &manager, |kind| matches!(kind, TermKind::Add(_)));

    assert_eq!(add_count, 1); // Only one unique Add node (shared)
}

#[test]
fn test_flatten_and() {
    let mut manager = TermManager::new();

    let a = manager.mk_var("a", manager.sorts.bool_sort);
    let b = manager.mk_var("b", manager.sorts.bool_sort);
    let c = manager.mk_var("c", manager.sorts.bool_sort);

    // Create (and (and a b) c)
    let inner = manager.mk_and([a, b]);
    let nested = manager.mk_and([inner, c]);

    // Flatten should give us (and a b c)
    let flattened = flatten_associative(nested, &mut manager);

    // Verify it's an And with 3 arguments
    if let Some(term) = manager.get(flattened) {
        if let TermKind::And(args) = &term.kind {
            assert_eq!(args.len(), 3);
            assert!(args.contains(&a));
            assert!(args.contains(&b));
            assert!(args.contains(&c));
        } else {
            panic!("Expected And term");
        }
    }
}

#[test]
fn test_flatten_or() {
    let mut manager = TermManager::new();

    let a = manager.mk_var("a", manager.sorts.bool_sort);
    let b = manager.mk_var("b", manager.sorts.bool_sort);
    let c = manager.mk_var("c", manager.sorts.bool_sort);
    let d = manager.mk_var("d", manager.sorts.bool_sort);

    // Create (or (or a b) (or c d))
    let left = manager.mk_or([a, b]);
    let right = manager.mk_or([c, d]);
    let nested = manager.mk_or([left, right]);

    // Flatten should give us (or a b c d)
    let flattened = flatten_associative(nested, &mut manager);

    // Verify it's an Or with 4 arguments
    if let Some(term) = manager.get(flattened) {
        if let TermKind::Or(args) = &term.kind {
            assert_eq!(args.len(), 4);
        } else {
            panic!("Expected Or term");
        }
    }
}

#[test]
fn test_flatten_add() {
    let mut manager = TermManager::new();

    let x = manager.mk_var("x", manager.sorts.int_sort);
    let one = manager.mk_int(1);
    let two = manager.mk_int(2);
    let three = manager.mk_int(3);

    // Create (+ (+ x 1) (+ 2 3))
    let left = manager.mk_add([x, one]);
    let right = manager.mk_add([two, three]);
    let nested = manager.mk_add([left, right]);

    // Flatten should give us (+ x 1 2 3)
    let flattened = flatten_associative(nested, &mut manager);

    // Verify it's an Add with 4 arguments
    if let Some(term) = manager.get(flattened) {
        if let TermKind::Add(args) = &term.kind {
            assert_eq!(args.len(), 4);
        } else {
            panic!("Expected Add term");
        }
    }
}

#[test]
fn test_flatten_mul() {
    let mut manager = TermManager::new();

    let x = manager.mk_var("x", manager.sorts.int_sort);
    let y = manager.mk_var("y", manager.sorts.int_sort);
    let two = manager.mk_int(2);

    // Create (* (* x 2) y)
    let inner = manager.mk_mul([x, two]);
    let nested = manager.mk_mul([inner, y]);

    // Flatten should give us (* x 2 y)
    let flattened = flatten_associative(nested, &mut manager);

    // Verify it's a Mul with 3 arguments
    if let Some(term) = manager.get(flattened) {
        if let TermKind::Mul(args) = &term.kind {
            assert_eq!(args.len(), 3);
        } else {
            panic!("Expected Mul term");
        }
    }
}

#[test]
fn test_flatten_deeply_nested() {
    let mut manager = TermManager::new();

    let a = manager.mk_var("a", manager.sorts.bool_sort);
    let b = manager.mk_var("b", manager.sorts.bool_sort);
    let c = manager.mk_var("c", manager.sorts.bool_sort);
    let d = manager.mk_var("d", manager.sorts.bool_sort);

    // Create (and (and (and a b) c) d)
    let inner1 = manager.mk_and([a, b]);
    let inner2 = manager.mk_and([inner1, c]);
    let nested = manager.mk_and([inner2, d]);

    // Flatten should give us (and a b c d)
    let flattened = flatten_associative(nested, &mut manager);

    // Verify it's an And with 4 arguments
    if let Some(term) = manager.get(flattened) {
        if let TermKind::And(args) = &term.kind {
            assert_eq!(args.len(), 4);
        } else {
            panic!("Expected And term");
        }
    }
}

#[test]
fn test_flatten_mixed_operations() {
    let mut manager = TermManager::new();

    let a = manager.mk_var("a", manager.sorts.bool_sort);
    let b = manager.mk_var("b", manager.sorts.bool_sort);
    let c = manager.mk_var("c", manager.sorts.bool_sort);

    // Create (and (or a b) c) - should only flatten the And, not the Or
    let or_term = manager.mk_or([a, b]);
    let and_term = manager.mk_and([or_term, c]);

    let flattened = flatten_associative(and_term, &mut manager);

    // Should be (and (or a b) c) with 2 And arguments
    if let Some(term) = manager.get(flattened) {
        if let TermKind::And(args) = &term.kind {
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected And term");
        }
    }
}

#[test]
fn test_flatten_already_flat() {
    let mut manager = TermManager::new();

    let a = manager.mk_var("a", manager.sorts.bool_sort);
    let b = manager.mk_var("b", manager.sorts.bool_sort);
    let c = manager.mk_var("c", manager.sorts.bool_sort);

    // Create (and a b c) - already flat
    let flat_term = manager.mk_and([a, b, c]);

    let flattened = flatten_associative(flat_term, &mut manager);

    // Should remain the same
    assert_eq!(flat_term, flattened);
}

// =====================================================================
// Regression tests for the recursive -> iterative conversion.
//
// Every walker in this module used to recurse once per term-nesting level
// with no depth guard, so a sufficiently deep term (constructible directly
// through `TermManager`'s builder API -- no parser or its `MAX_PARSE_DEPTH`
// involved) crashed the whole process with `fatal runtime error: stack
// overflow` rather than returning a value. Each test below builds its term
// with a plain `for` loop -- a recursive helper would overflow before the
// walk under test even started -- and runs the walk on a thread with a
// deliberately small (128 KiB) stack, the size a non-main / embedder worker
// thread typically gets. A stack overflow aborts the process rather than
// unwinding, so "the call returned at all" *is* part of the assertion; every
// test also checks the returned value is correct, not merely that one came
// back.
// =====================================================================

/// Stack size every deep test runs under: the ~1 MiB a non-main thread gets
/// by default on most platforms, and far less than a libtest thread's.
const SMALL_STACK: usize = 1 << 17;

/// A depth well past anything a native-stack recursion could survive.
const DEEP: usize = 12_500;

/// Run `body` on a thread with a deliberately small stack and return its
/// value. A stack overflow inside `body` aborts the process rather than
/// unwinding, so this helper cannot turn one into a test failure -- that is
/// the point: the test binary itself would abort, which is a loud, visible
/// signal.
fn on_small_stack<T, F>(body: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(SMALL_STACK)
        .spawn(body)
        .expect("spawn small-stack thread")
        .join()
        .expect("small-stack thread panicked")
}

/// Build `depth` levels of `(+ ... (+ (+ leaf v_0) v_1) ... v_{depth-1})`,
/// interned raw (not through `mk_add`) so the result is a genuinely nested
/// chain rather than the single flat n-ary node `mk_add` would fold repeated
/// calls into. Returns the deepest term and the leaf-most variable.
fn deep_add_chain(manager: &mut TermManager, depth: usize, leaf_name: &str) -> (TermId, TermId) {
    let int_sort = manager.sorts.int_sort;
    let leaf = manager.mk_var(leaf_name, int_sort);
    let mut term = leaf;
    for i in 0..depth {
        let v = manager.mk_var(&format!("v{i}"), int_sort);
        let args: SmallVec<[TermId; 4]> = smallvec![term, v];
        term = manager.intern_term(TermKind::Add(args), int_sort);
    }
    (term, leaf)
}

/// Like `deep_add_chain`, but every leaf is a distinct integer constant, so
/// the resulting term is ground.
fn deep_ground_add_chain(manager: &mut TermManager, depth: usize) -> TermId {
    let int_sort = manager.sorts.int_sort;
    let mut term = manager.mk_int(BigInt::from(0));
    for i in 0..depth {
        let v = manager.mk_int(BigInt::from(i as i64));
        let args: SmallVec<[TermId; 4]> = smallvec![term, v];
        term = manager.intern_term(TermKind::Add(args), int_sort);
    }
    term
}

#[test]
fn test_deep_add_chain_structural_hash_on_small_stack() {
    let (hash_a, hash_b) = on_small_stack(|| {
        let mut manager = TermManager::new();
        let (term_a, _) = deep_add_chain(&mut manager, DEEP, "leaf");
        let hash_a = structural_hash(term_a, &manager);
        // Hashing the same term again must be deterministic.
        let hash_a_again = structural_hash(term_a, &manager);
        assert_eq!(hash_a, hash_a_again);

        // A structurally distinct chain (different leaf name) must hash
        // differently with overwhelming probability.
        let (term_b, _) = deep_add_chain(&mut manager, DEEP, "different_leaf");
        let hash_b = structural_hash(term_b, &manager);
        (hash_a, hash_b)
    });
    assert_ne!(hash_a, hash_b);
}

#[test]
fn test_deep_add_chain_is_ground_on_small_stack() {
    let (non_ground, ground) = on_small_stack(|| {
        let mut manager = TermManager::new();
        let (var_chain, _) = deep_add_chain(&mut manager, DEEP, "leaf");
        let ground_chain = deep_ground_add_chain(&mut manager, DEEP);
        (
            is_ground(var_chain, &manager),
            is_ground(ground_chain, &manager),
        )
    });
    assert!(
        !non_ground,
        "a chain rooted in a free variable is not ground"
    );
    assert!(
        ground,
        "a chain built entirely from integer constants is ground"
    );
}

#[test]
fn test_deep_add_chain_term_complexity_on_small_stack() {
    let complexity = on_small_stack(|| {
        let mut manager = TermManager::new();
        let (term, _) = deep_add_chain(&mut manager, DEEP, "leaf");
        term_complexity(term, &manager)
    });
    // Each wrap is `Add([prev, fresh_var])`: weight = args.len() + 1 +
    // sum(children) = 2 + 1 + (complexity(prev) + 1) = complexity(prev) + 4.
    // The leaf itself has complexity 1, so after `DEEP` wraps:
    // complexity = 1 + 4 * DEEP.
    let expected = 1 + 4 * DEEP;
    assert_eq!(complexity, expected);
}

#[test]
fn test_deep_two_chains_structurally_equal_differ_at_bottom_on_small_stack() {
    // Two chains, identical at every level except the leaf reached only
    // after walking all the way down -- this exercises the lockstep pair
    // stack without an early exit anywhere above the very last frame.
    let (equal_to_self, differ_at_bottom) = on_small_stack(|| {
        let mut manager = TermManager::new();
        let (term_a, _) = deep_add_chain(&mut manager, DEEP, "same_leaf");
        let (term_a_again, _) = deep_add_chain(&mut manager, DEEP, "same_leaf");
        let (term_b, _) = deep_add_chain(&mut manager, DEEP, "different_leaf");
        (
            structurally_equal(term_a, term_a_again, &manager),
            structurally_equal(term_a, term_b, &manager),
        )
    });
    assert!(
        equal_to_self,
        "two independently-built but identical deep chains must compare equal"
    );
    assert!(
        !differ_at_bottom,
        "chains differing only at the deepest leaf must not compare equal"
    );
}

#[test]
fn test_deep_two_chains_alpha_equivalent_differ_at_bottom_on_small_stack() {
    let (equal_to_self, differ_at_bottom) = on_small_stack(|| {
        let mut manager = TermManager::new();
        let (term_a, _) = deep_add_chain(&mut manager, DEEP, "same_leaf");
        let (term_a_again, _) = deep_add_chain(&mut manager, DEEP, "same_leaf");
        let (term_b, _) = deep_add_chain(&mut manager, DEEP, "different_leaf");
        (
            alpha_equivalent(term_a, term_a_again, &manager),
            alpha_equivalent(term_a, term_b, &manager),
        )
    });
    assert!(equal_to_self);
    assert!(!differ_at_bottom);
}

/// `flatten_associative` rebuilds bottom-up and splices a growing flat
/// argument list at every level, which is O(depth^2) total work for a chain
/// like this -- a pre-existing property of the flattening *algorithm*
/// itself (the original recursive version does the same splicing), not a
/// regression introduced by making the walk iterative. `DEEP` (12_500)
/// would make the O(n^2) term dominate the test run, so this uses a smaller
/// depth that still comfortably exceeds any native call stack (the original
/// recursive form overflowed a 1 MiB stack within a few thousand levels for
/// much cheaper walks than this one -- see `model/evaluator/tests.rs`).
const FLATTEN_DEEP: usize = 4_000;

#[test]
fn test_deep_add_chain_flatten_associative_on_small_stack() {
    let arg_count = on_small_stack(|| {
        let mut manager = TermManager::new();
        let (term, _) = deep_add_chain(&mut manager, FLATTEN_DEEP, "leaf");
        let flattened = flatten_associative(term, &mut manager);
        match manager.get(flattened).map(|t| t.kind.clone()) {
            Some(TermKind::Add(args)) => args.len(),
            other => panic!("expected a flat Add, got {other:?}"),
        }
    });
    // `leaf` plus one fresh variable per wrap.
    assert_eq!(arg_count, FLATTEN_DEEP + 1);
}

// ---------------------------------------------------------------------
// Shallow pinned-value tests: a mixed term touching binders, Ite, n-ary
// Add, Store/Select and Match, so the conversion cannot silently change any
// answer for the kinds a purely-Add-shaped deep chain never exercises.
// ---------------------------------------------------------------------

/// A term exercising `Add` (n-ary), `Lt`, `Store`, `Select`, `Ite`, `Forall`
/// and `Let` in one structure.
fn build_mixed_term(manager: &mut TermManager) -> TermId {
    let int_sort = manager.sorts.int_sort;
    let array_sort = manager.sorts.array(int_sort, int_sort);

    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let arr = manager.mk_var("arr", array_sort);
    let five = manager.mk_int(5);

    let sum = manager.mk_add([x, y, five]);
    let cond = manager.mk_lt(x, y);
    let stored = manager.mk_store(arr, x, sum);
    let selected = manager.mk_select(stored, y);
    let ite = manager.mk_ite(cond, selected, sum);

    let w = manager.mk_var("w", int_sort);
    let forall_body = manager.mk_ge(w, x);
    let forall = manager.mk_forall([("w", int_sort)], forall_body);

    let z = manager.mk_var("z", int_sort);
    let let_body = manager.mk_eq(z, five);
    let let_term = manager.mk_let([("z", ite)], let_body);

    manager.mk_and([let_term, forall])
}

#[test]
fn test_mixed_term_is_ground_and_complexity_are_internally_consistent() {
    let mut manager = TermManager::new();
    let term = build_mixed_term(&mut manager);

    // Contains free variables x/y/arr, so not ground.
    assert!(!is_ground(term, &manager));
    // Complexity must be deterministic across repeated calls.
    let c1 = term_complexity(term, &manager);
    let c2 = term_complexity(term, &manager);
    assert_eq!(c1, c2);
    assert!(c1 > 1);
}

#[test]
fn test_mixed_term_structurally_equal_and_hash_are_reflexive() {
    let mut manager = TermManager::new();
    let term_a = build_mixed_term(&mut manager);
    let term_b = build_mixed_term(&mut manager);

    // Built the same way twice: hash-consing means these should even share
    // an id, but structural equality must hold regardless.
    assert!(structurally_equal(term_a, term_b, &manager));
    assert_eq!(
        structural_hash(term_a, &manager),
        structural_hash(term_b, &manager)
    );
    assert!(alpha_equivalent(term_a, term_b, &manager));
}

#[test]
fn test_mixed_term_flatten_associative_only_flattens_add_and_and() {
    let mut manager = TermManager::new();
    let term = build_mixed_term(&mut manager);
    let flattened = flatten_associative(term, &mut manager);

    // Top-level node is `And([let_term, forall])`: already flat with 2 args,
    // so flattening must not change its arity.
    match manager.get(flattened).map(|t| t.kind.clone()) {
        Some(TermKind::And(args)) => assert_eq!(args.len(), 2),
        other => panic!("expected a flat And, got {other:?}"),
    }
}

/// A `Match` term (algebraic datatype pattern match), built directly via
/// `intern_term` since this module exposes no `mk_match` constructor.
fn build_match_term(manager: &mut TermManager) -> TermId {
    let int_sort = manager.sorts.int_sort;
    let scrutinee = manager.mk_var("d", int_sort);
    let cons = manager.intern_str("cons");
    let nil = manager.intern_str("nil");
    let one = manager.mk_int(1);
    let zero = manager.mk_int(0);

    let cases: SmallVec<[MatchCase; 4]> = smallvec![
        MatchCase {
            constructor: Some(cons),
            bindings: SmallVec::new(),
            body: one,
        },
        MatchCase {
            constructor: Some(nil),
            bindings: SmallVec::new(),
            body: zero,
        },
    ];
    manager.intern_term(TermKind::Match { scrutinee, cases }, int_sort)
}

#[test]
fn test_match_term_is_handled_by_every_walker() {
    let mut manager = TermManager::new();
    let term = build_match_term(&mut manager);
    let term_again = build_match_term(&mut manager);

    // scrutinee (`d`) is free, so the term is not ground.
    assert!(!is_ground(term, &manager));
    // 5 + complexity(scrutinee = 1) + complexity(case bodies: 1 each).
    assert_eq!(term_complexity(term, &manager), 8);
    assert_eq!(
        structural_hash(term, &manager),
        structural_hash(term_again, &manager)
    );
    assert!(structurally_equal(term, term_again, &manager));

    // flatten_associative doesn't rewrite `Match` at all -- it isn't one of
    // the nine kinds this function recurses into -- so the term must come
    // back completely unchanged (same id).
    let flattened = flatten_associative(term, &mut manager);
    assert_eq!(flattened, term);
}

// =====================================================================
// Pinned `structural_hash` values.
//
// These literals were captured by a one-time A/B comparison against the
// *original* recursive `structural_hash_impl` (reconstructed verbatim from
// the pre-conversion `ast/utils.rs`, byte for byte the same match arms just
// called directly instead of through the `HashTask` stack) for each of these
// same terms; the iterative version in `hash.rs` matched it exactly for all
// eight. Pinning the resulting values here means any *future* change that
// silently alters the hash -- a reordered match arm, a dropped field, a
// different traversal order -- fails this test immediately, without needing
// to keep a whole second implementation of the algorithm around permanently
// for comparison.
#[test]
fn test_structural_hash_is_pinned() {
    let mut manager = TermManager::new();

    let five = manager.mk_int(5);
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let y = manager.mk_var("y", manager.sorts.int_sort);
    let add = manager.mk_add([x, y, five]);
    let true_term = manager.mk_true();
    let not_term = manager.mk_not(true_term);
    let lt = manager.mk_lt(x, y);
    let ite = manager.mk_ite(lt, add, five);
    let mixed = build_mixed_term(&mut manager);
    let matched = build_match_term(&mut manager);
    let (deep500, _) = deep_add_chain(&mut manager, 500, "leaf");

    let cases: [(&str, TermId, u64); 8] = [
        ("five", five, 0xe672_6b24_3366_bc0d),
        ("x", x, 0x310d_cc87_0957_f08b),
        ("add", add, 0xce22_9b07_edc5_2831),
        ("not_term", not_term, 0xa8b9_8aa7_17c4_d5eb),
        ("ite", ite, 0xf102_89ed_feac_017c),
        ("mixed", mixed, 0x2b3b_6df5_090b_4ef6),
        ("matched", matched, 0x02e2_a2f6_deb5_5d6c),
        ("deep500", deep500, 0xc477_fc5b_c1a0_45f8),
    ];
    for (name, term, expected) in cases {
        let actual = structural_hash(term, &manager);
        assert_eq!(
            actual, expected,
            "structural_hash(`{name}`) drifted: expected 0x{expected:016x}, got 0x{actual:016x}"
        );
    }
}

// =====================================================================
// Regression tests for the two `equality.rs` bugs fixed alongside this
// module split:
//
// (1) `alpha_equivalent` never inserted anything into its bound-variable
//     `env`, so it could not recognize two quantifiers/lets differing only
//     by a bound variable's name as equivalent -- contrary to its own doc
//     example. Fixed by `AlphaEnv` in `equality/alpha.rs`.
// (2) `structurally_equal` (and, for floating-point/datatype kinds,
//     `alpha_equivalent` too) had no arm at all for `Forall`/`Exists`/`Let`,
//     any FP operator, any `Dt*` kind, or any `Str*` operator, silently
//     falling to `_ => false` -- indistinguishable from a genuine mismatch.
//     Fixed by making both functions' outer match exhaustive over
//     `TermKind` with no wildcard arm.
// =====================================================================

#[test]
fn test_alpha_equivalent_quantifier_doc_example() {
    // The exact example from `alpha_equivalent`'s own doc comment:
    // (forall ((x Int)) (> x 0)) and (forall ((y Int)) (> y 0)) must be
    // recognized as alpha-equivalent despite the different bound-variable
    // name. This is the one bug-1 assertion that fails against the
    // pre-fix code (which never populated `env`, so this returned `false`).
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let zero = manager.mk_int(0);

    let x = manager.mk_var("x", int_sort);
    let gt_x_zero = manager.mk_gt(x, zero);
    let lhs = manager.mk_forall([("x", int_sort)], gt_x_zero);

    let y = manager.mk_var("y", int_sort);
    let gt_y_zero = manager.mk_gt(y, zero);
    let rhs = manager.mk_forall([("y", int_sort)], gt_y_zero);

    assert_ne!(
        lhs, rhs,
        "different bound-variable names must not hash-cons to the same term"
    );
    assert!(
        alpha_equivalent(lhs, rhs, &manager),
        "(forall ((x Int)) (> x 0)) and (forall ((y Int)) (> y 0)) must be alpha-equivalent"
    );
}

#[test]
fn test_alpha_equivalent_different_arity_not_equivalent() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let zero = manager.mk_int(0);

    let x = manager.mk_var("x", int_sort);
    let gt_x_zero = manager.mk_gt(x, zero);
    let lhs = manager.mk_forall([("x", int_sort)], gt_x_zero);

    let y = manager.mk_var("y", int_sort);
    let gt_y_zero = manager.mk_gt(y, zero);
    let rhs = manager.mk_forall([("y", int_sort), ("z", int_sort)], gt_y_zero);

    assert!(
        !alpha_equivalent(lhs, rhs, &manager),
        "binders of different arity must not be alpha-equivalent"
    );
}

#[test]
fn test_alpha_equivalent_different_sort_not_equivalent() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let real_sort = manager.sorts.real_sort;

    // `manager.mk_eq(v, v)` short-circuits to the `True` constant for
    // identical operands, so the body ends up trivially `True` on both
    // sides here -- irrelevant to this test, since the rejection this test
    // pins is supposed to come from the *sort* mismatch in `vars` alone,
    // checked before either body is ever compared.
    let x = manager.mk_var("x", int_sort);
    let body_lhs = manager.mk_eq(x, x);
    let lhs = manager.mk_forall([("x", int_sort)], body_lhs);

    let y = manager.mk_var("y", real_sort);
    let body_rhs = manager.mk_eq(y, y);
    let rhs = manager.mk_forall([("y", real_sort)], body_rhs);

    assert!(
        !alpha_equivalent(lhs, rhs, &manager),
        "(forall ((x Int)) ...) and (forall ((y Real)) ...) must not be alpha-equivalent, \
         even with matching arity and shape"
    );
}

#[test]
fn test_alpha_equivalent_different_body_constant_not_equivalent() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let zero = manager.mk_int(0);
    let one = manager.mk_int(1);

    let x = manager.mk_var("x", int_sort);
    let gt_x_zero = manager.mk_gt(x, zero);
    let lhs = manager.mk_forall([("x", int_sort)], gt_x_zero);

    let y = manager.mk_var("y", int_sort);
    let gt_y_one = manager.mk_gt(y, one);
    let rhs = manager.mk_forall([("y", int_sort)], gt_y_one);

    assert!(
        !alpha_equivalent(lhs, rhs, &manager),
        "(forall ((x Int)) (> x 0)) and (forall ((y Int)) (> y 1)) must not be alpha-equivalent"
    );
}

#[test]
fn test_alpha_equivalent_mapping_must_be_bijective() {
    // If the bound-variable correspondence were not enforced bijectively,
    // both `x` and `q` (two genuinely distinct lhs variables) could appear
    // to correspond to the single rhs variable `y` (bound twice, the inner
    // occurrence shadowing the outer one) -- wrongly equating
    // `(forall x (forall q (= x q)))` (asserting two *different* bound
    // variables are equal) with `(forall y (forall y (= y y)))` (a
    // tautology, since shadowing makes both occurrences the same
    // variable). These are not alpha-equivalent: the lhs is not a
    // tautology, the rhs is.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let q = manager.mk_var("q", int_sort);
    let eq_xq = manager.mk_eq(x, q);
    let inner_lhs = manager.mk_forall([("q", int_sort)], eq_xq);
    let lhs = manager.mk_forall([("x", int_sort)], inner_lhs);

    // `manager.mk_eq(y, y)` would short-circuit to the `True` constant (its
    // smart-constructor simplification for identical operands), which would
    // make this test pass for the wrong reason (a `True`/`Eq` kind mismatch
    // against `eq_xq`, rather than exercising bijectivity at all). Build the
    // literal `Eq(y, y)` node directly via `intern_term` to bypass that.
    let y = manager.mk_var("y", int_sort);
    let bool_sort = manager.sorts.bool_sort;
    let eq_yy = manager.intern_term(TermKind::Eq(y, y), bool_sort);
    let inner_rhs = manager.mk_forall([("y", int_sort)], eq_yy);
    let rhs = manager.mk_forall([("y", int_sort)], inner_rhs);

    assert!(
        !alpha_equivalent(lhs, rhs, &manager),
        "a non-bijective bound-variable mapping must not make \
         (forall x (forall q (= x q))) equivalent to (forall y (forall y (= y y)))"
    );
}

#[test]
fn test_alpha_equivalent_visited_cache_keyed_on_environment() {
    // Regression test for the `visited` cycle-guard's soundness once results
    // depend on `env` (see `equality/alpha.rs`'s module docs). Both `forall`s
    // on each side share the *exact same* body subterm verbatim (`z`/`q` are
    // unused, so the body doesn't depend on which outer variable is bound):
    //
    //   lhs: (and (forall ((x Int)) (= x w)) (forall ((z Int)) (= x w)))
    //   rhs: (and (forall ((p Int)) (= p w)) (forall ((q Int)) (= p w)))
    //
    // The first conjuncts bind `x <-> p` and correctly find `(= x w)`
    // equivalent to `(= p w)`. The second conjuncts push *the exact same*
    // `(TermId, TermId)` pair (the shared `(= x w)`/`(= p w)` subterms) but
    // under the unrelated `z <-> q` correspondence, where `x` and `p` are
    // free and must NOT be treated as corresponding (they're different
    // names). A `visited` set keyed only on the id pair would have already
    // marked that pair "equal" while processing the first conjunct and
    // would wrongly skip re-checking it here, making the whole call return
    // `true` for two formulas that depend on different free variables --
    // this test fails against exactly that (env-unaware) implementation.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let w = manager.mk_var("w", int_sort);

    let x = manager.mk_var("x", int_sort);
    let eq_x_w = manager.mk_eq(x, w);
    let forall1_lhs = manager.mk_forall([("x", int_sort)], eq_x_w);
    let forall2_lhs = manager.mk_forall([("z", int_sort)], eq_x_w);

    let p = manager.mk_var("p", int_sort);
    let eq_p_w = manager.mk_eq(p, w);
    let forall1_rhs = manager.mk_forall([("p", int_sort)], eq_p_w);
    let forall2_rhs = manager.mk_forall([("q", int_sort)], eq_p_w);

    // Listed with the `z <-> q` (irrelevant-context) conjunct *first* and
    // the `x <-> p` (relevant-context) conjunct *second*: the walk's LIFO
    // stack pops n-ary children in reverse push order, so the second
    // conjunct is processed first, establishes the `(eq_x_w, eq_p_w)`
    // pair's *correct* answer under `x <-> p`, and only then does the first
    // conjunct revisit the exact same pair under the unrelated `z <-> q`.
    // Listing them the other way around would let the irrelevant-context
    // conjunct fail on its own before the relevant one ever runs, which
    // would prove nothing about the cache.
    let lhs = manager.mk_and([forall2_lhs, forall1_lhs]);
    let rhs = manager.mk_and([forall2_rhs, forall1_rhs]);

    assert!(
        !alpha_equivalent(lhs, rhs, &manager),
        "a shared subterm pair revisited under a different, unrelated \
         environment must be independently re-checked, not skipped as \
         \"already equal\" from an earlier, differently-scoped visit"
    );
}

#[test]
fn test_alpha_equivalent_sibling_scopes_are_independent() {
    // The mirror, positive case: two sibling `forall`s reusing the *same*
    // bound name `x` on the lhs must still be free to correspond to
    // *different* rhs names in each sibling (`p` in the first, `q` in the
    // second), since each binder's scope is independent.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let zero = manager.mk_int(0);

    let x1 = manager.mk_var("x", int_sort);
    let gt_x1_zero = manager.mk_gt(x1, zero);
    let forall1_lhs = manager.mk_forall([("x", int_sort)], gt_x1_zero);
    let x2 = manager.mk_var("x", int_sort);
    let lt_x2_zero = manager.mk_lt(x2, zero);
    let forall2_lhs = manager.mk_forall([("x", int_sort)], lt_x2_zero);
    let lhs = manager.mk_and([forall1_lhs, forall2_lhs]);

    let p = manager.mk_var("p", int_sort);
    let gt_p_zero = manager.mk_gt(p, zero);
    let forall1_rhs = manager.mk_forall([("p", int_sort)], gt_p_zero);
    let q = manager.mk_var("q", int_sort);
    let lt_q_zero = manager.mk_lt(q, zero);
    let forall2_rhs = manager.mk_forall([("q", int_sort)], lt_q_zero);
    let rhs = manager.mk_and([forall1_rhs, forall2_rhs]);

    assert!(
        alpha_equivalent(lhs, rhs, &manager),
        "independent sibling binders may map the same lhs name to different rhs names"
    );
}

#[test]
fn test_alpha_equivalent_shadowing_both_directions() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let zero = manager.mk_int(0);

    // lhs shadows (inner `x` rebinds outer `x`); rhs uses two distinct names.
    let inner_x = manager.mk_var("x", int_sort);
    let gt_inner_x_zero = manager.mk_gt(inner_x, zero);
    let inner_body_lhs = manager.mk_forall([("x", int_sort)], gt_inner_x_zero);
    let lhs_shadowed = manager.mk_forall([("x", int_sort)], inner_body_lhs);

    let q = manager.mk_var("q", int_sort);
    let gt_q_zero = manager.mk_gt(q, zero);
    let inner_body_rhs = manager.mk_forall([("q", int_sort)], gt_q_zero);
    let rhs_distinct = manager.mk_forall([("p", int_sort)], inner_body_rhs);

    assert!(
        alpha_equivalent(lhs_shadowed, rhs_distinct, &manager),
        "an inner binder shadowing the outer name (lhs) must still correspond correctly \
         to two distinct rhs names, since only the innermost binding is ever observable"
    );

    // Mirror: rhs shadows, lhs uses two distinct names.
    assert!(
        alpha_equivalent(rhs_distinct, lhs_shadowed, &manager),
        "shadowing must be handled symmetrically regardless of which side does it"
    );
}

#[test]
fn test_alpha_equivalent_exists_and_let_rename_bound_variables() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let zero = manager.mk_int(0);

    let x = manager.mk_var("x", int_sort);
    let gt_x_zero = manager.mk_gt(x, zero);
    let exists_lhs = manager.mk_exists([("x", int_sort)], gt_x_zero);
    let y = manager.mk_var("y", int_sort);
    let gt_y_zero = manager.mk_gt(y, zero);
    let exists_rhs = manager.mk_exists([("y", int_sort)], gt_y_zero);
    assert!(alpha_equivalent(exists_lhs, exists_rhs, &manager));

    let five = manager.mk_int(5);
    let z = manager.mk_var("z", int_sort);
    let gt_z_zero = manager.mk_gt(z, zero);
    let let_lhs = manager.mk_let([("z", five)], gt_z_zero);
    let w = manager.mk_var("w", int_sort);
    let gt_w_zero = manager.mk_gt(w, zero);
    let let_rhs = manager.mk_let([("w", five)], gt_w_zero);
    assert!(
        alpha_equivalent(let_lhs, let_rhs, &manager),
        "let-bound names may differ as long as the bound values and body shape match, \
         with the body's use of its own bound variable correctly renamed"
    );

    // But structurally_equal permits *no* renaming at all, even for `let`.
    assert!(
        !structurally_equal(let_lhs, let_rhs, &manager),
        "structurally_equal must require identical let-bound names, not just identical values"
    );
}

#[test]
fn test_structurally_equal_quantifier_requires_identical_names() {
    // The doc-example pair *is* alpha-equivalent (see
    // `test_alpha_equivalent_quantifier_doc_example`) but must NOT be
    // structurally equal: structural equality permits no renaming.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let zero = manager.mk_int(0);

    let x = manager.mk_var("x", int_sort);
    let gt_x_zero = manager.mk_gt(x, zero);
    let lhs = manager.mk_forall([("x", int_sort)], gt_x_zero);
    let y = manager.mk_var("y", int_sort);
    let gt_y_zero = manager.mk_gt(y, zero);
    let rhs = manager.mk_forall([("y", int_sort)], gt_y_zero);

    assert!(!structurally_equal(lhs, rhs, &manager));

    // Identical names must, of course, compare equal (and reach the new
    // `Forall` arm to do so, not just an id shortcut, since it's rebuilt
    // via a fresh `intern_term` call below rather than `mk_forall`).
    let x_again = manager.mk_var("x", int_sort);
    let gt_x_again_zero = manager.mk_gt(x_again, zero);
    let rhs_same_name = manager.mk_forall([("x", int_sort)], gt_x_again_zero);
    assert_eq!(
        lhs, rhs_same_name,
        "identical quantifiers must hash-cons to the same id"
    );
    assert!(structurally_equal(lhs, rhs_same_name, &manager));
}

#[test]
fn test_structurally_equal_and_alpha_equivalent_ignore_patterns() {
    // `patterns` (quantifier trigger hints) are deliberately not part of
    // either comparison -- see `equality/mod.rs`'s module docs. Building the
    // same quantifier with and without an explicit pattern list produces
    // two genuinely different `TermId`s (patterns are part of the
    // interning key), which must still compare equal both ways.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let fx = manager.mk_apply("f", [x], int_sort);
    let zero = manager.mk_int(0);
    let body = manager.mk_gt(fx, zero);

    let no_patterns = manager.mk_forall([("x", int_sort)], body);
    let with_patterns = manager.mk_forall_with_patterns([("x", int_sort)], body, [[fx]]);

    assert_ne!(
        no_patterns, with_patterns,
        "differing patterns must produce genuinely different term ids"
    );
    assert!(
        structurally_equal(no_patterns, with_patterns, &manager),
        "structurally_equal must ignore quantifier trigger patterns"
    );
    assert!(
        alpha_equivalent(no_patterns, with_patterns, &manager),
        "alpha_equivalent must ignore quantifier trigger patterns"
    );
}

#[test]
fn test_structurally_equal_and_alpha_equivalent_distinguish_forall_from_exists() {
    // `Forall` and `Exists` share the exact same field shape and both
    // produce `Bool` sort, so the generic sort-mismatch fast path cannot
    // distinguish them -- only the `core::mem::discriminant` guard on the
    // grouped match arm does.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let zero = manager.mk_int(0);
    let body = manager.mk_gt(x, zero);

    let forall_t = manager.mk_forall([("x", int_sort)], body);
    let exists_t = manager.mk_exists([("x", int_sort)], body);

    assert!(!structurally_equal(forall_t, exists_t, &manager));
    assert!(!alpha_equivalent(forall_t, exists_t, &manager));
}

#[test]
fn test_fp_terms_structural_and_alpha_equal_and_distinguish_operators() {
    let mut manager = TermManager::new();
    let eb = 8u32;
    let sb = 24u32;

    let a = manager.mk_fp_lit(false, 5, 10, eb, sb);
    let a_again = manager.mk_fp_lit(false, 5, 10, eb, sb);
    assert_eq!(
        a, a_again,
        "identical FP literals must hash-cons to the same id"
    );
    assert!(structurally_equal(a, a_again, &manager));
    assert!(alpha_equivalent(a, a_again, &manager));

    let differs_in_significand = manager.mk_fp_lit(false, 5, 11, eb, sb);
    assert!(!structurally_equal(a, differs_in_significand, &manager));
    assert!(!alpha_equivalent(a, differs_in_significand, &manager));

    // Same {eb, sb} (so the generic sort check cannot help), different
    // "special value" kind -- only the discriminant guard distinguishes.
    let nan = manager.mk_fp_nan(eb, sb);
    let pos_inf = manager.mk_fp_plus_infinity(eb, sb);
    assert!(!structurally_equal(nan, pos_inf, &manager));
    assert!(!alpha_equivalent(nan, pos_inf, &manager));

    // FpAbs vs FpNeg: same shape (one TermId operand), different operator.
    let abs_a = manager.mk_fp_abs(a);
    let neg_a = manager.mk_fp_neg(a);
    assert!(!structurally_equal(abs_a, neg_a, &manager));
    assert!(!alpha_equivalent(abs_a, neg_a, &manager));

    // FpSqrt vs FpRoundToIntegral: same shape (RoundingMode + TermId).
    let sqrt_a = manager.mk_fp_sqrt(RoundingMode::RNE, a);
    let round_a = manager.mk_fp_round_to_integral(RoundingMode::RNE, a);
    assert!(!structurally_equal(sqrt_a, round_a, &manager));
    assert!(!alpha_equivalent(sqrt_a, round_a, &manager));

    // FpAdd with different rounding modes must not compare equal.
    let b = manager.mk_fp_lit(false, 3, 7, eb, sb);
    let add_rne = manager.mk_fp_add(RoundingMode::RNE, a, b);
    let add_rtz = manager.mk_fp_add(RoundingMode::RTZ, a, b);
    assert!(!structurally_equal(add_rne, add_rtz, &manager));
    assert!(!alpha_equivalent(add_rne, add_rtz, &manager));

    // FpToFp vs SBVToFp: same {rm, arg, eb, sb} shape, different conversion.
    let to_fp = manager.mk_fp_to_fp(RoundingMode::RNE, a, eb, sb);
    let sbv_to_fp = manager.mk_sbv_to_fp(RoundingMode::RNE, a, eb, sb);
    assert!(!structurally_equal(to_fp, sbv_to_fp, &manager));
    assert!(!alpha_equivalent(to_fp, sbv_to_fp, &manager));

    // FpToSBV vs FpToUBV: same {rm, arg, width} shape, different conversion.
    let to_sbv = manager.mk_fp_to_sbv(RoundingMode::RNE, a, 32);
    let to_ubv = manager.mk_fp_to_ubv(RoundingMode::RNE, a, 32);
    assert!(!structurally_equal(to_sbv, to_ubv, &manager));
    assert!(!alpha_equivalent(to_sbv, to_ubv, &manager));
}

#[test]
fn test_datatype_terms_structural_and_alpha_equal_and_distinguish_operators() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let bool_sort = manager.sorts.bool_sort;
    let one = manager.mk_int(1);
    let two = manager.mk_int(2);

    let cons_a = manager.mk_dt_constructor("cons", [one, two], int_sort);
    let cons_a_again = manager.mk_dt_constructor("cons", [one, two], int_sort);
    assert_eq!(cons_a, cons_a_again);
    assert!(structurally_equal(cons_a, cons_a_again, &manager));
    assert!(alpha_equivalent(cons_a, cons_a_again, &manager));

    let nil_a: TermId = manager.mk_dt_constructor("nil", Vec::<TermId>::new(), int_sort);
    assert!(!structurally_equal(cons_a, nil_a, &manager));
    assert!(!alpha_equivalent(cons_a, nil_a, &manager));

    // DtTester vs DtSelector: force both to `Bool` sort and use the same
    // constructor/selector name and argument, so only the discriminant
    // guard -- not the generic sort check or a name mismatch -- can tell
    // "is-cons" apart from a (nonsensically bool-sorted) "cons" selector.
    let is_cons = manager.mk_dt_tester("cons", cons_a);
    let selector_as_bool = manager.mk_dt_selector("cons", cons_a, bool_sort);
    assert!(!structurally_equal(is_cons, selector_as_bool, &manager));
    assert!(!alpha_equivalent(is_cons, selector_as_bool, &manager));
}

/// Depth for nested-`forall` alpha-equivalence: `AlphaEnv::bind` clones a
/// `BTreeMap` holding one entry per enclosing binder, so this is O(depth)
/// per level and O(depth^2) total across the whole chain -- the same
/// caveat `flatten_associative`'s `FLATTEN_DEEP` documents for its own
/// splicing cost. `DEEP` (12_500) would make the O(n^2) term dominate the
/// test run, so this uses a smaller depth that still comfortably exceeds
/// any native call stack.
const ALPHA_QUANTIFIER_DEEP: usize = 3_000;

/// Build `depth` levels of nested `forall`s: the innermost binds `l0`/`r0`
/// and its body compares that *own* bound variable against `0` (lhs) /
/// `rhs_const` (rhs) -- exactly the doc-example shape -- while every level
/// above it (`l1..l{depth-1}` / `r1..r{depth-1}`) binds a fresh,
/// differently-named variable that the body never references. So the
/// entire chain is alpha-equivalent when `rhs_const == 0` (every level
/// renames cleanly, right down to the innermost comparison), and differs
/// only at the very bottom -- the deepest possible frame -- whenever
/// `rhs_const != 0`, with no other mismatch anywhere above it.
fn deep_nested_forall_chain(
    manager: &mut TermManager,
    depth: usize,
    rhs_const: i64,
) -> (TermId, TermId) {
    let int_sort = manager.sorts.int_sort;
    let zero = manager.mk_int(0);
    let rhs_bound = manager.mk_int(rhs_const);

    let l0 = manager.mk_var("l0", int_sort);
    let gt_l0_zero = manager.mk_gt(l0, zero);
    let mut lhs = manager.mk_forall([("l0", int_sort)], gt_l0_zero);

    let r0 = manager.mk_var("r0", int_sort);
    let gt_r0_const = manager.mk_gt(r0, rhs_bound);
    let mut rhs = manager.mk_forall([("r0", int_sort)], gt_r0_const);

    for i in 1..depth {
        let lhs_name = format!("l{i}");
        let rhs_name = format!("r{i}");
        lhs = manager.mk_forall([(lhs_name.as_str(), int_sort)], lhs);
        rhs = manager.mk_forall([(rhs_name.as_str(), int_sort)], rhs);
    }
    (lhs, rhs)
}

#[test]
fn test_deep_nested_forall_alpha_equivalent_on_small_stack() {
    let (equal_when_renamed, differ_only_at_leaf) = on_small_stack(|| {
        let mut manager = TermManager::new();
        let (lhs_equal, rhs_equal) =
            deep_nested_forall_chain(&mut manager, ALPHA_QUANTIFIER_DEEP, 0);
        let (lhs_differ, rhs_differ) =
            deep_nested_forall_chain(&mut manager, ALPHA_QUANTIFIER_DEEP, 1);

        (
            alpha_equivalent(lhs_equal, rhs_equal, &manager),
            alpha_equivalent(lhs_differ, rhs_differ, &manager),
        )
    });
    assert!(
        equal_when_renamed,
        "a deep chain of foralls differing only in bound-variable names at every \
         level (lhs) vs (rhs), right down to the innermost comparison, must be alpha-equivalent"
    );
    assert!(
        !differ_only_at_leaf,
        "a deep chain identical at every level except the innermost constant \
         (reached only after walking all the way down) must not be alpha-equivalent"
    );
}
