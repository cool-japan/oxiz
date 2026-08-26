//! Tests for the explicit-stack backtracking E-matcher.
//!
//! These pin the three properties the conversion away from
//! `match_recursive` had to preserve or fix: both `Eq` orientations are
//! still explored, bindings made along a failed alternative are undone, and
//! neither depth nor nested equalities can blow up the native stack or the
//! search time.
//!
//! `Eq` nodes are built with [`TermManager::intern_term`] rather than
//! `mk_eq` on purpose: `mk_eq` canonicalises its operand order by `TermId`,
//! which would leave the stored orientation at the mercy of allocation
//! order and make "this only matches after swapping" untestable.

use super::*;
use crate::ast::TermManager;

/// Build `Eq(lhs, rhs)` with the operand order exactly as given.
fn eq_ordered(manager: &mut TermManager, lhs: TermId, rhs: TermId) -> TermId {
    let bool_sort = manager.sorts.bool_sort;
    manager.intern_term(TermKind::Eq(lhs, rhs), bool_sort)
}

/// The bound-variable list a `Pattern` would carry for `names`, all of
/// integer sort.
fn bound_int_vars(manager: &mut TermManager, names: &[&str]) -> SmallVec<[(Spur, SortId); 2]> {
    let int_sort = manager.sorts.int_sort;
    names
        .iter()
        .map(|name| (manager.intern_str(name), int_sort))
        .collect()
}

/// Resolve a substitution to a `(name, term)` list for readable assertions.
fn resolved(manager: &TermManager, subst: &FxHashMap<Spur, TermId>) -> Vec<(String, TermId)> {
    let mut out: Vec<(String, TermId)> = subst
        .iter()
        .map(|(&name, &term)| (manager.resolve_str(name).to_string(), term))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

// ===========================================================================
// Eq orientations
// ===========================================================================

/// `Eq(x, k)` matches `Eq(k, m)` only in the swapped orientation, and the
/// `x := k` binding made while the first orientation failed must not survive
/// to block it.
///
/// The retired recursion threaded one `&mut` binding map through
/// `(a && b) || (c && d)` with no undo, so it reported *no match* here.
#[test]
fn eq_orientation_swap_after_undoing_a_failed_binding() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let k = manager.mk_var("k", int_sort);
    let m = manager.mk_var("m", int_sort);

    let pattern = eq_ordered(&mut manager, x, k);
    let ground = eq_ordered(&mut manager, k, m);

    let bound = bound_int_vars(&mut manager, &["x"]);
    let matcher = PatternMatcher::new();
    let subst = matcher
        .try_match_term(pattern, ground, &bound, &manager)
        .expect("Eq(x, k) matches Eq(k, m) with x := m after swapping");

    assert_eq!(resolved(&manager, &subst), vec![("x".to_string(), m)]);
}

/// A variable bound only while exploring a failed alternative must not
/// appear in the successful match's substitution.
///
/// `PatternMatcher::match_against` gates instantiation on "every bound
/// variable is assigned", so a leaked binding is not cosmetic: it lets the
/// quantifier be instantiated with a term the trigger never matched.
#[test]
fn failed_alternative_leaves_no_binding_behind() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let y = manager.mk_var("y", int_sort);
    let k = manager.mk_var("k", int_sort);
    let k2 = manager.mk_var("k2", int_sort);

    // h(Eq(y, k), g(k)) against h(Eq(k, k2), g(k)): the equality's first
    // orientation binds y := k and then fails on (k, k2).
    let pattern_eq = eq_ordered(&mut manager, y, k);
    let ground_eq = eq_ordered(&mut manager, k, k2);
    let pattern_g = manager.mk_apply("g", vec![k], int_sort);
    let pattern = manager.mk_apply("h", vec![pattern_eq, pattern_g], int_sort);
    let ground = manager.mk_apply("h", vec![ground_eq, pattern_g], int_sort);

    let bound = bound_int_vars(&mut manager, &["y"]);
    let matcher = PatternMatcher::new();
    let subst = matcher
        .try_match_term(pattern, ground, &bound, &manager)
        .expect("second orientation matches with y := k2");

    // y := k came from the failed orientation and must be gone.
    assert_eq!(resolved(&manager, &subst), vec![("y".to_string(), k2)]);
}

/// Backtracking must reach across *sibling arguments*: the first argument
/// admits two orientations, and only the one the second argument agrees
/// with is a match.
#[test]
fn backtracks_across_sibling_arguments() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let p = manager.mk_var("p", int_sort);
    let q = manager.mk_var("q", int_sort);

    // h(Eq(x, y), g(x)) vs h(Eq(p, q), g(q)).
    // Sibling 1 matches either way: {x:=p, y:=q} or {x:=q, y:=p}.
    // Sibling 2 forces x := q, so the first orientation must be retracted.
    let pattern_eq = eq_ordered(&mut manager, x, y);
    let ground_eq = eq_ordered(&mut manager, p, q);
    let pattern_g = manager.mk_apply("g", vec![x], int_sort);
    let ground_g = manager.mk_apply("g", vec![q], int_sort);
    let pattern = manager.mk_apply("h", vec![pattern_eq, pattern_g], int_sort);
    let ground = manager.mk_apply("h", vec![ground_eq, ground_g], int_sort);

    let bound = bound_int_vars(&mut manager, &["x", "y"]);
    let matcher = PatternMatcher::new();
    let subst = matcher
        .try_match_term(pattern, ground, &bound, &manager)
        .expect("the swapped orientation of sibling 1 agrees with sibling 2");

    assert_eq!(
        resolved(&manager, &subst),
        vec![("x".to_string(), q), ("y".to_string(), p)]
    );
}

/// Build a `depth`-level nested equality pair whose every level matches only
/// after swapping: `p_k = Eq(p_{k-1}, c_k)` against `g_k = Eq(c_k, g_{k-1})`.
///
/// Returns `(pattern, ground, expected binding for x)`.
fn nested_swapped_eq_chain(manager: &mut TermManager, depth: usize) -> (TermId, TermId, TermId) {
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let base = manager.mk_var("c0", int_sort);

    let mut pattern = x;
    let mut ground = base;
    for level in 1..=depth {
        let anchor = manager.mk_var(&format!("c{level}"), int_sort);
        pattern = eq_ordered(manager, pattern, anchor);
        ground = eq_ordered(manager, anchor, ground);
    }
    (pattern, ground, base)
}

/// Four levels of nested equalities, each needing the swapped orientation.
#[test]
fn four_level_nested_eq_matches_through_every_swap() {
    let mut manager = TermManager::new();
    let (pattern, ground, expected) = nested_swapped_eq_chain(&mut manager, 4);

    let bound = bound_int_vars(&mut manager, &["x"]);
    let matcher = PatternMatcher::new();
    let subst = matcher
        .try_match_term(pattern, ground, &bound, &manager)
        .expect("every level matches after swapping");

    assert_eq!(
        resolved(&manager, &subst),
        vec![("x".to_string(), expected)]
    );
}

/// The same shape at a depth where the retired `4^depth` `Eq` arm would not
/// finish: 16 levels is ~4.3e9 recursive calls there, and one backtracking
/// step per level here.
#[test]
fn nested_eq_does_not_blow_up_exponentially() {
    let mut manager = TermManager::new();
    let (pattern, ground, expected) = nested_swapped_eq_chain(&mut manager, 16);

    let bound = bound_int_vars(&mut manager, &["x"]);
    let matcher = PatternMatcher::new();

    let started = oxiz_time::Instant::now();
    let subst = matcher
        .try_match_term(pattern, ground, &bound, &manager)
        .expect("every level matches after swapping");
    let elapsed = started.elapsed();

    assert_eq!(
        resolved(&manager, &subst),
        vec![("x".to_string(), expected)]
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "16 nested equalities took {elapsed:?}; the search is no longer linear"
    );
}

// ===========================================================================
// Non-matches still fail (and fail fast)
// ===========================================================================

#[test]
fn arity_and_head_mismatches_reject() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;

    let x = manager.mk_var("x", int_sort);
    let a = manager.mk_var("a", int_sort);
    let b = manager.mk_var("b", int_sort);

    let pattern = manager.mk_apply("f", vec![x, x], int_sort);
    let wrong_arity = manager.mk_apply("f", vec![a], int_sort);
    let wrong_head = manager.mk_apply("g", vec![a, b], int_sort);
    let inconsistent = manager.mk_apply("f", vec![a, b], int_sort);

    let bound = bound_int_vars(&mut manager, &["x"]);
    let matcher = PatternMatcher::new();

    assert!(
        matcher
            .try_match_term(pattern, wrong_arity, &bound, &manager)
            .is_none()
    );
    assert!(
        matcher
            .try_match_term(pattern, wrong_head, &bound, &manager)
            .is_none()
    );
    // f(x, x) vs f(a, b): the repeated variable cannot be both.
    assert!(
        matcher
            .try_match_term(pattern, inconsistent, &bound, &manager)
            .is_none()
    );
}

// ===========================================================================
// Deep regression: no native stack involved
// ===========================================================================

/// A 6 250-level pattern and ground term matched on a 128 KiB stack.
///
/// The retired `match_recursive` needed one native frame per level in both
/// the `Apply` chain and the `Eq` chain, so either half of this overflowed
/// and aborted the process. Both halves are now heap-stack walks.
#[test]
fn deep_pattern_and_term_match_on_a_small_stack() {
    const DEPTH: usize = 6_250;
    const STACK_SIZE: usize = 1 << 17;

    let worker = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let mut manager = TermManager::new();
            let int_sort = manager.sorts.int_sort;

            // Half 1: f(f(...f(x)...)) against f(f(...f(a)...)).
            let x = manager.mk_var("x", int_sort);
            let a = manager.mk_var("a", int_sort);
            let mut pattern = x;
            let mut ground = a;
            for _ in 0..DEPTH {
                pattern = manager.mk_apply("f", vec![pattern], int_sort);
                ground = manager.mk_apply("f", vec![ground], int_sort);
            }

            let bound = bound_int_vars(&mut manager, &["x"]);
            let matcher = PatternMatcher::new();
            let subst = matcher
                .try_match_term(pattern, ground, &bound, &manager)
                .expect("a deep unary chain matches");
            assert_eq!(subst.len(), 1);

            // Half 2: the same depth of nested equalities, every level
            // requiring the swapped orientation (so every level also
            // creates and consumes a choice point).
            let (deep_pattern, deep_ground, expected) =
                nested_swapped_eq_chain(&mut manager, DEPTH);
            let deep_subst = matcher
                .try_match_term(deep_pattern, deep_ground, &bound, &manager)
                .expect("a deep equality chain matches after swapping at every level");
            assert_eq!(
                resolved(&manager, &deep_subst),
                vec![("x".to_string(), expected)]
            );
        })
        .expect("spawning the 128 KiB worker thread");

    worker.join().expect("the deep match must not overflow");
}
