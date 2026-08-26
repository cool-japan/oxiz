//! Iterative (explicit-stack) term-DAG walk primitives shared across Spacer.
//!
//! Every predicate and collector in this crate that used to descend a term
//! with native recursion is expressed here instead. Two independent failure
//! modes motivated the rewrite:
//!
//! * **Unbounded native recursion.** Term nesting in Spacer comes straight
//!   from parsed CHC/SMT input, so a `(and (and (and …)))` chain of arbitrary
//!   depth is attacker-controlled. Every one of these walks returns a plain
//!   `bool`/`()`/`usize` with *no* error channel, so a depth cap could only
//!   ever produce a silently wrong answer — strictly worse than the stack
//!   overflow it would replace. The fix is a heap-allocated `Vec` stack,
//!   which is bounded only by available memory.
//! * **Exponential re-expansion.** Terms are hash-consed, so the "tree" is
//!   really a DAG: a term built by `n` doublings has `2^n` tree paths but
//!   only `O(n)` distinct nodes. A recursive walk with no visited set
//!   re-expands every path. Every walk here keeps a `visited` set keyed on
//!   [`TermId`], which is sound for all of them because none of them
//!   interprets binders (they only classify or collect).
//!
//! Child enumeration uses [`oxiz_core::ast::traversal::get_children`], which
//! matches *exhaustively* over [`TermKind`] with no catch-all arm — so a new
//! term variant becomes a compile error there rather than a silently skipped
//! subterm here. The per-site hand-written `match` arms these walks replaced
//! all ended in `_ => false` / `_ => {}`, which silently ignored whole
//! families of terms (`Ite`, `Implies`, `Apply`, `Select`/`Store`, every
//! bitvector and string operation, quantifier bodies, `Let` bodies…). For a
//! *conservative* predicate such as "does this contain an existential
//! variable" that fallthrough is an unsoundness: answering "no" for a term
//! that does contain one lets Spacer keep a conjunct it must have projected
//! away.
//!
//! Reference: Z3's `ast.cpp` / `ast_util.cpp` perform the analogous walks
//! with explicit stacks for the same reasons.

use oxiz_core::ast::traversal::get_children;
use oxiz_core::{TermId, TermKind, TermManager};
use rustc_hash::FxHashSet;

/// Walk every distinct node reachable from `root` and report whether `pred`
/// holds for at least one of them.
///
/// `pred` receives the node's own [`TermId`] together with its kind, or
/// `None` when the id is absent from `manager` (a dangling id — callers
/// decide what that means for them, since the honest answer is "unknown").
/// The walk stops as soon as `pred` returns `true`.
///
/// Every node is visited at most once, so a shared DAG costs `O(nodes)`,
/// not `O(paths)`.
pub fn any_node(
    manager: &TermManager,
    root: TermId,
    mut pred: impl FnMut(TermId, Option<&TermKind>) -> bool,
) -> bool {
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = vec![root];

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let kind = manager.get(current).map(|t| &t.kind);
        if pred(current, kind) {
            return true;
        }
        if let Some(kind) = kind {
            stack.extend(get_children(kind).iter().copied());
        }
    }

    false
}

/// Walk every distinct node reachable from `root` in pre-order (parent
/// before children, children left to right), calling `visit` on each node
/// that exists in `manager`.
///
/// The ordering matches what the recursive implementations this replaces
/// produced, so collectors that push into an order-sensitive `Vec` keep
/// their exact output sequence. Dangling ids are skipped, exactly as the
/// old `let Some(t) = manager.get(term) else { return }` guards did.
pub fn for_each_node(
    manager: &TermManager,
    root: TermId,
    mut visit: impl FnMut(TermId, &TermKind),
) {
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = vec![root];

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(term) = manager.get(current) else {
            continue;
        };
        visit(current, &term.kind);
        // Push children reversed so they pop left to right.
        let children = get_children(&term.kind);
        stack.extend(children.iter().rev().copied());
    }
}

/// Collect the distinct [`TermKind::Var`] nodes reachable from `root`, in
/// pre-order of first occurrence.
#[must_use]
pub fn collect_vars(manager: &TermManager, root: TermId) -> Vec<TermId> {
    let mut vars = Vec::new();
    for_each_node(manager, root, |id, kind| {
        if matches!(kind, TermKind::Var(_)) {
            vars.push(id);
        }
    });
    vars
}

/// Flatten a nested `And` tree into its non-`And` conjuncts, left to right.
///
/// Only `And` nodes are descended; anything else is emitted verbatim, which
/// is exactly what the recursive `collect_conjuncts` / `assert_flat`
/// implementations did. A dangling id is emitted verbatim too (the old code
/// returned `vec![term]` for it), so no conjunct is ever silently dropped.
///
/// An `And` node already expanded is not expanded again. Conjunction is
/// idempotent, so re-emitting a shared sub-conjunction adds nothing
/// logically, while re-expanding it costs exponential time and output size
/// on a shared DAG (`c_n = (and c_{n-1} c_{n-1})` has `2^n` tree leaves and
/// `n` distinct nodes).
#[must_use]
pub fn flatten_conjuncts(manager: &TermManager, root: TermId) -> Vec<TermId> {
    let mut out = Vec::new();
    let mut expanded: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = vec![root];

    while let Some(current) = stack.pop() {
        match manager.get(current).map(|t| &t.kind) {
            Some(TermKind::And(args)) => {
                if !expanded.insert(current) {
                    continue;
                }
                // Reversed so the conjuncts pop in source order.
                stack.extend(args.iter().rev().copied());
            }
            _ => out.push(current),
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stack size and nesting depth shared by the deep-recursion tests below.
    ///
    /// The two are scaled together on purpose: what these tests actually pin
    /// is the *ratio* -- about 21 bytes of stack per nesting level
    /// (128 KiB / 6_250). A natively recursive walk needs far more than that
    /// per frame and still overflows, so the regression keeps every bit of
    /// its detection power. The pair used to be 1 MiB / 50_000 -- the same
    /// 21 bytes -- but `mk_and`/`mk_or` flatten their arguments, so a chain
    /// built with `acc = mk_or([acc, lit])` is quadratic, and 50_000 levels
    /// cost tens of GB of live terms. Never raise `DEEP_DEPTH` without
    /// raising `DEEP_STACK` by the same factor.
    const DEEP_STACK: usize = 1 << 17;
    const DEEP_DEPTH: u32 = 6_250;

    /// Build `not(not(...(x)))` nested `depth` levels deep.
    ///
    /// Interned directly rather than via `mk_not`, which folds double
    /// negation (`not(not(p)) == p`): a loop of `mk_not` calls oscillates
    /// between depth 0 and 1 and never actually nests.
    fn deep_not(manager: &mut TermManager, depth: usize) -> TermId {
        let bool_sort = manager.sorts.bool_sort;
        let x = manager.mk_var("x", bool_sort);
        let mut current = x;
        for _ in 0..depth {
            current = manager.intern_term(TermKind::Not(current), bool_sort);
        }
        current
    }

    #[test]
    fn any_node_finds_var_at_extreme_depth() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let bool_sort = manager.sorts.bool_sort;
                // `mk_not` folds `not(not(p))` back to `p`, and `mk_or`
                // flattens a nested `Or` child into its parent, so neither
                // builds a genuinely deep tree. Intern the `Or` nodes
                // directly so the chain really is `DEEP_DEPTH` deep.
                let x = manager.mk_var("x", bool_sort);
                let mut current = x;
                for i in 0..DEEP_DEPTH {
                    let lit = manager.mk_var(&format!("p{i}"), bool_sort);
                    current =
                        manager.intern_term(TermKind::Or(vec![current, lit].into()), bool_sort);
                }
                assert!(any_node(&manager, current, |id, kind| {
                    matches!(kind, Some(TermKind::Var(_))) && id == x
                }));
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep any_node walk must return");
    }

    #[test]
    fn any_node_is_linear_on_shared_dag() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let one = manager.mk_int(1);
        let mut current = manager.mk_add([x, one]);
        // 60 doublings => 2^60 tree paths, 60 distinct nodes.
        for _ in 0..60 {
            current = manager.mk_add([current, current]);
        }
        // Searching for something absent forces the *whole* DAG to be walked.
        assert!(!any_node(&manager, current, |_, kind| matches!(
            kind,
            Some(TermKind::StringLit(_))
        )));
    }

    #[test]
    fn flatten_conjuncts_preserves_order_and_terminates_deep() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let a = manager.mk_var("a", manager.sorts.bool_sort);
                let b = manager.mk_var("b", manager.sorts.bool_sort);
                let c = manager.mk_var("c", manager.sorts.bool_sort);
                let inner = manager.mk_and([b, c]);
                let outer = manager.mk_and([a, inner]);
                assert_eq!(flatten_conjuncts(&manager, outer), vec![a, b, c]);

                // Deep left-nested conjunction, interned directly rather
                // than via `mk_and` (which would flatten it back into a
                // single wide `And`). `flatten_conjuncts` exists precisely
                // to un-nest a tree like this, so the test is meaningless
                // unless the tree really is `DEEP_DEPTH` deep.
                let bool_sort = manager.sorts.bool_sort;
                let mut leaves = Vec::with_capacity(DEEP_DEPTH as usize);
                let l0 = manager.mk_var("l0", bool_sort);
                leaves.push(l0);
                let mut deep = l0;
                for i in 1..DEEP_DEPTH {
                    let lit = manager.mk_var(&format!("l{i}"), bool_sort);
                    leaves.push(lit);
                    deep = manager.intern_term(TermKind::And(vec![deep, lit].into()), bool_sort);
                }
                assert_eq!(flatten_conjuncts(&manager, deep), leaves);
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep flatten must return");
    }

    /// A single, genuinely *wide* `And` -- built via `mk_and`, which folds a
    /// flat operand list into one `TermKind::And(args)` of length
    /// `DEEP_DEPTH` -- exercises a different path through the explicit stack
    /// than the deep left-nested tree above: one `stack.extend` of
    /// `DEEP_DEPTH` items plus one `expanded` insertion, versus `DEEP_DEPTH`
    /// two-element extends. Both regressions matter and neither subsumes the
    /// other, so both are pinned.
    #[test]
    fn flatten_conjuncts_flattens_a_wide_conjunction() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let leaves: Vec<TermId> = (0..DEEP_DEPTH)
            .map(|i| manager.mk_var(&format!("w{i}"), bool_sort))
            .collect();
        let wide = manager.mk_and(leaves.iter().copied());
        assert_eq!(flatten_conjuncts(&manager, wide), leaves);
    }

    #[test]
    fn for_each_node_visits_deeply_nested_term() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let x = manager.mk_var("x", manager.sorts.int_sort);
                let mut current = x;
                for i in 0..DEEP_DEPTH {
                    let k = manager.mk_int(i);
                    current = manager.mk_add([current, k]);
                }
                let mut count = 0usize;
                for_each_node(&manager, current, |_, _| count += 1);
                assert!(count > DEEP_DEPTH as usize);
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep for_each_node walk must return");
    }

    #[test]
    fn deep_not_helper_builds_a_term() {
        let mut manager = TermManager::new();
        let t = deep_not(&mut manager, 3);
        assert!(manager.get(t).is_some());
    }

    #[test]
    fn collect_vars_reports_first_occurrence_order() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y]);
        let eq = manager.mk_eq(sum, x);
        assert_eq!(collect_vars(&manager, eq), vec![x, y]);
    }
}
