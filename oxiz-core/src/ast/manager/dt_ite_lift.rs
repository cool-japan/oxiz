//! A tester, a selector or an equality applied to a datatype-sorted `ite`,
//! lifted into the `ite`'s branches at construction (`#P2b-90`, re-fix pass
//! 18, decision (84)).
//!
//! ```text
//! ((_ is C) (ite c a b))  ->  (ite c ((_ is C) a) ((_ is C) b))
//! (sel (ite c a b))       ->  (ite c (sel a) (sel b))
//! (= (ite c a b) t)       ->  (ite c (= a t) (= b t))
//! ```
//!
//! Each rewrite is an identity of every theory with an `ite` (the operator
//! distributes over the two branches), so it changes no model.  It is done
//! where the term is built because the solver decides a non-Bool `ite` through
//! a proxy constant (`solver::encode::bool_euf_encoding` replaces `(ite c a b)`
//! by `p` beside `c ⇒ p = a`, `¬c ⇒ p = b`) while the datatype axioms read the
//! assertions as written: a tester applied to the written `ite` and the same
//! tester applied to the proxy were two unrelated atoms, so `c` false beside
//! `(< 0 (ite ((_ is cons) (ite c l1 nil)) 1 0))` answered `sat` on every
//! build (0.3.3, `c4b04b7`, `c702310` and every round-4 pass through 17; z3
//! `unsat`).  Lifted, the tester reads the branches — a variable and a
//! constructor application, both axiomatised (and `((_ is cons) nil)` folds to
//! `false` at once, `dt_fold`) — and no tester over an `ite` is ever built.
//! Substitution rebuilds testers, selectors and equalities through these
//! entry points, so an `ite` that a substitution puts under one (a quantifier
//! instance, a printed function table substituted into an assertion) is lifted
//! too.  An `ite` in any other position — an argument of an uninterpreted
//! function or of a constructor — stays, and the solver reads it through its
//! proxy (`solver::dt_axioms::Solver::encoded_dt_scan`).
//!
//! # Bounds
//!
//! The lift rebuilds the `ite` DAG once, memoised per node, so it builds at
//! most one `ite` per `ite` node of the operand and is linear in its size; an
//! operand with more than [`MAX_ITE_LIFT_NODES`] `ite` nodes is left as
//! written (the solver then reads it through its proxy, as any `ite` the lift
//! does not reach).  An equality of two `ite`s lifts both sides, a product,
//! only while the product of their node counts stays within the same bound.
//!
//! What the bound costs (adversarial recheck 18, `TODO.md` `#P2b-93`, open):
//! past it the proxy route is sound but does not decide at scale.  A tester
//! over a chain of 250 `ite` nodes whose branches are all `nil` is refuted in
//! 0.47 s; the same chain at 258 nodes gets no answer in 130 s (release), and
//! with the lift switched off the proxy route alone refutes 120 nodes in
//! 0.42 s, 150 in 22.4 s and none of 180 or 250 in 130 s.  Never a wrong
//! verdict at any size measured.  The levers: bound only the equality's
//! product (the tester and selector lifts are linear and memoised, so any
//! size lifts), or one axiom family per proxy rather than per member in the
//! encoded scan.

use super::super::term::{TermId, TermKind};
use super::TermManager;
use crate::prelude::*;

/// `ite` nodes one lift may rebuild (and the bound on the product of the two
/// sides of a lifted equality).
pub const MAX_ITE_LIFT_NODES: usize = 256;

impl TermManager {
    /// The `ite` nodes of the datatype-sorted `ite` tree rooted at `root`, in
    /// post-order (branches before the `ite` that holds them); `None` when
    /// `root` is not a datatype-sorted `ite` or the tree has more than
    /// [`MAX_ITE_LIFT_NODES`] `ite` nodes.
    pub(super) fn dt_ite_nodes(&self, root: TermId) -> Option<Vec<TermId>> {
        let node = self.get(root)?;
        if !matches!(node.kind, TermKind::Ite(..)) || !self.sorts.is_datatype(node.sort) {
            return None;
        }
        let sort = node.sort;
        let mut order: Vec<TermId> = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        // (node, branches already pushed)
        let mut stack: Vec<(TermId, bool)> = vec![(root, false)];
        while let Some((current, expanded)) = stack.pop() {
            if expanded {
                order.push(current);
                continue;
            }
            if !seen.insert(current) {
                continue;
            }
            if seen.len() > MAX_ITE_LIFT_NODES {
                return None;
            }
            let TermKind::Ite(_, then_branch, else_branch) = self.get(current)?.kind else {
                continue;
            };
            stack.push((current, true));
            for branch in [else_branch, then_branch] {
                if self
                    .get(branch)
                    .is_some_and(|t| matches!(t.kind, TermKind::Ite(..)) && t.sort == sort)
                    && !seen.contains(&branch)
                {
                    stack.push((branch, false));
                }
            }
        }
        Some(order)
    }

    /// `leaf` applied to every leaf (every branch that is not itself an
    /// `ite`) of the datatype-sorted `ite` tree `arg`, with the tree rebuilt
    /// over the results; `None` when `arg` is not such an `ite` or is too
    /// large to lift (see the module docs).
    pub(super) fn lift_over_dt_ite(
        &mut self,
        arg: TermId,
        leaf: &mut dyn FnMut(&mut Self, TermId) -> TermId,
    ) -> Option<TermId> {
        let order = self.dt_ite_nodes(arg)?;
        let mut lifted: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut leaves: FxHashMap<TermId, TermId> = FxHashMap::default();
        for ite in order {
            let TermKind::Ite(condition, then_branch, else_branch) = self.get(ite)?.kind else {
                return None;
            };
            let mut branch_value = |manager: &mut Self, branch: TermId| -> TermId {
                if let Some(&done) = lifted.get(&branch) {
                    return done;
                }
                if let Some(&done) = leaves.get(&branch) {
                    return done;
                }
                let value = leaf(manager, branch);
                leaves.insert(branch, value);
                value
            };
            let then_value = branch_value(self, then_branch);
            let else_value = branch_value(self, else_branch);
            let rebuilt = self.mk_ite(condition, then_value, else_value);
            lifted.insert(ite, rebuilt);
        }
        lifted.get(&arg).copied()
    }

    /// `(= lhs rhs)` lifted over a datatype-sorted `ite` on either side, or
    /// `None` when neither side is one (or the lift is out of bounds).
    pub(super) fn lift_dt_ite_eq(&mut self, lhs: TermId, rhs: TermId) -> Option<TermId> {
        let left = self.dt_ite_nodes(lhs).map(|nodes| nodes.len());
        let right = self.dt_ite_nodes(rhs).map(|nodes| nodes.len());
        match (left, right) {
            (Some(l), Some(r)) => {
                if l.saturating_mul(r) > MAX_ITE_LIFT_NODES {
                    return None;
                }
                self.lift_over_dt_ite(lhs, &mut |m, leaf| m.mk_eq(leaf, rhs))
            }
            (Some(_), None) => self.lift_over_dt_ite(lhs, &mut |m, leaf| m.mk_eq(leaf, rhs)),
            (None, Some(_)) => self.lift_over_dt_ite(rhs, &mut |m, leaf| m.mk_eq(lhs, leaf)),
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::TermManager;
    use super::MAX_ITE_LIFT_NODES;
    use crate::ast::term::TermKind;
    use crate::sort::{DataTypeConstructor, SortId};
    use smallvec::smallvec;

    /// `L = nil | cons(hd Int, tl L)` in `manager`, with its sort.
    fn list(manager: &mut TermManager) -> SortId {
        let sort = manager.sorts.mk_datatype_sort("L");
        let int = manager.sorts.int_sort;
        let cons = DataTypeConstructor {
            name: manager.intern_str("cons"),
            selectors: smallvec![
                (manager.intern_str("hd"), int),
                (manager.intern_str("tl"), sort)
            ],
        };
        let nil = DataTypeConstructor {
            name: manager.intern_str("nil"),
            selectors: smallvec![],
        };
        manager.sorts.declare_datatype("L", vec![nil, cons]);
        sort
    }

    /// `M15`: the tester over `(ite c l1 nil)` is `(and c ((_ is cons) l1))`
    /// — the `nil` branch folds to `false` — and no tester over an `ite` is
    /// built.
    #[test]
    fn a_tester_over_an_ite_reads_its_branches() {
        let mut m = TermManager::new();
        let sort = list(&mut m);
        let bool_sort = m.sorts.bool_sort;
        let c = m.mk_var("c", bool_sort);
        let l1 = m.mk_var("l1", sort);
        let nil = m.mk_dt_constructor("nil", [], sort);
        let chosen = m.mk_ite(c, l1, nil);
        let tester = m.mk_dt_tester("cons", chosen);
        let on_l1 = m.mk_dt_tester("cons", l1);
        let fls = m.mk_false();
        assert_eq!(tester, m.mk_ite(c, on_l1, fls));
        assert!(!matches!(
            m.get(tester).map(|t| &t.kind),
            Some(TermKind::DtTester { .. })
        ));
    }

    /// A selector and an equality lift the same way; an `ite` nested in a
    /// branch is lifted through, and a shared node is rebuilt once.
    #[test]
    fn a_selector_and_an_equality_lift_through_nested_ites() {
        let mut m = TermManager::new();
        let sort = list(&mut m);
        let bool_sort = m.sorts.bool_sort;
        let int = m.sorts.int_sort;
        let (c, d) = (m.mk_var("c", bool_sort), m.mk_var("d", bool_sort));
        let (l1, l2) = (m.mk_var("l1", sort), m.mk_var("l2", sort));
        let inner = m.mk_ite(d, l1, l2);
        let outer = m.mk_ite(c, inner, l1);
        let head = m.mk_dt_selector("hd", outer, int);
        let (h1, h2) = (
            m.mk_dt_selector("hd", l1, int),
            m.mk_dt_selector("hd", l2, int),
        );
        let inner_head = m.mk_ite(d, h1, h2);
        assert_eq!(head, m.mk_ite(c, inner_head, h1));
        let eq = m.mk_eq(outer, l2);
        let (e1, e2) = (m.mk_eq(l1, l2), m.mk_true());
        let inner_eq = m.mk_ite(d, e1, e2);
        assert_eq!(eq, m.mk_ite(c, inner_eq, e1));
    }

    /// An `ite` tree past the bound is left as written.
    #[test]
    fn an_ite_tree_past_the_bound_is_left_as_written() {
        let mut m = TermManager::new();
        let sort = list(&mut m);
        let bool_sort = m.sorts.bool_sort;
        let mut chain = m.mk_var("l0", sort);
        for n in 0..=MAX_ITE_LIFT_NODES {
            let cond = m.mk_var(&format!("c{n}"), bool_sort);
            let next = m.mk_var(&format!("l{}", n + 1), sort);
            chain = m.mk_ite(cond, next, chain);
        }
        let tester = m.mk_dt_tester("cons", chain);
        assert!(matches!(
            m.get(tester).map(|t| &t.kind),
            Some(TermKind::DtTester { .. })
        ));
    }
}
