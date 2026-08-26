//! Union-find equality classes shared by the lightweight theory modules.
//!
//! The theory modules in [`crate::theories`] each need the same small piece of
//! machinery: merge terms that have been asserted (or deduced) equal, ask
//! whether two terms are currently in the same class, and — when a class turns
//! out to be contradictory — hand back the chain of terms that links the two
//! offenders so the caller can report *why*.
//!
//! This is deliberately a plain union-find with an explanation graph, not a
//! congruence closure: congruence is applied by the individual theories over
//! the operators they know about (see `BitVectorTheory::propagate`).

use crate::ast::TermId;
#[allow(unused_imports)]
use crate::prelude::*;

/// Equality classes over terms, with explanations for merges.
///
/// Every successful merge records an undirected edge between the two merged
/// terms. [`EqClasses::explain`] walks those edges, so the chain it returns
/// consists of steps that were each individually asserted or deduced.
#[derive(Debug, Clone, Default)]
pub(crate) struct EqClasses {
    /// Union-find parent pointers; a root maps to itself.
    parent: FxHashMap<TermId, TermId>,
    /// Union-by-rank ranks, keyed by root.
    rank: FxHashMap<TermId, u32>,
    /// Adjacency of the merge steps, used only for explanations.
    edges: FxHashMap<TermId, Vec<TermId>>,
}

impl EqClasses {
    /// Create an empty set of classes.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Make `term` a known node (a singleton class if it is new).
    pub(crate) fn add(&mut self, term: TermId) {
        self.parent.entry(term).or_insert(term);
    }

    /// Representative of `term`'s class, with path compression.
    ///
    /// A term that was never added is its own representative.
    pub(crate) fn find(&mut self, term: TermId) -> TermId {
        let mut root = term;
        while let Some(&next) = self.parent.get(&root) {
            if next == root {
                break;
            }
            root = next;
        }

        let mut cursor = term;
        while let Some(&next) = self.parent.get(&cursor) {
            if next == cursor {
                break;
            }
            self.parent.insert(cursor, root);
            cursor = next;
        }

        root
    }

    /// Merge the classes of `a` and `b`.
    ///
    /// Returns `true` when this merge was new information (the two terms were
    /// in different classes), `false` when they were already equal.
    pub(crate) fn union(&mut self, a: TermId, b: TermId) -> bool {
        self.add(a);
        self.add(b);

        let (root_a, root_b) = (self.find(a), self.find(b));
        if root_a == root_b {
            return false;
        }

        let rank_a = self.rank.get(&root_a).copied().unwrap_or(0);
        let rank_b = self.rank.get(&root_b).copied().unwrap_or(0);
        if rank_a < rank_b {
            self.parent.insert(root_a, root_b);
        } else if rank_b < rank_a {
            self.parent.insert(root_b, root_a);
        } else {
            self.parent.insert(root_b, root_a);
            self.rank.insert(root_a, rank_a + 1);
        }

        self.edges.entry(a).or_default().push(b);
        self.edges.entry(b).or_default().push(a);
        true
    }

    /// Whether `a` and `b` are currently in the same class.
    pub(crate) fn are_equal(&mut self, a: TermId, b: TermId) -> bool {
        self.find(a) == self.find(b)
    }

    /// A chain of terms linking `a` to `b` through recorded merge steps.
    ///
    /// The returned vector starts at `a` and ends at `b`, and consecutive
    /// entries were merged directly. It is empty when the two terms are not
    /// connected; when `a == b` it is the single-element chain `[a]`.
    pub(crate) fn explain(&self, a: TermId, b: TermId) -> Vec<TermId> {
        if a == b {
            return vec![a];
        }

        let mut came_from: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut queue: VecDeque<TermId> = VecDeque::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        queue.push_back(a);
        seen.insert(a);

        while let Some(current) = queue.pop_front() {
            if current == b {
                let mut chain = vec![b];
                let mut cursor = b;
                while let Some(&previous) = came_from.get(&cursor) {
                    chain.push(previous);
                    cursor = previous;
                }
                chain.reverse();
                return chain;
            }

            let Some(neighbours) = self.edges.get(&current) else {
                continue;
            };
            for &neighbour in neighbours {
                if seen.insert(neighbour) {
                    came_from.insert(neighbour, current);
                    queue.push_back(neighbour);
                }
            }
        }

        Vec::new()
    }

    /// All known nodes, grouped by representative and sorted for determinism.
    pub(crate) fn classes(&mut self) -> Vec<Vec<TermId>> {
        let mut nodes: Vec<TermId> = self.parent.keys().copied().collect();
        nodes.sort_unstable_by_key(|term| term.0);

        let mut grouped: FxHashMap<TermId, Vec<TermId>> = FxHashMap::default();
        for node in nodes {
            let root = self.find(node);
            grouped.entry(root).or_default().push(node);
        }

        let mut classes: Vec<Vec<TermId>> = grouped.into_values().collect();
        classes.sort_unstable_by_key(|class| class.first().map_or(u32::MAX, |term| term.0));
        classes
    }

    /// Whether `term` has been added as a node.
    pub(crate) fn contains(&self, term: TermId) -> bool {
        self.parent.contains_key(&term)
    }

    /// Number of known nodes.
    pub(crate) fn len(&self) -> usize {
        self.parent.len()
    }

    /// Forget every class and every recorded merge.
    pub(crate) fn reset(&mut self) {
        self.parent.clear();
        self.rank.clear();
        self.edges.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_term_is_its_own_representative() {
        let mut classes = EqClasses::new();
        assert_eq!(classes.find(TermId(7)), TermId(7));
        assert_eq!(classes.len(), 0);
    }

    #[test]
    fn union_reports_new_information_once() {
        let mut classes = EqClasses::new();
        assert!(classes.union(TermId(1), TermId(2)));
        assert!(!classes.union(TermId(2), TermId(1)));
        assert!(classes.are_equal(TermId(1), TermId(2)));
    }

    #[test]
    fn transitive_merges_share_a_representative() {
        let mut classes = EqClasses::new();
        classes.union(TermId(1), TermId(2));
        classes.union(TermId(2), TermId(3));
        assert!(classes.are_equal(TermId(1), TermId(3)));
        assert_eq!(classes.classes().len(), 1);
    }

    #[test]
    fn explain_returns_the_merge_chain() {
        let mut classes = EqClasses::new();
        classes.union(TermId(1), TermId(2));
        classes.union(TermId(2), TermId(3));

        let chain = classes.explain(TermId(1), TermId(3));
        assert_eq!(chain, vec![TermId(1), TermId(2), TermId(3)]);
    }

    #[test]
    fn explain_is_empty_for_disconnected_terms() {
        let mut classes = EqClasses::new();
        classes.union(TermId(1), TermId(2));
        assert!(classes.explain(TermId(1), TermId(9)).is_empty());
    }

    #[test]
    fn reset_clears_everything() {
        let mut classes = EqClasses::new();
        classes.union(TermId(1), TermId(2));
        classes.reset();
        assert_eq!(classes.len(), 0);
        assert!(!classes.are_equal(TermId(1), TermId(2)));
    }
}
