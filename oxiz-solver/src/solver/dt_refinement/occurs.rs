//! Acyclicity **through the congruence closure** (`#P2b-83`, re-fix pass 16).
//!
//! The datatype axioms refute a cycle with the size measure
//! (`solver::dt_axioms`, "Acyclicity"), but only over the terms they
//! axiomatise: the terms of the assertions as written.  The search reasons
//! about the assertions as encoded, and the encoder purifies a numeric
//! argument of an uninterpreted function (`encode::numeric_purification`):
//! `(= (h 1) (cons x (cons 2 (h 1))))` is asserted as `(= (h n) (cons x (cons 2
//! (h n))))` beside `n = 1`, so the size lemmas speak of `(h 1)` and the atom
//! the search holds speaks of `(h n)`.  One constructor deep the refinement's
//! injectivity lemmas happened to close the gap (`#P2b-76`); two deep, every
//! build answered `sat` (z3: `unsat`), and the dev profile's datatype-model
//! net fired on the candidate.
//!
//! This is the occurs check Z3 runs, over the congruence classes of the
//! candidate: a class `K` holding a constructor application `u = C(…, a, …)`
//! whose datatype-sorted argument `a` lies in class `K'` is an edge `K → K'`.
//! A cycle `K₀ → K₁ → … → K₀` makes `u₀` a proper sub-term of itself, so its
//! explanation `E` — the literals that merge each argument with the next
//! application on the cycle — is refuted: the lemma is `¬E`, an instance of a
//! theorem of the datatype theory (acyclicity), valid at every assertion level.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashMap;

use super::super::Solver;

/// One edge of the class graph: the constructor application `u` (a member of
/// the source class) and its argument `a` (a member of the target class).
#[derive(Clone, Copy)]
struct Edge {
    application_node: u32,
    argument_node: u32,
    target: u32,
}

impl Solver {
    /// `Some(lemmas)`: one `¬E` per cycle the class graph of the candidate
    /// holds (empty when it is acyclic); `None` when a cycle's merges cannot
    /// be explained by literals.
    pub(super) fn constructor_cycle_lemmas(
        &mut self,
        manager: &mut TermManager,
    ) -> Option<Vec<TermId>> {
        let mut edges: FxHashMap<u32, Vec<Edge>> = FxHashMap::default();
        let mut nodes: Vec<(TermId, u32)> = Vec::new();
        for node in self.euf.all_node_indices() {
            if let Some(term) = self.euf.node_term(node) {
                nodes.push((term, node));
            }
        }
        nodes.sort_unstable_by_key(|&(term, _)| term.raw());
        for &(term, node) in &nodes {
            let Some(TermKind::DtConstructor { args, .. }) = manager.get(term).map(|t| &t.kind)
            else {
                continue;
            };
            let source = self.euf.find_immutable(node);
            for &arg in args.iter() {
                let is_datatype = manager
                    .get(arg)
                    .is_some_and(|t| manager.sorts.is_datatype(t.sort));
                if !is_datatype {
                    continue;
                }
                let Some(argument_node) = self.euf.term_to_node(arg) else {
                    continue;
                };
                edges.entry(source).or_default().push(Edge {
                    application_node: node,
                    argument_node,
                    target: self.euf.find_immutable(argument_node),
                });
            }
        }
        if edges.is_empty() {
            return Some(Vec::new());
        }
        let mut sources: Vec<u32> = edges.keys().copied().collect();
        sources.sort_unstable();

        // Iterative depth-first search; every back edge closes one cycle.
        let mut state: FxHashMap<u32, u8> = FxHashMap::default(); // 1 = on path, 2 = done
        let mut lemmas: Vec<TermId> = Vec::new();
        for root in sources {
            if state.contains_key(&root) {
                continue;
            }
            // (class, next edge index), and the edge taken into each path entry.
            let mut stack: Vec<(u32, usize)> = vec![(root, 0)];
            let mut taken: Vec<Edge> = Vec::new();
            state.insert(root, 1);
            while let Some(top) = stack.last_mut() {
                let (class, index) = *top;
                top.1 += 1;
                let Some(edge) = edges.get(&class).and_then(|out| out.get(index)).copied() else {
                    state.insert(class, 2);
                    stack.pop();
                    taken.pop();
                    continue;
                };
                match state.get(&edge.target).copied() {
                    Some(1) => {
                        // Back edge: the cycle is the path from `edge.target`
                        // to `class`, closed by `edge`.
                        let start = stack
                            .iter()
                            .position(|&(c, _)| c == edge.target)
                            .unwrap_or(0);
                        let mut cycle: Vec<Edge> = taken[start..].to_vec();
                        cycle.push(edge);
                        lemmas.push(self.cycle_lemma(&cycle, manager)?);
                    }
                    Some(_) => {}
                    None => {
                        state.insert(edge.target, 1);
                        stack.push((edge.target, 0));
                        taken.push(edge);
                    }
                }
            }
        }
        Some(lemmas)
    }

    /// `¬E` for one cycle: `E` merges each edge's argument with the
    /// application the next edge leaves from (the last edge's with the first
    /// edge's application).
    fn cycle_lemma(&mut self, cycle: &[Edge], manager: &mut TermManager) -> Option<TermId> {
        let mut hypotheses: Vec<TermId> = Vec::new();
        for (index, edge) in cycle.iter().enumerate() {
            let next = cycle.get(index + 1).or_else(|| cycle.first())?;
            if edge.argument_node == next.application_node {
                continue;
            }
            hypotheses.extend(self.explain_merge_as_literals(
                edge.argument_node,
                next.application_node,
                manager,
            )?);
        }
        hypotheses.sort_unstable_by_key(|term| term.raw());
        hypotheses.dedup();
        let negated: Vec<TermId> = hypotheses.iter().map(|&h| manager.mk_not(h)).collect();
        Some(manager.mk_or(negated))
    }
}
