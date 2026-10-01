//! Constructor injectivity and distinctness **through the congruence
//! closure** (`#P2b-76`).
//!
//! The datatype axioms ([`super::dt_axioms`]) are ground lemmas over the terms
//! the assertions spell, and they stop one level below that set (their own
//! termination argument).  `TermManager::mk_eq` decides an equality of two
//! constructor applications at every depth where both sides are written out
//! (`oxiz_core`'s `dt_eq`), but two applications the congruence closure merged
//! *through an opaque term* never meet in an atom:
//!
//! ```smt2
//! (assert (= l3 (cons 1 (cons 2 nil))))
//! (assert (= l3 (cons 1 (cons 2 (cons 3 nil)))))
//! ```
//!
//! puts both lists in `l3`'s class, the selector congruence of the axioms
//! reaches `(tail l3)` and stops, and the search answered `sat` on every build
//! (and printed one list for both, the `d03` model of recheck 14).
//!
//! This is the missing half, as a refinement round at the candidate model
//! (the array refinement's seam, which runs it first): every congruence class
//! holding two constructor applications `u = C(a⃗)` and `v = D(b⃗)` is checked,
//! and a violation becomes a lemma justified by the congruence closure's own
//! explanation `E` of `u = v` (the asserted literals the merge rests on):
//!
//! * `C ≠ D` — `¬E` (constructors are distinct);
//! * `C = D`, a field pair `aᵢ`, `bᵢ` not merged — `¬E ∨ aᵢ = bᵢ`
//!   (constructors are injective), where `aᵢ = bᵢ` is built by `mk_eq`, so a
//!   pair of literal applications that differ deeper folds it to `false`.
//!
//! Each lemma is an instance of a theorem of the datatype theory — `E` entails
//! `u = v` by congruence reasoning alone — so it is valid at every assertion
//! level and may only remove candidate models that are not datatype models.
//! * a cycle of classes along constructor arguments — `¬E` for the merges on
//!   the cycle (acyclicity, [`occurs`], `#P2b-83`).
//!
//! A class it cannot explain (a reason with no literal) leaves the candidate
//! unverified, and the round answers `unknown` rather than `sat`.  Each round
//! adds at least one lemma no earlier round added, and the rounds are charged
//! to the array refinement's round budget, so the loop is bounded.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};

use super::Solver;
use super::array_refinement::ArrayRefinementStep;

/// Acyclicity through the congruence closure (`#P2b-83`).
mod occurs;

impl Solver {
    /// One constructor-congruence refinement round against the candidate
    /// model (see the module docs).  `NoLemma` when every class is a datatype
    /// class; `Resolve` when lemmas were asserted and the solver is prepared
    /// for a fresh search; `OutOfBudget` (model cleared) when a violation
    /// cannot be explained, recurs, or the round budget is spent.
    pub(super) fn dt_refinement_round(
        &mut self,
        manager: &mut TermManager,
        rounds: &mut usize,
        conflict_budget: Option<u64>,
        deadline: Option<oxiz_time::Instant>,
    ) -> ArrayRefinementStep {
        if self.dt_axiom_instances.is_empty() {
            return ArrayRefinementStep::NoLemma;
        }
        let Some(mut lemmas) = self.constructor_class_lemmas(manager) else {
            return self.dt_refinement_gives_up();
        };
        let Some(cycles) = self.constructor_cycle_lemmas(manager) else {
            return self.dt_refinement_gives_up();
        };
        lemmas.extend(cycles);
        if lemmas.is_empty() {
            return ArrayRefinementStep::NoLemma;
        }
        let fresh: Vec<TermId> = lemmas
            .into_iter()
            .filter(|lemma| !self.dt_axiom_instances.contains(lemma))
            .collect();
        // A violation whose lemma is already asserted is a candidate the
        // clause database should have excluded: nothing new to learn, so the
        // candidate is not a model.
        if fresh.is_empty() {
            return self.dt_refinement_gives_up();
        }
        *rounds = rounds.saturating_add(1);
        if *rounds >= super::array_refinement::MAX_ARRAY_REFINEMENT_ROUNDS {
            return self.dt_refinement_gives_up();
        }
        for lemma in fresh {
            self.assert_dt_lemma(lemma, manager);
        }
        self.sat.set_deadline(deadline);
        self.bv.set_budget(conflict_budget, deadline);
        self.rebase_theory_state_for_round();
        self.debug_check_invariants("check_core: after datatype-lemma backtrack");
        ArrayRefinementStep::Resolve
    }

    /// The candidate is not a verified datatype model: clear it, as the array
    /// refinement's budget exits do.
    fn dt_refinement_gives_up(&mut self) -> ArrayRefinementStep {
        self.model = None;
        self.unsat_core = None;
        ArrayRefinementStep::OutOfBudget
    }

    /// The lemmas of every congruence class that holds two constructor
    /// applications violating distinctness or injectivity, in term-id order;
    /// `None` when a violation's merge cannot be explained by literals.
    fn constructor_class_lemmas(&mut self, manager: &mut TermManager) -> Option<Vec<TermId>> {
        let mut by_class: FxHashMap<u32, Vec<(u32, TermId)>> = FxHashMap::default();
        for node in self.euf.all_node_indices() {
            let Some(term) = self.euf.node_term(node) else {
                continue;
            };
            if manager
                .get(term)
                .is_some_and(|data| matches!(data.kind, TermKind::DtConstructor { .. }))
            {
                by_class
                    .entry(self.euf.find_immutable(node))
                    .or_default()
                    .push((node, term));
            }
        }
        let mut classes: Vec<Vec<(u32, TermId)>> = by_class
            .into_values()
            .filter(|members| members.len() > 1)
            .collect();
        for members in &mut classes {
            members.sort_unstable_by_key(|&(_, term)| term.raw());
        }
        classes.sort_unstable_by_key(|members| members.first().map(|&(_, term)| term.raw()));

        let mut lemmas: Vec<TermId> = Vec::new();
        for members in classes {
            let Some(&(first_node, first)) = members.first() else {
                continue;
            };
            for &(node, other) in members.iter().skip(1) {
                let conclusions = self.constructor_pair_conclusions(first, other, manager);
                if conclusions.is_empty() {
                    continue;
                }
                let hypotheses = self.explain_merge_as_literals(first_node, node, manager)?;
                for conclusion in conclusions {
                    let mut disjuncts: Vec<TermId> =
                        hypotheses.iter().map(|&h| manager.mk_not(h)).collect();
                    disjuncts.push(conclusion);
                    lemmas.push(manager.mk_or(disjuncts));
                }
            }
        }
        Some(lemmas)
    }

    /// What `u = v` entails that the congruence closure does not already
    /// hold: `[false]` for two different constructors, one field equality per
    /// field pair not merged for the same constructor, `[]` when nothing.
    fn constructor_pair_conclusions(
        &self,
        u: TermId,
        v: TermId,
        manager: &mut TermManager,
    ) -> Vec<TermId> {
        let (
            Some(TermKind::DtConstructor {
                constructor: c1,
                args: a1,
            }),
            Some(TermKind::DtConstructor {
                constructor: c2,
                args: a2,
            }),
        ) = (
            manager.get(u).map(|d| d.kind.clone()),
            manager.get(v).map(|d| d.kind.clone()),
        )
        else {
            return Vec::new();
        };
        if c1 != c2 || a1.len() != a2.len() {
            return vec![manager.mk_false()];
        }
        let mut out: Vec<TermId> = Vec::new();
        for (&x, &y) in a1.iter().zip(a2.iter()) {
            if self.euf_merged(x, y) {
                continue;
            }
            let equality = manager.mk_eq(x, y);
            if !manager
                .get(equality)
                .is_some_and(|d| matches!(d.kind, TermKind::True))
            {
                out.push(equality);
            }
        }
        out
    }

    /// Whether the congruence closure holds `x` and `y` in one class.
    fn euf_merged(&self, x: TermId, y: TermId) -> bool {
        if x == y {
            return true;
        }
        match (self.euf.term_to_node(x), self.euf.term_to_node(y)) {
            (Some(a), Some(b)) => self.euf.are_equal_immutable(a, b),
            _ => false,
        }
    }

    /// The congruence closure's explanation of `a = b` as the literals of the
    /// current assignment it rests on (each atom at the polarity it holds, a
    /// derived equality expanded into its own justification, a literal value
    /// dropped as a tautology); `None` when some reason names no literal.
    fn explain_merge_as_literals(
        &mut self,
        a: u32,
        b: u32,
        manager: &mut TermManager,
    ) -> Option<Vec<TermId>> {
        let reasons = self.euf.try_explain_eq(a, b)?;
        let mut out: Vec<TermId> = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut expanded: FxHashSet<TermId> = FxHashSet::default();
        let mut pending: Vec<TermId> = reasons;
        while let Some(reason) = pending.pop() {
            if !seen.insert(reason) {
                continue;
            }
            if let Some(&var) = self.term_to_var.get(&reason) {
                let value = self.sat.model().get(var.index()).copied()?;
                if value.is_true() {
                    out.push(reason);
                } else if value.is_false() {
                    out.push(manager.mk_not(reason));
                } else {
                    return None;
                }
                continue;
            }
            if let Some(justification) = self.derived_reasons.literals(reason) {
                if expanded.insert(reason) {
                    pending.extend(justification);
                }
                continue;
            }
            let tautology = manager.get(reason).is_some_and(|d| {
                matches!(
                    d.kind,
                    TermKind::True
                        | TermKind::False
                        | TermKind::IntConst(_)
                        | TermKind::RealConst(_)
                        | TermKind::BitVecConst { .. }
                )
            });
            if !tautology {
                return None;
            }
        }
        out.sort_unstable_by_key(|term| term.raw());
        Some(out)
    }
}
