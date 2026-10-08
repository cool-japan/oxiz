//! Functional consistency of the candidate model — model-based theory
//! combination for uninterpreted functions (re-fix pass 15, decision (68),
//! `#P2b-74`).
//!
//! The congruence closure keeps `f(y)` and `f(4)` in different classes
//! unless it has merged `y` with `4`, and the arithmetic solver is free to
//! give `y` the value `4` without telling it: `y = x + 2` beside `x = 2` is an
//! arithmetic fact, not an equality the closure sees.  The candidate model
//! then says `f(y) = 1` and `f(4) = 0` — two values for `f` at `4`, a model
//! that is not one.  Before this pass that surfaced only as a `debug_assert!`
//! in the table printer (`#P2b-74`) and, in release, as a table whose first
//! entry won — a published model that falsifies its script.
//!
//! The repair is the one Z3 makes (model-based theory combination): at a
//! quantifier-free candidate model, two applications of one function whose
//! arguments the theories valued alike and whose results they valued
//! differently get the Ackermann lemma `⋀ aᵢ = bᵢ ⇒ f(a⃗) = f(b⃗)`, and the
//! search runs again.  (`s00303` of adversarial recheck 14's corpus:
//! `f(m) + 7 = f(0)` printed `m = 0`.)
//! Each lemma is valid in every model (it is functional congruence), so no
//! verdict can change for it — only the candidate, which now has to agree
//! with itself.  Arguments or results no theory valued (a default the model
//! builder recorded) are left to the printer's fresh values (`#P2b-71`),
//! which key every table by the values of its arguments.  The rounds are
//! charged to the array refinement's round budget.
//!
//! A result is compared by the value its class prints (`#P2b-84`, re-fix pass
//! 16): a scalar by its literal, a datatype by its reconstructed value (a
//! literal constructor application in the class, or the model's value), an
//! uninterpreted sort by its congruence class, because the printer gives
//! every class its own witness.  Keyed on scalars alone, a function into a
//! datatype, an enumeration or an uninterpreted sort got no lemma: `(h x) =
//! [1]`, `(h y) = [2]` over `x, y ∈ [0, 5]` printed `x = y = 0` beside `h`
//! constantly `[1]` on every build since `c4b04b7`.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortKind;
use rustc_hash::FxHashMap;

use super::Solver;
use super::array_refinement::ArrayRefinementStep;

/// Whether `sort` CONTAINS a datatype (an enumeration is one) or an
/// uninterpreted sort: is one, or is an array whose index or element sort
/// contains one, at any depth.
pub(super) fn sort_contains_value_sort(
    sort: oxiz_core::sort::SortId,
    manager: &TermManager,
) -> bool {
    let mut pending = vec![sort];
    let mut seen: Vec<oxiz_core::sort::SortId> = Vec::new();
    while let Some(current) = pending.pop() {
        if seen.contains(&current) {
            continue;
        }
        seen.push(current);
        match manager.sorts.get(current).map(|s| &s.kind) {
            Some(SortKind::Datatype(_) | SortKind::Uninterpreted(_)) => return true,
            Some(SortKind::Array { domain, range }) => {
                pending.push(*domain);
                pending.push(*range);
            }
            _ => {}
        }
    }
    false
}

impl Solver {
    /// One functional-consistency round against the candidate model (see
    /// the module docs): `NoLemma` when every function agrees with itself,
    /// `Resolve` when lemmas were asserted and the solver is prepared for a
    /// fresh search, `OutOfBudget` (model cleared) when the round budget is
    /// spent.
    pub(super) fn uf_consistency_round(
        &mut self,
        manager: &mut TermManager,
        rounds: &mut usize,
        conflict_budget: Option<u64>,
        deadline: Option<oxiz_time::Instant>,
    ) -> ArrayRefinementStep {
        // Quantifier-free candidates only: on a quantified goal the MBQI loop
        // owns the search state across rounds, and a lemma per candidate there
        // kept adding original clauses on repeated checks of an unchanged goal
        // and ran one `scope_rebase_tests` goal past 180 s (measured, re-fix
        // pass 15); a quantified model that is not a function is withheld by
        // the printed-model certificate instead (`printed_check`).
        if self.has_quantifiers || !self.euf.has_app_nodes() || self.model.is_none() {
            return ArrayRefinementStep::NoLemma;
        }
        let lemmas: Vec<TermId> = self
            .functional_collisions(manager)
            .into_iter()
            .filter(|lemma| !self.dt_axiom_instances.contains(lemma))
            .collect();
        if lemmas.is_empty() {
            return ArrayRefinementStep::NoLemma;
        }
        *rounds = rounds.saturating_add(1);
        if *rounds >= super::array_refinement::MAX_ARRAY_REFINEMENT_ROUNDS {
            self.model = None;
            self.unsat_core = None;
            return ArrayRefinementStep::OutOfBudget;
        }
        for lemma in lemmas {
            // The datatype axioms' lemma path: encode, assert at the current
            // assertion level, deduplicate and journal the mark.
            self.assert_dt_lemma(lemma, manager);
        }
        self.sat.set_deadline(deadline);
        self.bv.set_budget(conflict_budget, deadline);
        self.instantiate_arith_axioms(manager);
        self.rebase_theory_state_for_round();
        self.debug_check_invariants("check_core: after functional-consistency backtrack");
        ArrayRefinementStep::Resolve
    }

    /// At a quantified `sat` exit whose candidate the ground gate refused
    /// (`quantified_model_refutes_ground_assertions`), the Ackermann lemma of
    /// every functional collision not asserted before; `true` when one was
    /// asserted, and the caller runs one more round instead of answering
    /// `unknown`.
    ///
    /// Only at those exits, never per round: a lemma per quantified candidate
    /// kept adding original clauses on repeated checks of an unchanged goal
    /// (`uf_consistency_round`'s note).  An exit is reached once per
    /// candidate the MBQI loop would otherwise report, every lemma is asserted
    /// at most once (the datatype lemma path deduplicates, and a refusal with
    /// nothing new still answers `unknown`), and the rounds it adds are the
    /// MBQI loop's own, under its iteration bound.  Recheck 14's
    /// `g14a/s00761` and `s01200` answered `unknown` here (the gate refusing
    /// `f(m) = 11` beside `f(5) = 0` with `m = 5`) where `c702310` and re-fix
    /// pass 14 answered `sat`.
    pub(super) fn assert_exit_consistency_lemmas(&mut self, manager: &mut TermManager) -> bool {
        if !self.euf.has_app_nodes() || self.model.is_none() {
            return false;
        }
        let before = self.dt_axiom_instances.len();
        let lemmas: Vec<TermId> = self
            .functional_collisions(manager)
            .into_iter()
            .filter(|lemma| !self.dt_axiom_instances.contains(lemma))
            .collect();
        for lemma in lemmas {
            self.assert_dt_lemma(lemma, manager);
        }
        self.dt_axiom_instances.len() > before
    }

    /// The Ackermann lemma of every pair of applications of one function
    /// whose arguments the candidate values alike and whose results it
    /// values differently, in term-id order.
    fn functional_collisions(&self, manager: &mut TermManager) -> Vec<TermId> {
        let Some(model) = self.model.as_ref() else {
            return Vec::new();
        };
        let mut applications: Vec<(TermId, oxiz_core::interner::Spur, Vec<TermId>)> = Vec::new();
        for node in self.euf.all_node_indices() {
            let Some(term) = self.euf.node_term(node) else {
                continue;
            };
            if let Some(TermKind::Apply { func, args }) = manager.get(term).map(|t| &t.kind) {
                applications.push((term, *func, args.to_vec()));
            }
        }
        applications.sort_unstable_by_key(|(term, _, _)| term.raw());
        applications.dedup_by_key(|(term, _, _)| *term);

        // (function, the theory's values of the arguments) -> applications.
        let mut points: FxHashMap<(oxiz_core::interner::Spur, Vec<String>), Vec<TermId>> =
            FxHashMap::default();
        let mut order: Vec<(oxiz_core::interner::Spur, Vec<String>)> = Vec::new();
        for (term, func, args) in &applications {
            let values: Option<Vec<String>> = args
                .iter()
                .map(|&arg| self.theory_class_value(arg, model, manager))
                .collect();
            let Some(values) = values else {
                continue;
            };
            let key = (*func, values);
            let slot = points.entry(key.clone()).or_default();
            if slot.is_empty() {
                order.push(key);
            }
            slot.push(*term);
        }

        let mut lemmas: Vec<TermId> = Vec::new();
        for key in order {
            let Some(members) = points.get(&key) else {
                continue;
            };
            let Some((&first, rest)) = members.split_first() else {
                continue;
            };
            let Some(first_value) = self.theory_class_value(first, model, manager) else {
                continue;
            };
            for &other in rest {
                if self.euf_same_class(first, other) {
                    continue;
                }
                let Some(other_value) = self.theory_class_value(other, model, manager) else {
                    continue;
                };
                if other_value == first_value {
                    continue;
                }
                let (Some(TermKind::Apply { args: a, .. }), Some(TermKind::Apply { args: b, .. })) = (
                    manager.get(first).map(|t| t.kind.clone()),
                    manager.get(other).map(|t| t.kind.clone()),
                ) else {
                    continue;
                };
                let premises: Vec<TermId> = a
                    .iter()
                    .zip(b.iter())
                    .filter(|(x, y)| x != y)
                    .map(|(&x, &y)| manager.mk_eq(x, y))
                    .collect();
                let premise = manager.mk_and(premises);
                let conclusion = manager.mk_eq(first, other);
                lemmas.push(manager.mk_implies(premise, conclusion));
            }
        }
        lemmas
    }

    /// The value a theory chose for `term`'s congruence class, as a
    /// comparable key: a model entry of a member that is not a default, a
    /// literal member, or — for a compound member such as `(+ p 1)`, which
    /// has no entry of its own — the value its leaves fold to.
    ///
    /// An ARRAY whose sort contains a datatype, an enumeration or an
    /// uninterpreted sort has no comparable value of its own (it reads such
    /// values), so it is keyed by its congruence class, as an uninterpreted
    /// element is (decision (86), re-fix pass 18): two reads of an array of
    /// arrays into `U`, or two results of a function into `(Array Int C)`, that
    /// the closure kept apart at arguments the theories valued alike earn the
    /// lemma, and their own reads then merge.  An array of scalars stays as it
    /// was (no key).
    pub(super) fn theory_class_value(
        &self,
        term: TermId,
        model: &crate::solver::Model,
        manager: &TermManager,
    ) -> Option<String> {
        if manager.get(term).is_some_and(|t| {
            matches!(
                manager.sorts.get(t.sort).map(|s| &s.kind),
                Some(SortKind::Array { .. })
            ) && sort_contains_value_sort(t.sort, manager)
        }) {
            return self
                .euf_class_representative(term)
                .map(|class| format!("array{class}"));
        }
        let mut members = self.euf_class_terms(term);
        members.sort_unstable_by_key(|member| member.raw());
        for &member in &members {
            if let Some(value) = model.get(member)
                && !model.is_defaulted(member)
            {
                if is_literal(value, manager) {
                    return self
                        .eval_in_model(value, model, manager, 0)
                        .map(|v| format!("{v:?}"));
                }
                // A datatype value (`#P2b-84`): values are hash-consed, so
                // one value is one term.
                if is_datatype_value(value, manager) {
                    return Some(format!("dt{}", value.raw()));
                }
            }
            if is_literal(member, manager) {
                return self
                    .eval_in_model(member, model, manager, 0)
                    .map(|v| format!("{v:?}"));
            }
            if is_datatype_value(member, manager) {
                return Some(format!("dt{}", member.raw()));
            }
        }
        // An uninterpreted sort's value is its congruence class: the printer
        // gives every class its own witness (`#P2b-71`, `#P2b-84`).
        let uninterpreted = manager.get(term).is_some_and(|t| {
            manager
                .sorts
                .get(t.sort)
                .is_some_and(|sort| matches!(sort.kind, SortKind::Uninterpreted(_)))
        });
        if uninterpreted {
            return self
                .euf_class_representative(term)
                .map(|rep| format!("uc{rep}"));
        }
        members
            .iter()
            .filter(|&&member| model.get(member).is_none() && is_compound(member, manager))
            .find_map(|&member| self.eval_in_model(member, model, manager, 0))
            .map(|v| format!("{v:?}"))
    }

    /// Whether the congruence closure holds `x` and `y` in one class.
    fn euf_same_class(&self, x: TermId, y: TermId) -> bool {
        if x == y {
            return true;
        }
        match (self.euf.term_to_node(x), self.euf.term_to_node(y)) {
            (Some(a), Some(b)) => self.euf.are_equal_immutable(a, b),
            _ => false,
        }
    }
}

/// Whether `term` is an arithmetic or bit-vector operator application (a
/// term whose value its leaves determine), rather than a leaf.
fn is_compound(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|t| {
        !matches!(
            t.kind,
            TermKind::Var(_)
                | TermKind::Apply { .. }
                | TermKind::Select(..)
                | TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
        )
    })
}

/// Whether `term` is a datatype value: a constructor application whose
/// every argument is a scalar literal or a datatype value.
fn is_datatype_value(term: TermId, manager: &TermManager) -> bool {
    let mut pending: Vec<TermId> = vec![term];
    let mut seen_constructor = false;
    while let Some(current) = pending.pop() {
        match manager.get(current).map(|t| &t.kind) {
            Some(TermKind::DtConstructor { args, .. }) => {
                seen_constructor = true;
                pending.extend(args.iter().copied());
            }
            _ if is_literal(current, manager) => {}
            _ => return false,
        }
    }
    seen_constructor
}

/// Whether `term` is a scalar literal (its own value).
fn is_literal(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|t| {
        matches!(
            t.kind,
            TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
        )
    })
}
