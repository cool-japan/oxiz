//! Publishing the opaque leaves an array model rests on (`#P2b-34`).
//!
//! An array read is a *leaf* of every theory that reasons about it: the
//! bit-blaster gives `(select arr i)` a free bit-vector, the tableau gives it a
//! column, EUF gives it an application node — and the search then **chooses** a
//! value for it, exactly as it chooses one for a declared variable.  That
//! choice is part of the model, and until this module it was published only
//! when the read happened to be a direct operand of a theory atom.
//!
//! The gap was visible wherever the read sat under an operator:
//! `(= (bvadd (select arr i) #x01) #x06)` answered `sat` — which is right,
//! `arr[i] = 5` satisfies it — and then printed `i = #x00` with
//! `arr = ((as const …) #x00)`, a model in which the assertion reads
//! `0 + 1 = 6`.  `(get-value ((select arr i)))` echoed the term back instead of
//! answering `#x05`.  Nothing was unsound (the verdict came from the circuit,
//! which had the right value all along), but the published model contradicted
//! the assertion it claimed to satisfy, and the model gate could not object
//! because the missing entry made the assertion `Undetermined` to it.
//!
//! So: every `select` the ground assertions (and the array lemmas derived from
//! them) mention gets its chosen value published, from whichever theory decided
//! it — the circuit for bit-vectors, the tableau for `Int`/`Real`, the
//! congruence class for an uninterpreted sort.  `model_fmt` then renders the
//! array itself as the `store` chain over those reads, so the array's printed
//! value and its reads can no longer contradict each other.

use crate::prelude::*;
use crate::solver::Solver;
use crate::solver::array_axioms::ground_children;
use crate::solver::types::Model;
use oxiz_core::ast::{TermId, TermKind, TermManager};

impl Solver {
    /// Publish a value for every array read the assertions mention that the
    /// passes before it left unassigned.
    ///
    /// Runs only for problems that actually contain array operations (the same
    /// guard `check_core` uses for the refinement loop), so a formula without a
    /// single `select` pays one boolean test.
    ///
    /// Deterministic: the reads are collected by the pre-order ground walk the
    /// array-axiom instantiator uses and then sorted by term id, so the model
    /// does not depend on hash iteration order.
    pub(super) fn publish_array_reads(&mut self, model: &mut Model, manager: &mut TermManager) {
        if !self.has_array_ops {
            return;
        }
        let mut reads = self.collect_ground_reads(manager);
        reads.sort_unstable_by_key(|t| t.raw());
        for term in reads {
            if model.get(term).is_none()
                && let Some(value) = self.chosen_leaf_value(term, model, manager)
            {
                model.set(term, value);
            }
            // The *index* is published too, and for the same reason: an array
            // model is a set of (index, value) pairs, so a read whose index the
            // model leaves blank names no position and cannot appear in the
            // printed array at all.  `(= (bvadd (select arr i) #x01) #x06)` is
            // the case — `i` occurs nowhere but under the read, so no theory
            // ever gave it a column, and the model completed it to the sort
            // default only at *print* time, where the array renderer could not
            // see it.  Publishing the same value the completion would report
            // keeps the array's `store` chain and the constant's own line
            // consistent by construction.
            let Some(index) = manager.get(term).and_then(|t| match t.kind {
                TermKind::Select(_, index) => Some(index),
                _ => None,
            }) else {
                continue;
            };
            if model.get(index).is_some() {
                continue;
            }
            if let Some(value) = self.chosen_leaf_value(index, model, manager) {
                model.set(index, value);
                continue;
            }
            // Unconstrained: the sort default, which is exactly what
            // `Context::get_model` reports for such a constant.
            let Some(sort) = manager.get(index).map(|t| t.sort) else {
                continue;
            };
            if let Some(default) = super::ground_default_term(manager, sort) {
                model.set(index, default);
            }
        }
    }

    /// Every `select` term reachable from the assertions and from the array
    /// lemmas asserted for them, outside any binder.
    ///
    /// The lemma instances are walked too because a read-over-write consequent
    /// introduces the base read `(select base j)` — a term the user never wrote
    /// but whose value the printed array model has to agree with.
    fn collect_ground_reads(&self, manager: &TermManager) -> Vec<TermId> {
        let mut reads: Vec<TermId> = Vec::new();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut children: Vec<TermId> = Vec::new();
        let mut stack: Vec<TermId> = self
            .assertions
            .iter()
            .copied()
            .chain(self.array_axiom_instances.iter().copied())
            .collect();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = manager.get(current) else {
                continue;
            };
            if matches!(data.kind, TermKind::Select(..)) {
                reads.push(current);
            }
            children.clear();
            ground_children(&data.kind, &mut children);
            stack.extend(children.iter().copied());
        }
        reads
    }

    /// The value the search chose for an opaque leaf, asked of the theory that
    /// owns it: the bit-blasted circuit for a bit-vector, the tableau for
    /// `Int`/`Real`, the congruence class for everything else.
    ///
    /// The three sources cannot disagree about a leaf: a bit-vector leaf has no
    /// tableau column (the unsigned relaxation was retired in `#P2b-28`), a
    /// numeric leaf has no circuit, and the congruence class is consulted only
    /// when neither theory holds the term at all.
    fn chosen_leaf_value(
        &self,
        term: TermId,
        model: &Model,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let sort = manager.get(term)?.sort;
        if let Some(width) = manager.sorts.get(sort).and_then(|s| s.bitvec_width())
            && let Some(value) = self.bv.get_value_big(term)
        {
            return Some(manager.mk_bitvec(
                oxiz_core::ast::bv_wrap_unsigned(&num_bigint::BigInt::from(value), width),
                width,
            ));
        }
        let is_int = sort == manager.sorts.int_sort;
        let is_real = sort == manager.sorts.real_sort;
        if (is_int || is_real)
            && let Some(value) = self.arith.value(term)
        {
            return Some(if is_int {
                manager.mk_int(*value.numer())
            } else {
                manager.mk_real(value)
            });
        }
        self.euf_class_value(term, model, manager)
    }
}
