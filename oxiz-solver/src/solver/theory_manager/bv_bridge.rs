//! The bit-vector bridge between [`TheoryManager`] and `oxiz_theories`' bit-blasting
//! [`BvSolver`].
//!
//! Split out of `theory_manager.rs` to keep that file under the workspace's
//! 2000-line ceiling; it is the same `impl<'a> TheoryManager<'a>` block, moved
//! verbatim.
//!
//! Everything a bit-vector atom needs on its way from the CDCL(T) search into
//! the bit-blaster lives here: the sort lookup that decides whether an atom is
//! bit-vector-shaped at all, the recursive encoding of both operands, and the
//! three-way translation of the bit-blasted verdict back into a
//! [`TheoryCheckResult`]. `bv_run_check` is the funnel every bit-vector atom
//! passes through, and therefore where two soundness rules are enforced: an
//! atom the circuit was never told about (`asserted == false`) must not have
//! the bit-blaster's `Sat` read as evidence about it (U-Z10's sibling family),
//! and an exhausted or failed bit-blasted check must not read as "no conflict"
//! (U-Z12, once the embedded solver gained a budget).

use super::TheoryManager;
use crate::prelude::*;
use crate::solver::theory_bv_encode::{debug_verify_bv_circuits, encode_bv_term_recursive};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_sat::TheoryCheckResult;
use smallvec::SmallVec;

/// What one direction of the bit-vector / EUF equality exchange did (see
/// [`TheoryManager::combine_bv_with_euf`]).
pub(super) enum Exchange {
    /// A shared equality was refuted, or a check ran out of budget; the
    /// carried result is what `final_check` must return.
    Refuted(TheoryCheckResult),
    /// Something crossed — a leaf equality was asserted and the circuit
    /// re-checked, an argument pair was merged, a leaf was created, or a
    /// probe replaced the circuit's model — so the other direction must run
    /// again.
    Changed,
    /// Nothing crossed and the current model honours every known equality.
    Unchanged,
}

/// What congruence closure said about the partition of the application
/// arguments the circuit's model induces (see
/// [`TheoryManager::share_bv_equalities_with_euf`]).
enum Partition {
    /// EUF accepts the partition and every induced leaf equality holds in
    /// the model: the combined model exists.
    Consistent,
    /// EUF refutes it, and the refutation entails this lemma.
    Lemma(Lemma),
    /// EUF refutes the assignment without the tentative merges: a plain
    /// theory conflict over the carried reason terms.
    Refuted(Vec<TermId>),
    /// The pass cannot vouch for the partition (an explanation could not be
    /// produced, or no tag was available): the model is not vouched for.
    Undecided,
}

/// A clause `(∨ differ: s ≠ t) ∨ (∨ equal: l = r)` entailed by the asserted
/// atoms in `explanation`, recorded under `tag`.
struct Lemma {
    differ: Vec<(TermId, TermId)>,
    equal: Vec<(TermId, TermId)>,
    explanation: Vec<TermId>,
    tag: TermId,
}

impl TheoryManager<'_> {
    /// Look up the BV bit-width of a term from its sort, if it has a BV sort.
    pub(super) fn bv_width_of(&self, term: TermId, manager: &TermManager) -> Option<u32> {
        manager
            .get(term)
            .and_then(|t| manager.sorts.get(t.sort))
            .and_then(|s| s.bitvec_width())
    }

    /// Bit-blast both operands of a BV constraint into the embedded SAT solver.
    ///
    /// Returns `true` if both operands are BV-sorted with equal width and both
    /// were fully encoded, so that `assert_eq` / `assert_neq` may be called.
    ///
    /// A `false` from the encoder is **not** answered with a fresh free
    /// bit-vector for the whole operand any more (`#P2b-24`).  That fallback
    /// abstracted a term whose semantics the solver knows — a `bvsub` over an
    /// `ite` — into unconstrained bits, so the circuit admitted models the
    /// term forbids; a release build published them as `sat` until the model
    /// gate refused them, and a debug build tripped the circuit self-check.
    /// The encoder now abstracts only opaque leaves itself, so what is left
    /// when it fails is structure this solver cannot model, and the honest
    /// answer is to record the atom on [`Self::bv_atom_unmodelled`] — the
    /// owning `Solver` then degrades a final `Sat` to `Unknown` — and assert
    /// nothing.  A width mismatch is recorded the same way: an ill-sorted
    /// atom is not one the circuit can be trusted about either.
    pub(super) fn bit_blast_bv_pair(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        manager: &TermManager,
    ) -> bool {
        match (
            self.bv_width_of(lhs, manager),
            self.bv_width_of(rhs, manager),
        ) {
            (Some(lw), Some(rw)) if lw == rw => {}
            (Some(_), Some(_)) => {
                self.bv_atom_unmodelled = true;
                return false;
            }
            _ => return false,
        }
        let mut encoded: FxHashSet<TermId> = FxHashSet::default();
        let lhs_ok = encode_bv_term_recursive(self.bv, lhs, manager, &mut encoded);
        let rhs_ok = lhs_ok && encode_bv_term_recursive(self.bv, rhs, manager, &mut encoded);
        // Whatever was encoded, the opaque leaves it abstracted are live
        // circuits now, and congruence closure must know them (`#P2b-29`).
        self.intern_opaque_leaves(manager);
        if !(lhs_ok && rhs_ok) {
            self.bv_atom_unmodelled = true;
            return false;
        }
        true
    }

    /// Intern every opaque leaf the bit-blaster currently abstracts (see
    /// [`BvSolver::opaque_leaves`](oxiz_theories::bv::BvSolver::opaque_leaves))
    /// into congruence closure, so that `f(a)` buried under a `bvadd` is a
    /// node EUF can close over when `a = b` is asserted.
    ///
    /// Called right after every encoding, so a leaf's EUF node is created in
    /// the same theory scope as its circuit and the two are retracted by the
    /// same `pop`.  Idempotent through `EufSolver::term_to_node`, and cheap:
    /// the list holds only the leaves, never the circuit's interior.
    pub(super) fn intern_opaque_leaves(&mut self, manager: &TermManager) {
        let leaves: SmallVec<[TermId; 8]> = self.bv.opaque_leaves().iter().copied().collect();
        for term in leaves {
            self.intern_term_for_congruence(term, manager);
        }
    }

    /// Bidirectional Nelson-Oppen equality exchange between congruence
    /// closure and the bit-blasted circuit, run by `final_check` before a
    /// full assignment is accepted (`#P2b-29`).
    ///
    /// Two directions, each of which was a wrong `sat` on its own:
    ///
    /// * **EUF → bit-vector** ([`Self::share_euf_equalities_with_bv`]): an
    ///   uninterpreted application or array `select` of bit-vector sort has
    ///   no circuit and is abstracted into a free bit-vector (an *opaque
    ///   leaf*, `BvSolver::new_opaque_leaf`).  Congruence closure owns the
    ///   equalities between such leaves — `f(a) = f(b)` once `a = b` — and
    ///   until this pass nothing carried them into the circuit, so `(= a b)
    ///   ∧ (distinct (bvadd (f a) #x01) (bvadd (f b) #x01))` answered `sat`.
    /// * **bit-vector → EUF** ([`Self::share_bv_equalities_with_euf`]): the
    ///   circuit entails equalities between the bit-vector terms that occur
    ///   as *arguments* of applications, and congruence closure only fires
    ///   on argument equalities it is told about, so `(= (bvadd x #x01)
    ///   (bvadd y #x01)) ∧ (distinct (g x) (g y))` answered `sat`.
    ///
    /// The two feed each other — a merged argument pair creates a
    /// congruence, a shared leaf equality changes the circuit's model — so
    /// they alternate until a round changes nothing.  Every equality that
    /// crosses in either direction is a *deduction* with a recorded
    /// explanation, never a guess (see the two methods).  Cost on a problem
    /// without function applications: one `has_app_nodes` test, which is
    /// what keeps this free for cargo-formal's QF_BV goals.
    ///
    /// Returns `Some(Conflict(..))` when a shared equality is refuted (the
    /// clause names the atoms, pins and justifications it rests on),
    /// `Some(Sat)` with `resource_exhausted` set when a check ran out of
    /// budget, and `None` when the assignment is consistent with every
    /// derived equality — or when the exchange could not decide it and
    /// [`Self::bv_euf_undecided`] was set, which the owning `Solver` reads as
    /// `unknown`, never `sat`.
    pub(super) fn combine_bv_with_euf(
        &mut self,
        manager: &TermManager,
    ) -> Option<TheoryCheckResult> {
        if !self.euf.has_app_nodes() {
            return None;
        }
        // A guard only: a round that continues has asserted a new leaf
        // equality or merged two EUF classes, and both are bounded by the
        // number of nodes.
        const MAX_COMBINE_ROUNDS: usize = 64;
        for _ in 0..MAX_COMBINE_ROUNDS {
            let leaves = match self.share_euf_equalities_with_bv() {
                Exchange::Refuted(result) => return Some(result),
                Exchange::Changed => true,
                Exchange::Unchanged => false,
            };
            let arguments = match self.share_bv_equalities_with_euf(manager) {
                Exchange::Refuted(result) => return Some(result),
                Exchange::Changed => true,
                Exchange::Unchanged => false,
            };
            if !leaves && !arguments {
                return None;
            }
        }
        self.bv_euf_undecided = true;
        None
    }

    /// EUF → bit-vector equality sharing for the circuit's opaque leaves
    /// (`#P2b-29`); one direction of [`Self::combine_bv_with_euf`].
    ///
    /// Model-based, the Nelson-Oppen way: the leaves are bucketed by EUF
    /// representative, and two leaves of one class whose circuit values
    /// *disagree* have their bit-equality asserted — with EUF's explanation
    /// of the equality recorded in [`DerivedReasons`](super::DerivedReasons)
    /// under a tag the conflict clause expands back into the literals that
    /// justify it ([`Self::assert_explained_bv_equality`]) — and the circuit
    /// is re-checked.  A class whose members already agree needs nothing:
    /// the current model honours the equality.  Each round asserts at least
    /// one new equality or ends the loop, and an asserted pair agrees in
    /// every later model, so the rounds are bounded by the number of leaf
    /// pairs.
    ///
    /// Before this pass, `(= a b) ∧ (distinct (bvadd (f a) #x01) (bvadd (f b)
    /// #x01))` answered `sat`: EUF saw two `bvadd` terms it could not
    /// relate, the circuit saw two free vectors, and the model gate could
    /// not evaluate `f`.  (The bare pair `(distinct (f a) (f b))` was always
    /// refuted, by EUF alone.)
    ///
    /// [`Exchange::Refuted`] carries a conflict (the explanation names the
    /// constraint atoms, the pinned selectors and the equality's own
    /// justification) or `Sat` with `resource_exhausted` set when a re-check
    /// ran out of budget; [`Exchange::Changed`] means at least one equality
    /// was asserted and the circuit re-checked; [`Exchange::Unchanged`] means
    /// the current model already honours every derived leaf equality.
    pub(super) fn share_euf_equalities_with_bv(&mut self) -> Exchange {
        use oxiz_theories::Theory;
        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
        if self.bv.opaque_leaves().is_empty() || !self.euf.has_app_nodes() {
            return Exchange::Unchanged;
        }
        // A guard only: the loop ends on its own when a round asserts nothing
        // (see the doc), and a pair EUF cannot explain is skipped, so it can
        // never keep a round alive.
        const MAX_SHARING_ROUNDS: usize = 64;
        let mut changed = false;
        for _ in 0..MAX_SHARING_ROUNDS {
            // Bucket by representative, in leaf-creation order, so which
            // member becomes the witness — and hence which equality is
            // asserted and tagged — never depends on hash iteration order.
            let leaves: Vec<TermId> = self.bv.opaque_leaves().to_vec();
            let mut by_class: FxHashMap<u32, SmallVec<[TermId; 4]>> = FxHashMap::default();
            let mut class_order: Vec<u32> = Vec::new();
            for term in leaves {
                let Some(node) = self.euf.term_to_node(term) else {
                    continue;
                };
                let root = self.euf.find(node);
                let members = by_class.entry(root).or_insert_with(|| {
                    class_order.push(root);
                    SmallVec::new()
                });
                members.push(term);
            }
            let mut asserted_any = false;
            for root in class_order {
                let Some(members) = by_class.get(&root) else {
                    continue;
                };
                let (Some(&first), rest) = (members.first(), members.get(1..).unwrap_or(&[]))
                else {
                    continue;
                };
                let first_value = self.bv.get_value_big(first);
                for &other in rest {
                    if self.bv.get_value_big(other) == first_value {
                        continue;
                    }
                    if self.assert_explained_bv_equality(first, other) {
                        asserted_any = true;
                    }
                }
            }
            if !asserted_any {
                return if changed {
                    Exchange::Changed
                } else {
                    Exchange::Unchanged
                };
            }
            changed = true;
            self.bv_pin_pending = false;
            match self.bv.check() {
                Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) => {
                    return Exchange::Refuted(self.report_theory_conflict(conflict_terms));
                }
                Ok(TheoryCheckResultEnum::Sat) | Ok(TheoryCheckResultEnum::Propagate(_)) => {}
                Ok(TheoryCheckResultEnum::Unknown) | Err(_) => {
                    self.resource_exhausted = true;
                    return Exchange::Refuted(TheoryCheckResult::Sat);
                }
            }
        }
        Exchange::Changed
    }

    /// Assert the bit-equality of two opaque leaves congruence closure holds
    /// equal, carrying EUF's explanation, and report whether anything reached
    /// the circuit.
    ///
    /// The crossing point from congruence closure into the bit-blaster,
    /// upholding the same two invariants as `assert_explained_equality`
    /// does for the tableau: an equality EUF cannot explain is **not
    /// propagated** (a lost propagation costs completeness; an
    /// unexplainable fact in a conflict costs soundness), and one that is
    /// propagated has its explanation recorded under a tag that names no
    /// literal — the leaf itself, which is bit-vector sorted and so has no
    /// SAT variable of its own.  That tag is also recorded on the circuit as
    /// a constraint term, so every explanation the embedded solver hands
    /// back includes it and `terms_to_conflict_clause` expands it into the
    /// equality atoms it stands for.
    fn assert_explained_bv_equality(&mut self, lhs: TermId, rhs: TermId) -> bool {
        let (Some(n1), Some(n2)) = (self.euf.term_to_node(lhs), self.euf.term_to_node(rhs)) else {
            return false;
        };
        let justification = self.euf.explain_eq(n1, n2);
        if n1 != n2 && justification.is_empty() {
            return false;
        }
        // `terms_to_conflict_clause` resolves a term that names a SAT
        // variable as that atom's literal *before* it looks for a derived
        // explanation, so the tag must be a term without one.  A bit-vector
        // sorted leaf never has one on a well-sorted input; if both did, the
        // equality is left to the pins the outer assignment replays.
        let Some(tag) = self.derived_tag_for(lhs, rhs) else {
            return false;
        };
        if !self.bv.assert_eq(lhs, rhs) {
            return false;
        }
        self.derived_reasons.record(tag, justification);
        self.bv.record_constraint_term(tag);
        true
    }

    /// Bit-vector → EUF: the circuit's model partitions the application
    /// arguments, and congruence closure must accept that partition
    /// (`#P2b-29`; the other direction of [`Self::combine_bv_with_euf`]).
    ///
    /// The candidates are the bit-vector-sorted terms that occur as an
    /// argument of some application or `select` congruence closure holds —
    /// the only terms whose equality can fire a congruence.  Two candidates
    /// the circuit's current model gives the same value while EUF holds them
    /// in different classes are the gap: the model says they are equal and
    /// EUF has not been told.  Rather than probing each pair, the pass asks
    /// EUF about the whole partition at once, inside a scratch EUF scope: it
    /// merges every model-equal pair tentatively and checks for a conflict,
    /// and — when there is none — whether the classes the tentative merges
    /// induce agree with the model on the circuit's opaque leaves (`f(s) =
    /// f(t)` must hold in the model once `s = t` does).
    ///
    /// * EUF accepts the partition and the induced leaf equalities hold:
    ///   the combined model exists, nothing crosses, and the answer is
    ///   [`Exchange::Unchanged`] (or `Changed` if an earlier lemma moved the
    ///   model).
    /// * EUF refutes it: its explanation names the tentative merges `s_i =
    ///   t_i` that took part and the asserted atoms `A` it used, so
    ///   `A ⊨ ∨_i s_i ≠ t_i` — a **lemma**, entailed by the assignment, never
    ///   a guess.  It is asserted into the circuit as one clause
    ///   (`BvSolver::assert_any`), recorded in
    ///   [`DerivedReasons`](super::DerivedReasons) under a tag that expands
    ///   to `A`, and the circuit is re-checked: the next model must separate
    ///   one of the pairs, or the circuit refutes the lemma and the conflict
    ///   clause names `A` together with the circuit's own hypotheses.
    /// * EUF accepts it but two leaves it now holds equal differ in the
    ///   model: `A ∧ ∧_i s_i = t_i ⊨ l = r` for the merges on the proof
    ///   path, so `A ⊨ (∨_i s_i ≠ t_i) ∨ l = r` is the lemma, handled the
    ///   same way.
    ///
    /// Each lemma is violated by the model it was derived from, so every
    /// round moves the model; the number of rounds is bounded (`MAX_LEMMAS`)
    /// and running out sets [`Self::bv_euf_undecided`] (`unknown`).  The
    /// bounded tests never reach it (their longest loop is 208 rounds); the
    /// 1,600-script campaigns in `oxiz-solver/tests/bv_euf_combination.rs`
    /// reach it on a handful of width-3 QF_ABV scripts nesting `select` four
    /// to six deep, which enumerate the partitions of ~15 argument terms
    /// one lemma at a time — under a temporary 8,192-round cap two of them
    /// decide `sat` at rounds 2,030 and 1,645.  What a give-up costs is
    /// bounded by `BvSolver`'s `eq_cache`: before it every lemma re-encoded
    /// its pair disequalities and the embedded instance grew with the round
    /// count (52–82 s to reach the cap), with it a round costs a few
    /// milliseconds and the cap is reached in 1.4–1.8 s.  Making the
    /// argument equalities atoms of the outer search, so CDCL splits and
    /// learns over them instead of this loop, is the real remedy and is
    /// recorded open under `#P2b-29`.  An argument with no circuit yet (one
    /// that occurs in no bit-vector atom) is bit-blasted first, at the
    /// current theory scope; one the encoder cannot model marks the atom
    /// set unmodelled.  This is what keeps `(distinct (g x) (g y))` over two
    /// free variables `sat`: the first model has `x = y = 0`, EUF refutes
    /// `x = y` under `g(x) ≠ g(y)`, the lemma `x ≠ y` is asserted, and the
    /// next model — the one `build_model` publishes — separates them.
    pub(super) fn share_bv_equalities_with_euf(&mut self, manager: &TermManager) -> Exchange {
        use oxiz_theories::Theory;
        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;

        let forbidden_func_ids: FxHashSet<u32> = self
            .quantifier_uf_funcs
            .iter()
            .map(|spur| spur.into_inner().get())
            .collect();
        let mut candidates: Vec<TermId> = self
            .euf
            .app_argument_terms_excluding_funcs(&forbidden_func_ids)
            .into_iter()
            .filter(|&term| self.bv_width_of(term, manager).is_some())
            .collect();
        if candidates.len() < 2 {
            return Exchange::Unchanged;
        }
        // Deterministic order: which member of a value bucket becomes the
        // witness decides which pairs are merged and which lemma is derived.
        candidates.sort_unstable_by_key(|t| t.raw());

        // Every candidate needs a circuit for its value to mean anything.
        let leaves_before = self.bv.opaque_leaves().len();
        let mut encoded: FxHashSet<TermId> = FxHashSet::default();
        for &term in &candidates {
            if self.bv.get_bv(term).is_some() {
                continue;
            }
            if !encode_bv_term_recursive(self.bv, term, manager, &mut encoded) {
                self.bv_atom_unmodelled = true;
                return Exchange::Unchanged;
            }
        }
        self.intern_opaque_leaves(manager);
        let mut changed = self.bv.opaque_leaves().len() != leaves_before;
        // A circuit built just now has no value in the model snapshot the
        // last check left (its bits read as `0`, a constant's pinned bits
        // included), so the partition below would be read off nothing: one
        // check gives every candidate a real value first.
        if !encoded.is_empty() {
            changed = true;
            self.bv_pin_pending = false;
            if self.charge_bv_embedded_check() {
                self.resource_exhausted = true;
                return Exchange::Refuted(TheoryCheckResult::Sat);
            }
            match self.bv.check() {
                Ok(TheoryCheckResultEnum::Sat) | Ok(TheoryCheckResultEnum::Propagate(_)) => {}
                Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) => {
                    return Exchange::Refuted(self.report_theory_conflict(conflict_terms));
                }
                Ok(TheoryCheckResultEnum::Unknown) | Err(_) => {
                    self.resource_exhausted = true;
                    return Exchange::Refuted(TheoryCheckResult::Sat);
                }
            }
        }

        // A guard only: every lemma is violated by the model it came from,
        // so each round moves the model, and the lemmas are pairwise
        // distinct clauses over a finite set of pairs.  Generous because a
        // narrow domain (width 1 or 2) with many arguments has many
        // partitions to rule out one lemma at a time.
        const MAX_LEMMAS: usize = 8192;
        for _ in 0..MAX_LEMMAS {
            let lemma = match self.model_partition_lemma(&candidates) {
                Partition::Consistent => {
                    return if changed {
                        Exchange::Changed
                    } else {
                        Exchange::Unchanged
                    };
                }
                Partition::Refuted(explanation) => {
                    return Exchange::Refuted(self.report_theory_conflict(explanation));
                }
                Partition::Undecided => {
                    self.bv_euf_undecided = true;
                    return Exchange::Unchanged;
                }
                Partition::Lemma(lemma) => lemma,
            };
            if !self.bv.assert_any(&lemma.differ, &lemma.equal) {
                self.bv_euf_undecided = true;
                return Exchange::Unchanged;
            }
            self.derived_reasons
                .record(lemma.tag, lemma.explanation.iter().copied());
            self.bv.record_constraint_term(lemma.tag);
            changed = true;
            self.bv_pin_pending = false;
            if self.charge_bv_embedded_check() {
                self.resource_exhausted = true;
                return Exchange::Refuted(TheoryCheckResult::Sat);
            }
            match self.bv.check() {
                Ok(TheoryCheckResultEnum::Sat) | Ok(TheoryCheckResultEnum::Propagate(_)) => {}
                Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) => {
                    return Exchange::Refuted(self.report_theory_conflict(conflict_terms));
                }
                Ok(TheoryCheckResultEnum::Unknown) | Err(_) => {
                    self.resource_exhausted = true;
                    return Exchange::Refuted(TheoryCheckResult::Sat);
                }
            }
        }
        self.bv_euf_undecided = true;
        Exchange::Unchanged
    }

    /// Ask congruence closure, in a scratch scope, whether it accepts the
    /// partition of `candidates` the circuit's current model induces (see
    /// [`Self::share_bv_equalities_with_euf`]), and turn a refusal into the
    /// lemma it entails.
    fn model_partition_lemma(&mut self, candidates: &[TermId]) -> Partition {
        use oxiz_theories::Theory;

        // The model-equal pairs EUF does not already hold equal, bucketed by
        // value: the first member of a bucket is merged with every other
        // member, which is the whole bucket by transitivity.
        let mut by_value: FxHashMap<(u32, num_bigint::BigUint), SmallVec<[TermId; 4]>> =
            FxHashMap::default();
        let mut bucket_order: Vec<(u32, num_bigint::BigUint)> = Vec::new();
        for &term in candidates {
            let Some(width) = self.bv_width_of(term, self.manager) else {
                continue;
            };
            let Some(value) = self.bv.get_value_big(term) else {
                continue;
            };
            let key = (width, value);
            let members = by_value.entry(key.clone()).or_insert_with(|| {
                bucket_order.push(key);
                SmallVec::new()
            });
            members.push(term);
        }
        let mut pairs: Vec<(TermId, TermId)> = Vec::new();
        for key in &bucket_order {
            let Some(members) = by_value.get(key) else {
                continue;
            };
            let Some(&first) = members.first() else {
                continue;
            };
            for &other in members.iter().skip(1) {
                let (Some(n1), Some(n2)) =
                    (self.euf.term_to_node(first), self.euf.term_to_node(other))
                else {
                    continue;
                };
                if self.euf.find(n1) == self.euf.find(n2) {
                    continue;
                }
                // Two bit-vector literals of one value under two term ids are
                // equal in every model: merge them for good, under the
                // tautological reason the constant interning uses, rather
                // than treat the pair as a model coincidence.
                if self.is_bv_constant(first) && self.is_bv_constant(other) {
                    self.tautological_reasons.insert(first);
                    let _ = self.euf.merge(n1, n2, first);
                    continue;
                }
                pairs.push((first, other));
            }
        }
        if pairs.is_empty() {
            return Partition::Consistent;
        }

        // Tentative merges, each tagged with one of its own endpoints — a
        // term that names no literal — so EUF's explanation can be read back
        // into the pairs it rests on.  A tag shared by two pairs (an
        // endpoint common to both) makes both suspects, which only weakens
        // the lemma.
        let mut used: FxHashSet<TermId> = FxHashSet::default();
        let mut tagged: Vec<(TermId, TermId, TermId)> = Vec::with_capacity(pairs.len());
        for &(s, t) in &pairs {
            let Some(tag) = self.tentative_tag_for(s, t, &used) else {
                return Partition::Undecided;
            };
            used.insert(tag);
            tagged.push((tag, s, t));
        }
        self.euf.push();
        for &(tag, s, t) in &tagged {
            let (Some(n1), Some(n2)) = (self.euf.term_to_node(s), self.euf.term_to_node(t)) else {
                continue;
            };
            if self.euf.merge(n1, n2, tag).is_err() {
                self.euf.pop();
                return Partition::Undecided;
            }
        }
        let outcome = match self.euf.check_conflicts() {
            Some(explanation) => self.lemma_from_explanation(&tagged, explanation, None),
            None => self.induced_leaf_lemma(&tagged),
        };
        self.euf.pop();
        outcome
    }

    /// With the tentative merges in place and no EUF conflict, find two
    /// opaque leaves congruence closure now holds equal whose circuit values
    /// differ, and derive the lemma that separates one of the merged pairs
    /// or equates the leaves.
    fn induced_leaf_lemma(&mut self, tagged: &[(TermId, TermId, TermId)]) -> Partition {
        let leaves: Vec<TermId> = self.bv.opaque_leaves().to_vec();
        let mut by_class: FxHashMap<u32, SmallVec<[TermId; 4]>> = FxHashMap::default();
        let mut class_order: Vec<u32> = Vec::new();
        for term in leaves {
            let Some(node) = self.euf.term_to_node(term) else {
                continue;
            };
            let root = self.euf.find(node);
            let members = by_class.entry(root).or_insert_with(|| {
                class_order.push(root);
                SmallVec::new()
            });
            members.push(term);
        }
        for root in class_order {
            let Some(members) = by_class.get(&root) else {
                continue;
            };
            let Some(&first) = members.first() else {
                continue;
            };
            let first_value = self.bv.get_value_big(first);
            for &other in members.iter().skip(1) {
                if self.bv.get_value_big(other) == first_value {
                    continue;
                }
                let (Some(n1), Some(n2)) =
                    (self.euf.term_to_node(first), self.euf.term_to_node(other))
                else {
                    continue;
                };
                let Some(explanation) = self.euf.try_explain_eq(n1, n2) else {
                    return Partition::Undecided;
                };
                return self.lemma_from_explanation(tagged, explanation, Some((first, other)));
            }
        }
        Partition::Consistent
    }

    /// Read an EUF explanation that may rest on tentative merges back into a
    /// lemma: the merged pairs it names must not all hold, or (when given)
    /// the two leaves must be equal.  The lemma's own justification is the
    /// explanation with the tentative tags removed — except a tag that also
    /// carries a recorded explanation or is a registered tautology, which is
    /// kept: dropping it could lose a genuine justification, keeping it can
    /// only lengthen the clause.
    fn lemma_from_explanation(
        &self,
        tagged: &[(TermId, TermId, TermId)],
        explanation: Vec<TermId>,
        equal: Option<(TermId, TermId)>,
    ) -> Partition {
        let mut differ: Vec<(TermId, TermId)> = Vec::new();
        let mut justification: Vec<TermId> = Vec::with_capacity(explanation.len());
        for term in explanation {
            let mut tentative = false;
            for &(tag, s, t) in tagged {
                if tag == term {
                    tentative = true;
                    if !differ.contains(&(s, t)) {
                        differ.push((s, t));
                    }
                }
            }
            let keep = !tentative
                || self.derived_reasons.literals(term).is_some()
                || self.tautological_reasons.contains(&term);
            if keep && !justification.contains(&term) {
                justification.push(term);
            }
        }
        let equal: Vec<(TermId, TermId)> = equal.into_iter().collect();
        if differ.is_empty() && equal.is_empty() {
            // EUF refuted the assignment without the tentative merges: a
            // plain theory conflict, explained by what it named.
            return Partition::Refuted(justification);
        }
        let tag = differ
            .first()
            .and_then(|&(s, t)| self.derived_tag_for(s, t))
            .or_else(|| equal.first().and_then(|&(l, r)| self.derived_tag_for(l, r)));
        let Some(tag) = tag else {
            return Partition::Undecided;
        };
        Partition::Lemma(Lemma {
            differ,
            equal,
            explanation: justification,
            tag,
        })
    }

    /// The reason term for a tentative merge of `s` and `t`: an endpoint
    /// that names no literal, is not a bit-vector literal (a constant is the
    /// tautological reason of its own canonical merges), and — when both
    /// qualify — has not tagged another pair of this round.
    fn tentative_tag_for(&self, s: TermId, t: TermId, used: &FxHashSet<TermId>) -> Option<TermId> {
        let eligible =
            |term: TermId| !self.term_to_var.contains_key(&term) && !self.is_bv_constant(term);
        [s, t]
            .into_iter()
            .filter(|&term| eligible(term))
            .min_by_key(|term| used.contains(term))
    }

    /// Whether `term` is a bit-vector literal.
    fn is_bv_constant(&self, term: TermId) -> bool {
        self.manager
            .get(term)
            .is_some_and(|x| matches!(x.kind, TermKind::BitVecConst { .. }))
    }

    /// The term a derived bit-vector equality `lhs = rhs` (or a lemma over
    /// it) is recorded under: one that names no SAT variable, so
    /// `terms_to_conflict_clause` looks up its explanation instead of a
    /// literal, preferring a non-constant.  `None` when both sides name an
    /// atom, which cannot happen for bit-vector-sorted terms of a
    /// well-sorted input.
    fn derived_tag_for(&self, lhs: TermId, rhs: TermId) -> Option<TermId> {
        let has_literal = |t: TermId| self.term_to_var.contains_key(&t);
        [lhs, rhs]
            .into_iter()
            .filter(|&t| !has_literal(t))
            .min_by_key(|&t| self.is_bv_constant(t))
    }

    /// Run the embedded BV SAT check after the caller has asserted a constraint.
    ///
    /// Records `constraint_term` so the conflict clause is non-empty, then
    /// returns `Some(Conflict(..))` if the embedded solver reports UNSAT and
    /// `None` otherwise (so the caller falls through to its conservative path).
    ///
    /// `operands` are the two sides of the atom just asserted.  When the check
    /// comes back SAT they are handed to [`debug_verify_bv_circuits`], the
    /// debug-only model-validity net: every bit-blasted node under them must
    /// reproduce its own operation concretely on the model the solver just
    /// found.  That is the check which distinguishes "the search is right" from
    /// "the circuit is wrong", and it costs nothing in release builds.
    ///
    /// `asserted` is the caller's own answer to "did an `assert_*` actually
    /// reach the circuit for this atom".  When it is `false` the circuit was
    /// never *told* about the atom, so its `Sat` is not evidence about it: the
    /// atom is recorded on [`Self::bv_atom_unmodelled`] and this returns `None`,
    /// letting the CDCL(T) loop continue while the owning `Solver` degrades a
    /// final `Sat` to `Unknown`.  The constraint term is deliberately *not*
    /// recorded — no clause depends on it, so it does not belong in a conflict
    /// explanation.
    pub(super) fn bv_run_check(
        &mut self,
        constraint_term: TermId,
        operands: (TermId, TermId),
        manager: &TermManager,
        asserted: bool,
    ) -> Option<TheoryCheckResult> {
        use oxiz_theories::Theory;
        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
        if !asserted {
            self.bv_atom_unmodelled = true;
            return None;
        }
        self.bv.record_constraint_term(constraint_term);
        // Deterministic budget: one embedded check is one unit of it, and an
        // exhausted budget stops running them rather than reporting a verdict
        // from work it did not do (`#P2b-46`).
        if self.charge_bv_embedded_check() {
            self.resource_exhausted = true;
            return None;
        }
        self.bv_pin_pending = false;
        match self.bv.check() {
            Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) => {
                Some(self.conflict_from_terms(&conflict_terms))
            }
            Ok(TheoryCheckResultEnum::Sat) => {
                debug_verify_bv_circuits(self.bv, operands.0, manager);
                debug_verify_bv_circuits(self.bv, operands.1, manager);
                None
            }
            // A propagation carries no verdict about this atom; fall through to
            // the caller's conservative path exactly as before.
            Ok(TheoryCheckResultEnum::Propagate(_)) => None,
            // An exhausted or failed bit-blasted check is NOT "no conflict".
            //
            // Before the embedded solver had a budget these two were
            // unreachable, and a `_ => None` arm was harmless.  The moment
            // `(set-option :timeout N)` / `(set-option :max-conflicts N)` can
            // stop an embedded `solve()` (U-Z12), reading its `Unknown` as
            // "consistent" would let the CDCL(T) loop fall through to
            // `TheoryCheckResult::Sat` at the tail of `process_constraint` and
            // report a `sat` that no bit-blasted check ever confirmed.  Setting
            // `resource_exhausted` and answering `Some(Sat)` stops the search
            // the way the conflict-limit path already does, and the flag makes
            // the owning `Solver` answer `unknown`.  `Unsat` is unaffected:
            // dropping a theory conflict only weakens the clause set, and
            // `Unsat` of a weakening implies `Unsat` of the original.
            //
            // Spelled out per variant rather than left as `_` so a new
            // `TheoryCheckResult` variant is a compile error here instead of
            // silently inheriting whichever behaviour the wildcard had.
            Ok(TheoryCheckResultEnum::Unknown) | Err(_) => {
                self.resource_exhausted = true;
                Some(TheoryCheckResult::Sat)
            }
        }
    }

    /// Consult the embedded solver after an outer Boolean assignment was
    /// pinned into a live boolean node (see `BvSolver::assert_bool_value`);
    /// run by `final_check` when `bv_pin_pending` says a pin has not been
    /// examined since.
    ///
    /// `Some(Conflict(..))` when the pin refutes the circuit — the
    /// explanation is `BvSolver::collect_conflict_terms`, which names the
    /// pinned atoms — and `None` when the circuit stays satisfiable.  An
    /// exhausted or failed check is handled exactly as in
    /// [`Self::bv_run_check`]: it is not "no conflict".
    pub(super) fn bv_check_after_pin(&mut self) -> Option<TheoryCheckResult> {
        use oxiz_theories::Theory;
        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
        self.bv_pin_pending = false;
        if self.charge_bv_embedded_check() {
            self.resource_exhausted = true;
            return Some(TheoryCheckResult::Sat);
        }
        match self.bv.check() {
            Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) => {
                Some(self.conflict_from_terms(&conflict_terms))
            }
            Ok(TheoryCheckResultEnum::Sat) | Ok(TheoryCheckResultEnum::Propagate(_)) => None,
            Ok(TheoryCheckResultEnum::Unknown) | Err(_) => {
                self.resource_exhausted = true;
                Some(TheoryCheckResult::Sat)
            }
        }
    }

    /// Bit-blast `lhs`/`rhs`, assert `lhs != b` at the bit level, and check.
    ///
    /// Returns `Some(Conflict(..))` on a detected BV theory conflict, `None`
    /// otherwise (including when the operands are not equal-width BV terms).
    pub(super) fn bv_check_neq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        constraint_term: TermId,
        manager: &TermManager,
    ) -> Option<TheoryCheckResult> {
        if !self.bit_blast_bv_pair(lhs, rhs, manager) {
            return None;
        }
        let asserted = self.bv.assert_neq(lhs, rhs);
        self.bv_run_check(constraint_term, (lhs, rhs), manager, asserted)
    }

    /// Bit-blast `lhs`/`rhs`, assert `lhs = b` at the bit level, and check.
    ///
    /// Returns `Some(Conflict(..))` on a detected BV theory conflict, `None`
    /// otherwise (including when the operands are not equal-width BV terms).
    pub(super) fn bv_check_eq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        constraint_term: TermId,
        manager: &TermManager,
    ) -> Option<TheoryCheckResult> {
        if !self.bit_blast_bv_pair(lhs, rhs, manager) {
            return None;
        }
        let asserted = self.bv.assert_eq(lhs, rhs);
        self.bv_run_check(constraint_term, (lhs, rhs), manager, asserted)
    }
}
