//! Lazy array-theory axiom instantiation for the CDCL(T) loop.
//!
//! The syntactic pre-checks in [`super::check_array`] recognise a fixed set of
//! definite array conflicts, but they cannot decide the general case — e.g. a
//! read-over-write at a *provably different* index (`i != j` forcing
//! `select(store(a,i,v),j) = select(a,j)`), or extensionality on a disequality
//! between two array variables.  Left to the raw SAT core those atoms are free
//! Booleans, which risks a spurious `Sat`.
//!
//! This module supplies the missing decision power as a *lazy* refinement loop
//! driven from [`super::Solver::check`]: whenever the CDCL(T) core proposes a
//! candidate model, [`Solver::instantiate_array_axioms`] inspects the array
//! terms in that model and, for every array axiom instance the candidate does
//! not already satisfy, asserts the corresponding ground lemma and asks the
//! core to re-solve.  The three axiom families are:
//!
//!   * **Read-over-write** — for every `select(store(b,i,v), j)` (directly or
//!     through an asserted `B = store(b,i,v)` alias):
//!     `select(store(b,i,v),j) = ite(i = j, v, select(b,j))`.
//!   * **Extensionality** — for every array-sorted equality atom `a = b`, a
//!     witness index `k` (fresh but *deterministic* per unordered pair) with
//!     `a = b  ∨  select(a,k) != select(b,k)`.  When `a != b` is asserted this
//!     forces a concrete differing index.
//!   * **Select congruence** — for every array-sorted equality atom `a = b`
//!     and every index `j` read on either side:
//!     `a = b  ⇒  select(a,j) = select(b,j)`.
//!
//! Every asserted instance is a theorem of the (extensional) array theory, so
//! adding it never changes satisfiability — it only removes models that violate
//! array semantics.  Instances are deduplicated by their interned lemma term
//! id, and the reachable instance set is finite (bounded by the store-subterm ×
//! index-set product plus one witness per array pair), so the refinement loop
//! in `check` terminates: each round either asserts a strictly new instance or
//! reports that the candidate model is a genuine array model.
//!
//! The structural walk that finds those `select`s is exhaustive over the
//! ground term language (`ground_children`, delegating to
//! `term_walk::collect_structural_children`).  It was not until `#P2b-32`:
//! a hand-written child list covered only the Boolean connectives, `ite` and
//! `Apply`, so a read nested under a bit-vector or arithmetic operator —
//! `(bvadd (select (store arr i #x05) i) #x01)`, `(+ (select (store arr i 5)
//! i) 1)` — was never collected, no read-over-write instance was ever built
//! for it, and the leaf stayed a free bit-vector in the circuit or a free
//! column in the tableau: `(distinct (bvadd (select (store arr i #x05) i)
//! #x01) #x06)` answered `sat` on 0.3.3 and on every tree before the fix,
//! while the same read as a *direct* atom operand was decided.  The model
//! gate is the second line of defence for that family: it reads a `select`
//! over a `store` as read-over-write (`model_eval.rs`, `Op::Select`).
//!
//! Reference: Z3's `smt/theory_array.cpp` semantics (read-over-write and
//! extensionality axiom instantiation).

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::SortKind;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;

use super::{EvalVal, Solver};

/// Safety valve on the number of distinct array-axiom instances asserted across
/// a single `check`.  Deduplication is the real termination mechanism; this cap
/// only guards against pathological growth (deeply nested store chains crossed
/// with many array pairs) so a malformed input cannot make the refinement loop
/// consume unbounded memory.  Realistic array benchmarks add a handful of
/// instances.
///
/// Reaching it sets [`Solver::array_axioms_incomplete`], because the two ways
/// this function returns `false` mean opposite things: "the candidate model
/// satisfies every axiom" (a `Sat` may be reported) versus "the budget stopped
/// me looking" (it may not).  `check_core` cannot tell them apart from the
/// return value, so the flag carries the difference.
const MAX_ARRAY_AXIOM_INSTANCES: usize = 20_000;

/// Safety valve on the number of distinct *witness index* variables the
/// extensionality families mint across a single `check`.
///
/// Every witness is a fresh index, and a witness read of an array whose range
/// is itself an array is a new array term — which can enter a new pair, which
/// mints another witness.  The nesting depth bounds how *deep* that goes, but
/// not how *wide*: a formula over concrete arrays of arrays kept finding new
/// pairs at the same depth and the lemma set grew without settling.  The cap
/// stops the generation; like [`MAX_ARRAY_AXIOM_INSTANCES`] it is recorded in
/// [`Solver::array_axioms_incomplete`], so the `Sat` it would otherwise
/// license is downgraded to `Unknown` rather than trusted.
///
/// Realistic array benchmarks mint a handful: the whole 217-script corpus
/// stays three orders of magnitude below this.
const MAX_ARRAY_EXT_WITNESSES: usize = 512;

/// Largest index-sort cardinality at which extensionality is decided by
/// *enumerating* the index domain instead of by minting a Skolem witness
/// index per array pair (decision (10), `#P2b-38` strand (b)).
///
/// # Why enumerate at all
///
/// A Skolem witness is a fresh variable of the index sort, and it is fresh
/// *per unordered pair*: an `n`-ary `distinct` over array operands mints
/// `C(n,2)` of them.  Each one is a new bit-vector argument term whose read
/// cascades a read-over-write instance down every link of both store chains,
/// and — the part that actually bites — each one joins the candidate set of
/// the BV↔EUF partition-lemma exchange, whose cost grows with the number of
/// *partitions* of that set.  Four operands of depth four over a one-bit index
/// sort took 6.8 s against the 0.3.4 base's 10.7 ms, with 4,551 theory
/// conflicts against the base's zero, and every one of those conflicts came
/// from a partition space the six fresh indices created.
///
/// When the index sort is small and finite, none of that is necessary.  Over a
/// domain `D` that the solver can write out, extensionality is a finite
/// conjunction:
///
/// ```text
/// a = b  ⟺  ⋀_{i ∈ D} select(a, i) = select(b, i)
/// ```
///
/// so the pair is decided at the domain's *own* elements — index terms that
/// are shared by every pair, already in the formula's index set, and constant,
/// so the partition exchange learns nothing new from them.  The same
/// enumeration subsumes the chain-index walk (phase 3b) and the off-chain
/// Skolem index (phase 3c) for such a pair: an off-chain index is an element
/// of `D`, and `D` is fully covered.
///
/// # Why eight
///
/// The enumerated family costs `|D|` lemmas per pair and `|D|` reads per array
/// term, against one Skolem index and its cascade.  Eight keeps the enumerated
/// cost at or below the Skolem cost for the sorts that reach it — `Bool`,
/// `(_ BitVec 1)`, `(_ BitVec 2)`, `(_ BitVec 3)` — while a wider index sort
/// keeps the Skolem machinery, whose cost does not grow with the domain.
/// Measured on this tree (2026-09-18): at eight, the four in-tree `n`-ary
/// scripts and the `s30028_31` shape all answer within a small factor of the
/// base, and the 217-script `bench/` sweep is unchanged in every verdict.
const ARRAY_INDEX_ENUMERATION_LIMIT: u128 = 8;

impl Solver {
    /// One round of lazy array-axiom instantiation against the current candidate
    /// model.  Returns `true` when at least one new ground array lemma was
    /// asserted to the SAT core — in which case the caller must re-solve — and
    /// `false` when the candidate model already satisfies every applicable
    /// axiom instance (so the reported `Sat` is trustworthy for the array
    /// atoms).
    pub(super) fn instantiate_array_axioms(&mut self, manager: &mut TermManager) -> bool {
        if self.array_axiom_instances.len() >= MAX_ARRAY_AXIOM_INSTANCES {
            // Not "the model is fine" — "I stopped looking".  Flag it so the
            // `Sat` this `false` licenses is downgraded to `Unknown`.
            self.array_axioms_incomplete = true;
            return false;
        }

        // ---- Phase 1: collect array structure ---------------------------
        // Walk both the user assertions and every axiom instance asserted so
        // far, so selects introduced by earlier read-over-write / extensionality
        // lemmas seed further instantiation (saturation).
        let assertions: Vec<TermId> = self.assertions.clone();
        let instances: Vec<TermId> = self.array_axiom_instances.iter().copied().collect();

        let mut collected = ArrayStructure::default();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        // The user assertions first, with foreign positions recorded.  The ext
        // rule for shared array terms is about the terms the *input* shares
        // between theories; recording the ones an axiom instance introduced
        // instead feeds the rule its own output — every witness read of an
        // array of arrays is an array term in a foreign position, so the pair
        // set grew with the lemma set and a two-row matrix never converged.
        for &root in &assertions {
            collect_array_structure(root, manager, &mut visited, &mut collected, true);
        }
        // Then the instances asserted so far, so selects introduced by earlier
        // read-over-write / extensionality lemmas seed further instantiation
        // (saturation) — but without widening the foreign set.
        for &root in &instances {
            collect_array_structure(root, manager, &mut visited, &mut collected, false);
        }
        // Then every *ground instance* an instantiation path has asserted:
        // MBQI, blind, finite-domain and e-matching results
        // (`Solver::prepare_ground_instance` registers them).  These are the
        // roots that make quantified array reasoning work at all.  An instance
        // is ground by construction but never enters `self.assertions`, and
        // `ground_children` stops at binders, so before this loop existed an
        // array term that was first ground *after* substitution was a root of
        // nothing: no read-over-write lemma, no constant-array congruence, no
        // `ite` naming, and the read was a free value of the element sort.
        //
        // `record_foreign` is `false` for the same reason it is for the axiom
        // instances above: the ext rule for shared array terms is about what
        // the *input* shares between theories, and feeding it the solver's own
        // derived terms is what made the pair set grow with the lemma set.
        let instance_roots: Vec<TermId> = self.ground_array_roots.iter().copied().collect();
        for &root in &instance_roots {
            collect_array_structure(root, manager, &mut visited, &mut collected, false);
        }

        // Nothing for the three syntactic families *and* nothing for the ext
        // rule for shared array terms (phase 4) — only then is there nothing
        // to do.  Leaving `foreign` out of this test is what made
        // `(distinct (fa brr) (fa arr))` — five lines, no `select`, no
        // `store`, no array equality — return here before phase 4 ever ran:
        // the two arrays are forced into different EUF classes by the
        // disequality between the two applications, no witness read was ever
        // minted to tell them apart, and the model printed both as the sort
        // default with `fa` a constant function, falsifying the assertion it
        // claimed to satisfy (`#P2b-34`/`#P2b-37`).
        let foreign_candidates = collected
            .foreign
            .iter()
            .filter(|term| collected.shared_foreign.contains(term))
            .count();
        if collected.selects.is_empty()
            && collected.eq_pairs.is_empty()
            && collected.stores.is_empty()
            && (foreign_candidates == 0 || collected.foreign.len() < 2)
        {
            return false;
        }

        // A store's *own-index* read is a read the script need never spell
        // (`#P2b-37`): `(= (store c i v) c2)` with no `select` anywhere had no
        // collected read at all, so no family fired and the equality was a
        // free Boolean.  Registering it here — before the builders run —
        // gives read-over-write its RoW-1 instance (`select(store(b,i,v),i) =
        // v`) and puts `i` into the store's read-index set, so select
        // congruence carries that index across every equality the store takes
        // part in.
        register_store_own_index_reads(manager, &mut collected);

        // ---- Phase 2: build candidate ground axiom instances ------------
        let mut candidates: Vec<TermId> = Vec::new();
        build_read_over_write(manager, &collected, &mut candidates);
        build_array_ite_reads(manager, &collected, &mut candidates);
        build_const_array_reads(manager, &collected, &mut candidates);
        build_cardinality_refutations(manager, &collected, &mut candidates);
        let mut witnesses: FxHashSet<TermId> = FxHashSet::default();
        let polarity = self.pair_polarity(&collected, manager);
        build_extensionality_and_congruence(
            manager,
            &collected,
            &polarity,
            &mut candidates,
            &mut witnesses,
        );
        if witnesses.len() >= MAX_ARRAY_EXT_WITNESSES {
            self.array_axioms_incomplete = true;
        }

        // Two reads of one array at indices the *candidate model* happens to
        // give the same value must agree (`#P2b-34`).  Nothing asserted says
        // the indices are equal — congruence closure only knows the
        // equalities the formula states — so EUF never compares the two
        // reads, and a candidate model is free to publish
        // `select(a, k1) = 30` and `select(a, k2) = 11` with `k1` and `k2`
        // both assigned `3`.  That is not a model of the array theory, and it
        // is visible in the printed model as one array with two answers at
        // index `3`.  Guided by the candidate model, so only the pairs that
        // actually collide in it cost anything.
        self.build_index_congruence(manager, &collected, &mut candidates);

        // ---- Phase 3: filter (dedup + model) and assert -----------------
        if self.assert_new_instances(&candidates, manager) {
            return true;
        }

        // ---- Phase 3a: array-constant witness congruence ----------------
        // One (constant, pair) cell per round, for the reason
        // [`build_const_array_witness_cell`] documents: read eagerly the rule
        // is cubic in syntactic structure and a six-declaration script stopped
        // answering at all.
        if self.assert_const_array_witness_congruence(&collected, manager) {
            return true;
        }

        // ---- Phase 3b: chain-index select congruence --------------------
        // Deferred out of phase 2 for the reason
        // [`Solver::assert_chain_index_congruence`] documents (`#P2b-38`
        // strand (b)): the chain walk is the part of the congruence family
        // that scales with store-chain depth, and building it for all
        // `C(n,2)` pairs of an `n`-ary `distinct` up front put `O(n²)` fresh
        // bit-vector index terms in front of the BV↔EUF partition exchange
        // before anything had looked at whether the assignment already keeps
        // those pairs apart.  One pair per round, and only for pairs the
        // assignment has not already separated.
        if self.assert_chain_index_congruence(&collected, manager) {
            return true;
        }

        // ---- Phase 3c: off-chain Skolem indices -------------------------
        // The rule of last resort, deferred here for the reason
        // [`build_off_chain_family`] documents: its index is a fresh
        // bit-vector argument term, and minting one per pair eagerly made the
        // BV↔EUF partition exchange the dominant cost of the whole search.
        // One pair per round, in the order the pairs were collected.
        if self.assert_off_chain_family(&collected, manager) {
            return true;
        }

        // ---- Phase 4: foreign pairs -------------------------------------
        // Nothing new came out of the three syntactic families, so the
        // candidate model satisfies every instance they generate.  That is not
        // yet a model of the array theory: two array terms may be *shared*
        // with the uninterpreted fragment — arguments of an `Apply`, values
        // stored into another array, `ite` branches, operands of an array
        // (dis)equality — and sit in different EUF classes while no axiom ever
        // compared them.  `(distinct (f arr) (f brr))` with every index read
        // pinned equal is the minimal case: it is unsat, because arr and brr
        // are extensionally equal and congruence then equates `(f arr)` with
        // `(f brr)`, yet no equality atom over the two arrays appears in the
        // formula for the extensionality family to fire on.
        //
        // The ext rule for shared array terms (de Moura & Bjørner, FMCAD
        // 2009) closes exactly that gap: instantiate the witness lemma for
        // every unordered pair of foreign array terms whose classes differ in
        // the candidate model.  Deferring it to the round where nothing else
        // fires keeps the quadratic family off the common path.
        let foreign_pairs = self.foreign_pairs_in_different_classes(&collected, manager);
        if foreign_pairs.is_empty() {
            return false;
        }
        // One pair per round, not all of them.  Every witness is a fresh index
        // that becomes a read index of both arrays, and congruence then
        // instantiates every *other* pair at it as well — so `n` foreign pairs
        // asserted together cost `O(n^2)` lemmas in the very next round.  The
        // loop is lazy by design: one new lemma is enough to make the caller
        // re-solve, and if the next candidate model still needs another pair
        // this phase runs again.  `(distinct (f arr) (f brr))` has exactly one
        // pair and is unaffected; a formula with nine of them adds nine
        // lemmas over nine rounds instead of eighty-one in one.
        let mut extra: Vec<TermId> = Vec::new();
        for (a, b) in foreign_pairs {
            if witnesses.len() >= MAX_ARRAY_EXT_WITNESSES {
                self.array_axioms_incomplete = true;
                break;
            }
            let before = extra.len();
            push_witness_lemma(manager, a, b, &mut extra);
            if let Some(domain) = array_domain(a, manager) {
                let witness = extensionality_witness(manager, a, b, domain);
                witnesses.insert(witness);
            }
            // Only a lemma the dedup set does not already hold counts as
            // progress; a pair whose witness lemma is already asserted costs
            // nothing and the next pair is tried instead.
            if extra.len() > before
                && extra
                    .last()
                    .is_some_and(|inst| !self.array_axiom_instances.contains(inst))
            {
                break;
            }
        }
        self.assert_new_instances(&extra, manager)
    }

    /// Add `i = j ⇒ select(a,i) = select(a,j)` for every array `a` and every
    /// pair of indices read on it that the candidate model maps to the same
    /// value.
    ///
    /// The lemma is a theorem of the array theory in every model, so the
    /// model-guided restriction costs no soundness: it only decides *when* to
    /// spend an instance.  Restricting it matters because the unrestricted
    /// family is quadratic in the number of indices read on one array, which
    /// on a fifty-write store chain is thousands of lemmas that the search
    /// never needs.
    fn build_index_congruence(
        &self,
        manager: &mut TermManager,
        collected: &ArrayStructure,
        candidates: &mut Vec<TermId>,
    ) {
        let Some(model) = self.model.as_ref() else {
            return;
        };
        // Only reads that already exist as terms are compared, and only when
        // the candidate model *violates* the lemma — equal indices, different
        // values.  Building the family for every colliding pair instead cost
        // ten times the whole check on the read-after-write benchmarks, all of
        // it on lemmas the search never needed: the instance is a theorem
        // either way, so restricting it to the violated pairs decides only
        // when to spend an instance, never whether the answer is right.
        let mut reads: FxHashMap<(TermId, TermId), TermId> = FxHashMap::default();
        for &(select_term, array, index) in &collected.selects {
            reads.entry((array, index)).or_insert(select_term);
        }

        // Deterministic order: `read_indices` is a hash map, and the instances
        // it produces are asserted to the SAT core.
        let mut arrays: Vec<TermId> = collected.read_indices.keys().copied().collect();
        arrays.sort_unstable_by_key(|array: &TermId| array.raw());
        for array in arrays {
            let Some(indices) = collected.read_indices.get(&array) else {
                continue;
            };
            let folded: Vec<(TermId, Option<EvalVal>, Option<EvalVal>)> = indices
                .iter()
                .map(|&index| {
                    let index_value = self.eval_in_model(index, model, manager, 0);
                    let read_value = reads
                        .get(&(array, index))
                        .and_then(|&read| self.eval_in_model(read, model, manager, 0));
                    (index, index_value, read_value)
                })
                .collect();
            for (position, (first, first_index, first_read)) in folded.iter().enumerate() {
                for (second, second_index, second_read) in folded.iter().skip(position + 1) {
                    let (Some(first_index), Some(second_index)) =
                        (first_index.as_ref(), second_index.as_ref())
                    else {
                        continue;
                    };
                    if first_index != second_index {
                        continue;
                    }
                    let (Some(first_read), Some(second_read)) =
                        (first_read.as_ref(), second_read.as_ref())
                    else {
                        continue;
                    };
                    if first_read == second_read {
                        continue;
                    }
                    let index_eq = manager.mk_eq(*first, *second);
                    let read_first = manager.mk_select(array, *first);
                    let read_second = manager.mk_select(array, *second);
                    let reads_eq = manager.mk_eq(read_first, read_second);
                    candidates.push(manager.mk_implies(index_eq, reads_eq));
                }
            }
        }
    }

    /// Assert every candidate instance the dedup set does not already hold and
    /// the candidate model does not already *definitely* satisfy; returns
    /// whether anything was added (so the caller must re-solve).
    ///
    /// A `None` evaluation (opaque/undetermined) is treated as unsatisfied so
    /// completeness never depends on the model being able to evaluate a
    /// `select` — worst case this degenerates to eager instantiation, which is
    /// still sound and complete.
    fn assert_new_instances(&mut self, candidates: &[TermId], manager: &mut TermManager) -> bool {
        let mut to_add: Vec<TermId> = Vec::new();
        {
            let model = self.model.as_ref();
            for &inst in candidates {
                if self.array_axiom_instances.contains(&inst) {
                    continue;
                }
                let already_satisfied = match model {
                    Some(m) => matches!(
                        self.eval_in_model(inst, m, manager, 0),
                        Some(EvalVal::Bool(true))
                    ),
                    None => false,
                };
                if already_satisfied {
                    continue;
                }
                to_add.push(inst);
            }
        }

        let mut added = false;
        for inst in to_add {
            if self.array_axiom_instances.len() >= MAX_ARRAY_AXIOM_INSTANCES {
                // Instances the candidate model does *not* satisfy are being
                // left unasserted, so the axiomatisation this search runs
                // against is a strict subset of the array theory.
                self.array_axioms_incomplete = true;
                break;
            }
            // `insert` returns false if this exact instance is already tracked
            // (it may appear twice within one candidate batch).
            if !self.array_axiom_instances.insert(inst) {
                continue;
            }
            // Journal the instance so a `pop` retracts the dedup entry together
            // with the lemma clause the SAT core drops: keeping the entry would
            // silently suppress an axiom a later scope still needs.
            self.trail
                .push(super::trail::TrailOp::ArrayAxiomInstanceAdded { term: inst });
            let lit = self.encode(inst, manager);
            let _ = self.sat.add_clause([lit]);
            // The deterministic currency the refinement budget is denominated
            // in (decision (9)).  Counted here, at the one place a lemma
            // actually reaches the SAT core, so no builder can enlarge the
            // circuit without the budget seeing it.
            self.statistics.array_lemma_instances =
                self.statistics.array_lemma_instances.saturating_add(1);
            added = true;
        }

        added
    }

    /// What this candidate assignment and model have already committed to for
    /// each unordered array pair (`#P2b-38` strand (b)).
    ///
    /// Three sets, read off the SAT trail and the published model rather than
    /// off the syntax, because the syntax does not say which polarity a
    /// `distinct` or an `=` was given under negation:
    ///
    /// * `held_apart` — the trail assigns `(= a b)` **false**, or assigns an
    ///   `n`-ary `distinct` over array operands **true** with `a` and `b`
    ///   among them.  Every select-congruence lemma
    ///   `a = b ⇒ select(a,j) = select(b,j)` for the pair has a false
    ///   antecedent, so the family is satisfied by this assignment.
    /// * `held_equal` — the trail assigns `(= a b)` **true**.  The witness
    ///   lemma `a = b ∨ select(a,k) != select(b,k)` is satisfied by its first
    ///   disjunct, so minting `k` for it buys nothing.
    /// * `separated_by_reads` — the candidate *model* publishes reads of `a`
    ///   and `b` at one index with different values.  This candidate is not
    ///   claiming the two arrays are the same function, so it needs no witness
    ///   index to be told they differ; the same model-guided restriction the
    ///   ext rule for shared array terms uses in phase 4.
    ///
    /// # Why this is read before a family is *built*, not after
    ///
    /// The filter in [`Solver::assert_new_instances`] runs on an already
    /// interned lemma, and interning is the expensive half.  Each
    /// extensionality witness is a fresh bit-vector index term, and a read at
    /// it cascades a read-over-write instance down every link of both store
    /// chains; `collect_pair_indices` adds about ten more `select` terms per
    /// pair.  All of them join the candidate set of the BV↔EUF partition-lemma
    /// exchange (`theory_manager::bv_bridge`), whose cost grows with the
    /// number of *partitions* of that set — and an `n`-ary `distinct` pushes
    /// all `C(n,2)` pairs into [`ArrayStructure::eq_pairs`].  Four operands of
    /// depth four were enough to turn a script this tree answered in 0.06 ms
    /// into one that did not answer in two minutes, with 92 % of the samples
    /// inside `BvSolver::check`.  The assignment already knew those pairs were
    /// apart; asking it first is what keeps the circuit from being built.
    ///
    /// Conservative in the safe direction: a pair the trail has not decided
    /// appears in no set, so its families are built as before.  Skipping one
    /// costs no completeness either — every lemma here is a theorem of the
    /// array theory, so a skipped instance is deferred, not lost: if a later
    /// candidate changes its mind about the pair the family is built then.
    fn pair_polarity(&self, collected: &ArrayStructure, manager: &TermManager) -> PairPolarity {
        use oxiz_sat::LBool;

        let mut polarity = PairPolarity::default();
        for &(atom, lhs, rhs) in &collected.eq_atoms {
            match self
                .term_to_var
                .get(&atom)
                .map(|&var| self.sat.model_value(var))
            {
                Some(LBool::False) => {
                    polarity.held_apart.insert(unordered(lhs, rhs));
                }
                Some(LBool::True) => {
                    polarity.held_equal.insert(unordered(lhs, rhs));
                }
                _ => {}
            }
        }
        for &atom in &collected.distinct_atoms {
            if !self
                .term_to_var
                .get(&atom)
                .is_some_and(|&var| self.sat.model_value(var) == LBool::True)
            {
                continue;
            }
            let Some(TermKind::Distinct(args)) = manager.get(atom).map(|t| &t.kind) else {
                continue;
            };
            for (position, &lhs) in args.iter().enumerate() {
                if !is_array_sorted(lhs, manager) {
                    continue;
                }
                for &rhs in args.iter().skip(position + 1) {
                    if lhs != rhs && is_array_sorted(rhs, manager) {
                        polarity.held_apart.insert(unordered(lhs, rhs));
                    }
                }
            }
        }

        // Published reads of the candidate model, keyed by (array, index) —
        // the same table the ext rule for shared array terms builds in
        // `foreign_pairs_in_different_classes`.
        let mut reads: FxHashMap<(TermId, TermId), TermId> = FxHashMap::default();
        if let Some(model) = self.model.as_ref() {
            for (&term, &value) in model.assignments() {
                if let Some(TermKind::Select(array, index)) = manager.get(term).map(|t| &t.kind) {
                    reads.insert((*array, *index), value);
                }
            }
        }
        if !reads.is_empty() {
            for &(a, b) in &collected.eq_pairs {
                let differs = reads.iter().any(|(&(array, index), &value)| {
                    array == a && reads.get(&(b, index)).is_some_and(|&other| other != value)
                });
                if differs {
                    polarity.separated_by_reads.insert(unordered(a, b));
                }
            }
        }
        polarity
    }

    /// Phase 3a: the array-constant witness-congruence cell of the first
    /// (constant, pair) combination that still needs one, or `false` when
    /// every such cell the assignment has not already decided is saturated.
    ///
    /// The rule itself is [`build_const_array_witness_cell`], and that
    /// function's documentation carries both the soundness hole it closes
    /// (`#P2b-37`, two array constants reached only through a variable of
    /// indirection) and why it may not be built eagerly.
    ///
    /// Cells are enumerated constants-outer, pairs-inner, both in collection
    /// order — the term-graph walk's pre-order — so the choice is
    /// deterministic and does not depend on hashing or on machine speed.
    ///
    /// A pair the assignment holds *apart* is skipped: the congruence's
    /// antecedent `a = b` is false under this assignment, so the cell has
    /// nothing to say about it.  The constant's own witness read is skipped
    /// with it, because that read exists only to give the congruence something
    /// to compare.
    fn assert_const_array_witness_congruence(
        &mut self,
        collected: &ArrayStructure,
        manager: &mut TermManager,
    ) -> bool {
        if collected.const_arrays.is_empty() || collected.eq_pairs.is_empty() {
            return false;
        }
        let polarity = self.pair_polarity(collected, manager);
        let constants = collected.const_arrays.clone();
        let pairs = collected.eq_pairs.clone();
        for constant in constants {
            let Some(const_sort) = manager.get(constant).map(|t| t.sort) else {
                continue;
            };
            let Some(const_domain) = array_domain(constant, manager) else {
                continue;
            };
            for &(a, b) in &pairs {
                if polarity.held_apart.contains(&unordered(a, b)) {
                    continue;
                }
                if pair_is_enumerated(a, manager) {
                    // The enumerated family already compares this pair at
                    // every element of its index sort, and
                    // `build_const_array_reads` already decides the constant
                    // there.  A Skolem witness for it would be a fresh index
                    // the enumeration made unnecessary.
                    continue;
                }
                // The cell only makes sense where the constant is comparable
                // with the pair: same domain sort, so the witness indexes it,
                // and — when the constant is a member of the pair — the same
                // array sort as the other side.
                if array_domain(a, manager) != Some(const_domain) {
                    continue;
                }
                if constant == a || constant == b {
                    let other = if constant == a { b } else { a };
                    if manager.get(other).map(|t| t.sort) != Some(const_sort) {
                        continue;
                    }
                }
                let witness = extensionality_witness(manager, a, b, const_domain);
                let mut family: Vec<TermId> = Vec::new();
                build_const_array_witness_cell(manager, constant, witness, (a, b), &mut family);
                if self.assert_new_instances(&family, manager) {
                    return true;
                }
            }
        }
        false
    }

    /// Phase 3b: the *chain-index* select-congruence family of the first
    /// equality pair that still needs one, or `false` when every pair the
    /// assignment has not already separated is saturated.
    ///
    /// One pair per round, like the off-chain family in phase 3c and the
    /// foreign-pair rule in phase 4, and for the same reason: every index this
    /// family reaches becomes a read index of both arrays, so `n` pairs
    /// asserted in one round put `O(n)` fresh bit-vector argument terms in
    /// front of the partition exchange at once.  The witness-index instance
    /// stays eager in [`build_extensionality_and_congruence`] — it is one
    /// lemma per pair and it is what decides the `(as const)` and `distinct`
    /// shapes — while the chain walk, which is the part that scales with chain
    /// depth, waits until the cheaper families have nothing left to say.
    ///
    /// The pairs are tried in collection order (the term-graph walk's
    /// pre-order), so the choice is deterministic and does not depend on
    /// hashing or on machine speed.
    fn assert_chain_index_congruence(
        &mut self,
        collected: &ArrayStructure,
        manager: &mut TermManager,
    ) -> bool {
        let polarity = self.pair_polarity(collected, manager);
        let pairs = collected.eq_pairs.clone();
        for (a, b) in pairs {
            if polarity.held_apart.contains(&unordered(a, b)) {
                continue;
            }
            if pair_is_enumerated(a, manager) {
                // Subsumed: the enumerated family already carries congruence
                // across this pair at *every* element of its index sort, so
                // the chain walk can only re-derive instances at indices that
                // set already contains.  Skipping it is what keeps the walk's
                // fresh reads — the cost centre of `#P2b-38` strand (b) — off
                // the small-index-sort shapes entirely.
                continue;
            }
            let mut indices: Vec<TermId> = Vec::new();
            collect_pair_indices(manager, collected, a, &mut indices);
            collect_pair_indices(manager, collected, b, &mut indices);
            if indices.is_empty() {
                continue;
            }
            let mut family: Vec<TermId> = Vec::new();
            push_congruence_at(manager, a, b, &indices, &mut family);
            if self.assert_new_instances(&family, manager) {
                return true;
            }
        }
        false
    }

    /// Phase 3c: the off-chain Skolem index family of the *first* equality
    /// pair that still needs one, or `false` when every pair's family is
    /// already asserted or the cardinality guard refuses it.
    ///
    /// One pair per round, like the foreign-pair rule and for the same reason
    /// — see [`build_off_chain_family`] for why this family is not built with
    /// the other three.  The pairs are tried in collection order, which is the
    /// term-graph walk's pre-order, so the choice is deterministic.
    fn assert_off_chain_family(
        &mut self,
        collected: &ArrayStructure,
        manager: &mut TermManager,
    ) -> bool {
        let pairs = collected.eq_pairs.clone();
        for (a, b) in pairs {
            if pair_is_enumerated(a, manager) {
                // Subsumed, and for the strongest of the three reasons: an
                // off-chain index is by definition an element of the index
                // sort, and the enumerated family covers every element of it.
                // A Skolem index here would be a fresh variable ranging over a
                // domain already written out in full.
                continue;
            }
            let mut family: Vec<TermId> = Vec::new();
            build_off_chain_family(manager, collected, a, b, &mut family);
            if family.is_empty() {
                continue;
            }
            if self.assert_new_instances(&family, manager) {
                return true;
            }
        }
        false
    }

    /// Unordered pairs of foreign array terms the candidate model places in
    /// *different* EUF congruence classes — the pairs the ext rule has to
    /// compare (see `instantiate_array_axioms`, phase 4).
    ///
    /// A term the congruence closure never interned has no class at all; such
    /// a pair counts as differing, which is the direction that can only add
    /// lemmas (every witness lemma is a theorem of the array theory, so an
    /// unnecessary one costs a round, never an answer).
    fn foreign_pairs_in_different_classes(
        &self,
        collected: &ArrayStructure,
        manager: &TermManager,
    ) -> Vec<(TermId, TermId)> {
        // Published reads of the candidate model, keyed by (array, index):
        // a pair the model already separates at some index needs no witness,
        // because the model is not claiming the two arrays are the same
        // function.
        let mut reads: FxHashMap<(TermId, TermId), TermId> = FxHashMap::default();
        if let Some(model) = self.model.as_ref() {
            for (&term, &value) in model.assignments() {
                if let Some(TermKind::Select(array, index)) = manager.get(term).map(|t| &t.kind) {
                    reads.insert((*array, *index), value);
                }
            }
        }
        let separated = |a: TermId, b: TermId| -> bool {
            reads.iter().any(|(&(array, index), &value)| {
                array == a && reads.get(&(b, index)).is_some_and(|&other| other != value)
            })
        };

        let mut pairs: Vec<(TermId, TermId)> = Vec::new();
        for (position, &a) in collected.foreign.iter().enumerate() {
            for &b in collected.foreign.iter().skip(position + 1) {
                if a == b {
                    continue;
                }
                if !collected.shared_foreign.contains(&a) && !collected.shared_foreign.contains(&b)
                {
                    continue;
                }
                let class_a = self.euf_class_representative(a);
                let class_b = self.euf_class_representative(b);
                if let (Some(rep_a), Some(rep_b)) = (class_a, class_b)
                    && rep_a == rep_b
                {
                    continue;
                }
                // Model-guided: only a pair the candidate model reads as the
                // *same* function needs the ext rule.  Two arrays it already
                // separates are not the source of a congruence violation, and
                // instantiating them costs a witness index the search then has
                // to carry.
                if separated(a, b) {
                    continue;
                }
                pairs.push((a, b));
            }
        }
        pairs
    }

    /// Record that `term` mentions an array operation, so [`super::check_core`]
    /// runs the lazy refinement loop above for it.
    ///
    /// # Why the guard needs its own walk (`#P2b-33`)
    ///
    /// [`Solver::has_array_ops`](super::Solver::has_array_ops) is the *guard*
    /// on `instantiate_array_axioms`; this function and
    /// [`collect_array_structure`] are therefore two halves of one decision and
    /// must agree on what "mentions an array" means.  They did not.  The flag
    /// was raised only from
    /// [`Solver::track_theory_vars`](super::Solver::track_theory_vars) and the
    /// encoder's `Select`/`Store` arm, and `track_theory_vars` deliberately does
    /// **not** descend into an uninterpreted application's arguments (nor into
    /// `Distinct`, `Implies` or `Xor` operands) — see its own doc comment.  A
    /// read that occurs *only* as a function argument,
    /// `(distinct (f (select (store arr i v) i)) (f v))`, therefore left the
    /// flag `false`: the refinement loop never ran, no read-over-write instance
    /// was ever built, the read stayed an unconstrained leaf and the formula —
    /// unsatisfiable in QF_AUF, QF_AUFBV and QF_AUFLIA alike — answered `sat`.
    /// The instantiator's own walk (`ground_children`) would have collected that
    /// read; it was never given the chance.
    ///
    /// So the guard is computed here with exactly the walk the instantiator
    /// uses, binder exclusion included: an over-approximation is free (one
    /// wasted round of `instantiate_array_axioms`, which then reports "nothing
    /// to add"), an under-approximation is a wrong answer.
    ///
    /// Iterative, and stops at the first array term: for the array-free
    /// formulas that make up most of the corpus this is one linear scan per
    /// encoded term, and once the flag is set it costs nothing at all.
    pub(super) fn mark_array_ops(&mut self, term: TermId, manager: &TermManager) {
        if self.has_array_ops {
            return;
        }
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = vec![term];
        let mut children: Vec<TermId> = Vec::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = manager.get(current) else {
                continue;
            };
            // Any array *term*, not just a `select`/`store` application: an
            // array (dis)equality between two array constants —
            // `(= ((as const A) #b1) ((as const A) #b0))` — mentions neither
            // operator, so the guard stayed false, the refinement loop never
            // ran, and the atom was a free Boolean the SAT core satisfied by
            // fiat (`#P2b-37`).  Over-approximating costs one round of
            // `instantiate_array_axioms`, which then reports "nothing to
            // collect" and returns; under-approximating is a wrong answer.
            if matches!(data.kind, TermKind::Select(..) | TermKind::Store(..))
                || is_array_sorted(current, manager)
            {
                // `has_array_ops` is restored wholesale from the `push`
                // snapshot (see `trail.rs`), so — like the two pre-existing
                // write sites — this needs no journal entry of its own.
                self.has_array_ops = true;
                return;
            }
            children.clear();
            ground_children(&data.kind, &mut children);
            stack.extend(children.iter().copied());
        }
    }
}

/// Array terms and (dis)equalities gathered from a term-graph walk.
#[derive(Default)]
struct ArrayStructure {
    /// `(select_term, array_operand, index)` for every `select` encountered.
    selects: Vec<(TermId, TermId, TermId)>,
    /// Unordered array-sorted (dis)equality atom operands `(a, b)`
    /// (`a != b` syntactically), from `=` and from every pair of an `n`-ary
    /// `distinct` over array-sorted operands.  Deduplicated as an *unordered*
    /// pair: the extensionality witness and the congruence family of `{a, b}`
    /// do not depend on which side is which, and the same pair is reached
    /// twice as a matter of course — the witness lemma this module asserts for
    /// it contains `(= a b)`, which the next round's walk collects again from
    /// the instance list.  Building the family twice per round costs exactly
    /// the work it saves here.
    eq_pairs: Vec<(TermId, TermId)>,
    /// The unordered keys of [`ArrayStructure::eq_pairs`], for that dedup.
    eq_pair_keys: FxHashSet<(TermId, TermId)>,
    /// `(atom, lhs, rhs)` for every array-sorted `=` atom encountered.
    ///
    /// Kept beside [`ArrayStructure::eq_pairs`] because the *atom* is what
    /// carries the candidate assignment's polarity: a pair whose `(= a b)` the
    /// SAT core committed to **false** satisfies every select-congruence lemma
    /// `a = b ⇒ select(a,j) = select(b,j)` vacuously, and building that family
    /// for it is pure cost.  See
    /// [`Solver::pairs_the_assignment_separates`](super::Solver::pairs_the_assignment_separates).
    eq_atoms: Vec<(TermId, TermId, TermId)>,
    /// Every `n`-ary `distinct` atom with array-sorted operands, for the
    /// cardinality refutation ([`build_cardinality_refutations`]).
    distinct_atoms: Vec<TermId>,
    /// `array_variable -> store_term` for every asserted `var = store(...)`.
    aliases: FxHashMap<TermId, TermId>,
    /// Distinct indices read on each array operand (for select congruence).
    read_indices: FxHashMap<TermId, Vec<TermId>>,
    /// `(store_term, index, value)` for every `store` encountered, so the
    /// store's own-index read can be registered even when the script never
    /// spells it (`#P2b-37`).
    stores: Vec<(TermId, TermId, TermId)>,
    /// Every `((as const A) d)` term encountered, in first-encounter order and
    /// deduplicated, for [`build_const_array_defaults`].
    const_arrays: Vec<TermId>,
    /// The dedup set behind [`ArrayStructure::const_arrays`].
    const_array_keys: FxHashSet<TermId>,
    /// Array-sorted terms occurring in a *foreign* position — one where the
    /// array is handed to something other than the array operators themselves:
    /// an argument of an uninterpreted `Apply`, the value written into another
    /// array, a branch of an array-sorted `ite`, or an operand of an array
    /// (dis)equality.  These are the terms the ext rule for shared array terms
    /// compares pairwise (`instantiate_array_axioms`, phase 4).  Deduplicated,
    /// in first-encounter order.
    foreign: Vec<TermId>,
    /// The subset of [`ArrayStructure::foreign`] reached through a *genuinely
    /// shared* position — an `Apply` argument, a stored value or an `ite`
    /// branch — as opposed to an operand of an array (dis)equality.
    ///
    /// Every pair of (dis)equality operands already has its witness from the
    /// extensionality family, so a pair of two such terms adds a lemma that
    /// family has covered; only a pair with a *shared* term on at least one
    /// side is new information.  Requiring that is what keeps the quadratic
    /// rule off formulas built out of array equalities alone, where it
    /// otherwise multiplied the lemma set several-fold and the search with it.
    shared_foreign: FxHashSet<TermId>,
    /// `(ite_term, condition, then_branch, else_branch)` for every
    /// **array-sorted** `ite` encountered, deduplicated in first-encounter
    /// order (`#P2b-41`).
    ///
    /// The encoder names an array-sorted `ite` with a fresh variable and two
    /// defining implications, which is what makes a read *directly* through
    /// one decidable.  It is not enough on its own: the array theory walks the
    /// solver's own assertion list, which still holds the *un-eliminated*
    /// term, so a read through a `store` whose base is an `ite` —
    /// `select(store(ite(c,x,y), j, v), i)` — reduced by read-over-write to
    /// `select(ite(c,x,y), i)`, a read of a term no rule related to `x` or `y`.
    /// Seven lines plus a `store` were still a wrong `sat` after the encoder
    /// half landed.  [`build_array_ite_reads`] closes it from this side.
    array_ites: Vec<(TermId, TermId, TermId, TermId)>,
    /// The dedup set behind [`ArrayStructure::array_ites`].
    array_ite_keys: FxHashSet<TermId>,
}

/// The unordered key of an array pair, so the two sides' order never matters.
fn unordered(a: TermId, b: TermId) -> (TermId, TermId) {
    if a.raw() <= b.raw() { (a, b) } else { (b, a) }
}

/// What the candidate assignment and model already say about each array pair.
///
/// Built by [`Solver::pair_polarity`](super::Solver::pair_polarity), which
/// documents each set and why the extensionality families consult it before
/// interning anything.
#[derive(Default)]
struct PairPolarity {
    /// Pairs the trail holds *apart*: every select-congruence lemma for them
    /// has a false antecedent.
    held_apart: FxHashSet<(TermId, TermId)>,
    /// Pairs the trail holds *equal*: the witness lemma is satisfied by its
    /// first disjunct.
    held_equal: FxHashSet<(TermId, TermId)>,
    /// Pairs the candidate model already separates at a published read.
    separated_by_reads: FxHashSet<(TermId, TermId)>,
}

impl ArrayStructure {
    /// Record the unordered array pair `{a, b}` once.
    fn push_eq_pair(&mut self, a: TermId, b: TermId) {
        if self.eq_pair_keys.insert(unordered(a, b)) {
            self.eq_pairs.push((a, b));
        }
    }

    /// Record `term` as occupying a foreign position, if it is array-sorted
    /// and not already recorded.  `shared` distinguishes a genuinely shared
    /// position from an array (dis)equality operand — see
    /// [`ArrayStructure::shared_foreign`].
    fn note_foreign(&mut self, term: TermId, manager: &TermManager, record: bool) {
        if !record || !is_array_sorted(term, manager) {
            return;
        }
        if !self.foreign.contains(&term) {
            self.foreign.push(term);
        }
        self.shared_foreign.insert(term);
    }

    /// Record `term` as an operand of an array (dis)equality: a foreign
    /// position, but not a *shared* one — see
    /// [`ArrayStructure::shared_foreign`].
    fn note_foreign_operand(&mut self, term: TermId, manager: &TermManager, record: bool) {
        if !record || !is_array_sorted(term, manager) {
            return;
        }
        if !self.foreign.contains(&term) {
            self.foreign.push(term);
        }
    }
}

/// Gather array structure from `term`.  `visited` prevents re-descending
/// shared sub-terms of the interned DAG.
///
/// Iterative (explicit work stack), so nesting depth is bounded by memory
/// rather than by the native call stack — this walk has no error channel, so a
/// depth cap could only silently drop array structure and with it the
/// read-over-write / extensionality lemmas that make the answer sound.
/// Children are pushed in reverse, which reproduces the recursive pre-order
/// exactly and with it the order of `selects`, `eq_pairs` and `read_indices`.
fn collect_array_structure(
    term: TermId,
    manager: &TermManager,
    visited: &mut FxHashSet<TermId>,
    out: &mut ArrayStructure,
    record_foreign: bool,
) {
    let mut stack: Vec<TermId> = vec![term];
    while let Some(term) = stack.pop() {
        if !visited.insert(term) {
            continue;
        }
        let Some(data) = manager.get(term) else {
            continue;
        };
        match &data.kind {
            TermKind::Select(array, index) => {
                out.selects.push((term, *array, *index));
                let entry = out.read_indices.entry(*array).or_default();
                if !entry.contains(index) {
                    entry.push(*index);
                }
                stack.push(*index);
                stack.push(*array);
            }
            TermKind::Store(base, index, value) => {
                out.stores.push((term, *index, *value));
                // An array written *into* another array is shared with the
                // array-of-arrays fragment rather than consumed here.
                out.note_foreign(*value, manager, record_foreign);
                stack.push(*value);
                stack.push(*index);
                stack.push(*base);
            }
            TermKind::Eq(lhs, rhs) => {
                // Record an array-sorted equality atom (either polarity: the
                // extensionality / congruence lemmas are valid regardless).
                if lhs != rhs && is_array_sorted(*lhs, manager) && is_array_sorted(*rhs, manager) {
                    out.push_eq_pair(*lhs, *rhs);
                    // The atom itself, so the congruence family can ask the
                    // trail whether this candidate assignment already committed
                    // to `a != b` (see `pairs_the_assignment_separates`).
                    out.eq_atoms.push((term, *lhs, *rhs));
                    out.note_foreign_operand(*lhs, manager, record_foreign);
                    out.note_foreign_operand(*rhs, manager, record_foreign);
                }
                // Record a `var = store(...)` alias for alias-aware
                // read-over-write.
                record_alias(*lhs, *rhs, manager, &mut out.aliases);
                record_alias(*rhs, *lhs, manager, &mut out.aliases);
                stack.push(*rhs);
                stack.push(*lhs);
            }
            // `distinct` over array-sorted operands is an array *dis*equality
            // and needs the same witness as `(not (= a b))` — which reaches
            // the `Eq` arm above and always did.  Falling into the generic arm
            // instead contributed nothing at all, so `(distinct arr brr)` got
            // no witness index and the two arrays stayed free leaves the SAT
            // core could satisfy by fiat: a wrong `sat` needing no array
            // constant and no store (`#P2b-37`).  Every unordered pair of an
            // `n`-ary `distinct` is recorded, because `distinct` is pairwise.
            TermKind::Distinct(args) => {
                if args.iter().any(|&arg| is_array_sorted(arg, manager)) {
                    out.distinct_atoms.push(term);
                }
                for (position, &lhs) in args.iter().enumerate() {
                    if !is_array_sorted(lhs, manager) {
                        continue;
                    }
                    out.note_foreign_operand(lhs, manager, record_foreign);
                    for &rhs in args.iter().skip(position + 1) {
                        if lhs != rhs && is_array_sorted(rhs, manager) {
                            out.push_eq_pair(lhs, rhs);
                        }
                    }
                }
                stack.extend(args.iter().rev().copied());
            }
            // An array-sorted argument of an uninterpreted application, or an
            // array-sorted `ite` branch, is shared with the uninterpreted
            // fragment: congruence can equate two applications of it without
            // any array atom ever naming the arrays.
            TermKind::Apply { args, .. } => {
                // The array constant is an ordinary `Apply` under a reserved
                // function symbol (`CONST_ARRAY_FUNC`), so this is where one is
                // recognised.  Recorded for `build_const_array_defaults`,
                // which is the only rule that compares two of them directly.
                if const_array_default(term, manager).is_some() && out.const_array_keys.insert(term)
                {
                    out.const_arrays.push(term);
                }
                for &arg in args {
                    out.note_foreign(arg, manager, record_foreign);
                }
                stack.extend(args.iter().rev().copied());
            }
            TermKind::Ite(cond, then_branch, else_branch) => {
                let (cond, then_branch, else_branch) = (*cond, *then_branch, *else_branch);
                out.note_foreign(then_branch, manager, record_foreign);
                out.note_foreign(else_branch, manager, record_foreign);
                if array_domain(term, manager).is_some() && out.array_ite_keys.insert(term) {
                    out.array_ites.push((term, cond, then_branch, else_branch));
                }
                stack.push(else_branch);
                stack.push(then_branch);
                stack.push(cond);
            }
            _ => {
                let mut children: Vec<TermId> = Vec::new();
                ground_children(&data.kind, &mut children);
                stack.extend(children.into_iter().rev());
            }
        }
    }
}

/// If `var_term` is a plain variable and `store_term` is a `store` expression,
/// record `var_term -> store_term`.
fn record_alias(
    var_term: TermId,
    store_term: TermId,
    manager: &TermManager,
    aliases: &mut FxHashMap<TermId, TermId>,
) {
    let (Some(var_data), Some(store_data)) = (manager.get(var_term), manager.get(store_term))
    else {
        return;
    };
    if matches!(var_data.kind, TermKind::Var(_)) && matches!(store_data.kind, TermKind::Store(..)) {
        aliases.entry(var_term).or_insert(store_term);
    }
}

/// How far down a store chain one collected `select` is reduced in a single
/// pass.
///
/// The RoW-2 consequent of a read introduces `select(base, index)` — a *new*
/// select, which the structural walk only sees on the next refinement round.
/// Reducing one level per round makes an `n`-deep store chain cost `n` rounds,
/// and every round throws the search away and re-solves from root, so the
/// store-commutativity benchmarks (chains of 10, 20, 50 writes) spent all their
/// time replaying searches rather than deciding anything.  Following the chain
/// here instead collapses that to one or two rounds.
///
/// The budget bounds the work per select, and doubles as the cycle guard's
/// backstop: an alias cycle (`a = store(b,..)` together with `b = store(a,..)`)
/// is caught by `seen` below, but a budget that cannot run away is the cheaper
/// thing to reason about.  Chains longer than this still reduce — the remaining
/// levels simply arrive over later rounds, exactly as they did before.
const MAX_STORE_CHAIN_DEPTH: usize = 128;

/// Register `select(store(b,i,v), i)` as a collected read of every collected
/// `store` term, and `i` as one of that store's read indices (`#P2b-37`).
///
/// # Why a store's own index has to be read even when nothing reads it
///
/// The two families that can refute an array equality both need an index to
/// work at: read-over-write reduces a `select`, and select congruence carries
/// a *read* index across the equality.  A script like
/// `(= (store ((as const A) #b1) i #b0) ((as const A) #b1))` contains neither
/// — there is no `select` anywhere — so nothing fired and the equality was a
/// free Boolean the SAT core satisfied by fiat, a wrong `sat`.  The store's
/// own index is the one index at which the two sides are guaranteed to differ
/// if they differ at all for the reason the store introduced, so registering
/// that read gives RoW-1 its instance (`select(store(b,i,v), i) = v`) and
/// gives congruence an index to carry.
///
/// The read is interned, not asserted: it becomes a candidate instance only
/// through the ordinary builders, and the model filter still decides whether
/// the round needs it.
///
/// # Mutation witness
///
/// `#P2b-39` recorded this rule as implemented but not independently
/// observable, on the grounds that [`collect_pair_indices`] materialises the
/// same read one refinement round later.  **That record was wrong**, and the
/// witness is in this repository:
/// `tests/round4_pass2_recheck_pins.rs::the_store_own_index_read_is_what_decides_this_script`
/// answers `sat` with this registration and `unknown` without it — an early
/// `return` here turns that test red in about 0.9 s, with no wall-clock budget
/// involved, and restoring it turns it green again.
///
/// The subsumption argument holds only where the pair's index set already
/// reaches the store, and the witness script is a shape where it does not: the
/// deciding store sits under an `(as const …)` equality whose congruence family
/// never walks that chain.  Decision (7)'s mutation requirement is therefore
/// met for this rule, and the note is kept as the record of how it is met
/// rather than as an excuse for absent coverage.
fn register_store_own_index_reads(manager: &mut TermManager, collected: &mut ArrayStructure) {
    let mut known: FxHashSet<TermId> = collected
        .selects
        .iter()
        .map(|&(select_term, _, _)| select_term)
        .collect();
    let stores = core::mem::take(&mut collected.stores);
    for &(store_term, index, _) in &stores {
        let read = manager.mk_select(store_term, index);
        if known.insert(read) {
            collected.selects.push((read, store_term, index));
        }
        let entry = collected.read_indices.entry(store_term).or_default();
        if !entry.contains(&index) {
            entry.push(index);
        }
    }
    collected.stores = stores;
}

/// The *off-chain Skolem index* family for one array pair — the rule of last
/// resort, and the one the refinement loop reaches for only when nothing else
/// fires (see `instantiate_array_axioms`, phase 3c).
///
/// `(= (store ((as const A) #b0) i #b1) ((as const A) #b1))` is unsat because
/// the two sides differ at every index other than `i`, and over a two-element
/// index sort such an index exists.  No index in the formula names it, though:
/// every congruence instance is at an index the script or a store already
/// mentions, and at `i` the two sides agree.  So the pair gets its own Skolem
/// index `d`, asserted different from every store index on either chain — a
/// constraint on a *fresh* symbol that any model can satisfy as long as the
/// index sort has more elements than the chains have writes, which is exactly
/// the cardinality condition checked here.  Without the condition the
/// constraint would be unsatisfiable on a sort too small to hold an off-chain
/// index, and asserting it would report `unsat` for a satisfiable formula.
///
/// Only when the two chains bottom out at *different* arrays, which is what
/// makes an off-chain index informative: two chains over one base agree
/// everywhere their writes do not reach, whatever that index is, so the lemma
/// is vacuous — and minting it anyway put a fresh index and a `select` at it
/// on every level of every alias chain, which on the read-after-write
/// benchmarks cost twenty times the whole check.
///
/// # Why this runs in a phase of its own (`#P2b-38`)
///
/// Every off-chain index is a fresh *bit-vector argument term* of the two
/// reads taken at it, so it joins the candidate set of the BV↔EUF
/// partition-lemma exchange (`theory_manager::bv_bridge`), whose cost grows
/// with the number of partitions of that set.  Minting one per equality pair
/// eagerly is what turned an `n`-ary `distinct` over store chains into a
/// non-terminating search: four generated scripts that the tree answered in
/// milliseconds before the `distinct` pairs existed ran for minutes with no
/// answer.  Deferring the family to a round in which the three syntactic
/// families add nothing, and emitting it for one pair at a time, keeps it
/// available exactly where it is the only rule that can decide the pair —
/// `(= (store c i v) c2)` has no other family to fire — while keeping it off
/// formulas that never needed it.
fn build_off_chain_family(
    manager: &mut TermManager,
    collected: &ArrayStructure,
    a: TermId,
    b: TermId,
    candidates: &mut Vec<TermId>,
) {
    let mut chain_indices: Vec<TermId> = Vec::new();
    collect_chain_store_indices(manager, a, &mut chain_indices);
    collect_chain_store_indices(manager, b, &mut chain_indices);
    if chain_indices.is_empty()
        || chain_base(manager, collected, a) == chain_base(manager, collected, b)
    {
        return;
    }
    let Some(domain) = array_domain(a, manager) else {
        return;
    };
    let writes = chain_indices.len();
    // Unconditionally when the index sort has more elements than the chains
    // have writes; and *guarded by a coincidence* when it has exactly as many,
    // because two writes at one index leave a free element behind.
    // `(= (store (store ((as const A) #b1) i1 #b1) i0 #b0) ((as const A) #b0))`
    // over a two-element index sort is unsat exactly that way: congruence at
    // the chain's own indices forces `i0 = i1`, and the guarded lemma then
    // supplies the off-chain index that refutes the pair.  The guard is what
    // keeps it sound — with the indices pairwise distinct the writes may cover
    // the whole domain and no off-chain index need exist.
    let unconditional = index_sort_has_more_than(manager, domain, writes);
    let guarded =
        !unconditional && writes >= 2 && index_sort_has_more_than(manager, domain, writes - 1);
    if !unconditional && !guarded {
        return;
    }
    let off_chain = off_chain_witness(manager, a, b, domain);
    let all_distinct = guarded.then(|| manager.mk_distinct(chain_indices.clone()));
    for &store_index in &chain_indices {
        let same = manager.mk_eq(off_chain, store_index);
        let differs = manager.mk_not(same);
        let instance = match all_distinct {
            Some(distinct) => manager.mk_or([distinct, differs]),
            None => differs,
        };
        candidates.push(instance);
    }
    // Select congruence at the new index, so the two sides are compared there:
    // the disequalities above only say *where* the index is not.
    let read_a = manager.mk_select(a, off_chain);
    let read_b = manager.mk_select(b, off_chain);
    let reads_eq = manager.mk_eq(read_a, read_b);
    let eq_ab = manager.mk_eq(a, b);
    let cong = manager.mk_implies(eq_ab, reads_eq);
    candidates.push(cong);
}

/// Every index relevant to comparing `side` with the other side of an array
/// equality: the indices read on `side` itself and on every array term of its
/// store chain, together with each chain link's own store index.
///
/// Reading only `side`'s *own* indices is not enough (`#P2b-37`): in
/// `(= (store (store (store brr #b0 #b0) i w) #b1 x) ((as const A) #b1))` the
/// index that refutes the equality is `#b0`, which is a read index of the
/// innermost store rather than of the chain as a whole.  Following the chain
/// keeps the set local to the pair — it is bounded by the chain length, not by
/// the formula's whole index vocabulary.
fn collect_pair_indices(
    manager: &TermManager,
    collected: &ArrayStructure,
    side: TermId,
    out: &mut Vec<TermId>,
) {
    let mut current = side;
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    for depth in 0..MAX_STORE_CHAIN_DEPTH {
        if !seen.insert(current) {
            return;
        }
        if let Some(idxs) = collected.read_indices.get(&current) {
            for &idx in idxs {
                // A *synthetic* index belongs to the pair that minted it.
                // Sharing every pair's extensionality witness with every other
                // pair cost `O(n^2)` congruence instances — a formula with
                // nine foreign array terms did not finish — and each pair
                // already has its own witness, so nothing is lost.
                //
                // The one exception is an *off-chain* index read directly on
                // this pair's own side: that index carries the "outside both
                // chains" fact its own pair established, and a second
                // description of the same array needs it.  `(= (store a i #b1)
                // ((as const A) #b1))` together with `(= a <chain over
                // ((as const A) #b0)>)` is refuted only where the first pair's
                // off-chain index reaches the second pair's congruence.  Those
                // indices are rare — one per pair whose chains have different
                // bases — so admitting them costs nothing measurable.
                if is_synthetic_index(idx, manager)
                    && !(depth == 0 && is_off_chain_index(idx, manager))
                {
                    continue;
                }
                if !out.contains(&idx) {
                    out.push(idx);
                }
            }
        }
        let store_term = if as_store(current, manager).is_some() {
            current
        } else if let Some(&aliased) = collected.aliases.get(&current) {
            aliased
        } else {
            return;
        };
        let Some((base, store_index, _)) = as_store(store_term, manager) else {
            return;
        };
        if !out.contains(&store_index) {
            out.push(store_index);
        }
        current = base;
    }
}

/// The store indices written along `side`'s *syntactic* store chain,
/// innermost last.
///
/// These are the indices an off-chain Skolem index has to differ from; see the
/// `off_chain_witness` block in [`build_extensionality_and_congruence`].
///
/// Deliberately does **not** follow a `var = store(..)` alias, unlike the
/// read-index walk next door.  The lemma this feeds constrains a fresh index
/// away from the writes of *this* equality, and it is satisfiable as long as
/// the index sort has more elements than this equality has writes.  Counting
/// an aliased chain's writes too makes the cardinality guard refuse a rule
/// that was available: `(= (store a i1 #b1) ((as const A) #b1))` alongside
/// `(= a (store (store ((as const A) #b0) i0 #b1) i0 #b0))` has one write of
/// its own over a two-element index sort, so an off-chain index exists — and
/// it is the only thing that refutes the pair — while the alias made the chain
/// look two writes long and the guard blocked it.
fn collect_chain_store_indices(manager: &TermManager, side: TermId, out: &mut Vec<TermId>) {
    let mut current = side;
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    for _ in 0..MAX_STORE_CHAIN_DEPTH {
        if !seen.insert(current) {
            return;
        }
        let Some((base, store_index, _)) = as_store(current, manager) else {
            return;
        };
        if !out.contains(&store_index) {
            out.push(store_index);
        }
        current = base;
    }
}

/// The array a store chain is laid over: follow `side` through its `store`
/// terms (and through an asserted `var = store(..)` alias) until a term that
/// is neither.
fn chain_base(manager: &TermManager, collected: &ArrayStructure, side: TermId) -> TermId {
    let mut current = side;
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    for _ in 0..MAX_STORE_CHAIN_DEPTH {
        if !seen.insert(current) {
            return current;
        }
        let store_term = if as_store(current, manager).is_some() {
            current
        } else if let Some(&aliased) = collected.aliases.get(&current) {
            aliased
        } else {
            return current;
        };
        let Some((base, _, _)) = as_store(store_term, manager) else {
            return current;
        };
        current = base;
    }
    current
}

/// Whether the index sort `sort` provably has more than `count` elements.
///
/// A *lower* bound is what the caller needs, so every arm either states one it
/// can prove or gives up: an uninterpreted sort, a datatype, a sort parameter
/// and a floating-point sort all answer `false` however large they may really
/// be, because minting an off-chain index on a sort that turns out to be too
/// small would make a satisfiable formula `unsat`.
fn index_sort_has_more_than(manager: &TermManager, sort: SortId, count: usize) -> bool {
    let Some(bound) = index_sort_lower_bound(manager, sort) else {
        return false;
    };
    bound > count as u128
}

/// A provable lower bound on the number of distinct elements of `sort`, or
/// `None` when none is known.  `u128::MAX` stands for "unbounded".
fn index_sort_lower_bound(manager: &TermManager, sort: SortId) -> Option<u128> {
    let kind = &manager.sorts.get(sort)?.kind;
    match kind {
        SortKind::Bool => Some(2),
        // `2^width`, saturating: a width at or above 127 is unbounded for
        // every purpose this bound serves.
        SortKind::BitVec(width) => Some(if *width >= 127 {
            u128::MAX
        } else {
            1u128 << *width
        }),
        SortKind::Int | SortKind::Real | SortKind::String => Some(u128::MAX),
        SortKind::RoundingMode => Some(5),
        // `|R|^|D|`, monotone in both, so lower bounds compose.
        SortKind::Array { domain, range } => {
            let domain_bound = index_sort_lower_bound(manager, *domain)?;
            let range_bound = index_sort_lower_bound(manager, *range)?;
            if range_bound < 2 {
                return Some(range_bound);
            }
            if domain_bound >= 127 || range_bound == u128::MAX {
                return Some(u128::MAX);
            }
            let exponent = u32::try_from(domain_bound).ok()?;
            Some(range_bound.checked_pow(exponent).unwrap_or(u128::MAX))
        }
        SortKind::FloatingPoint { .. }
        | SortKind::Uninterpreted(_)
        | SortKind::Parameter(_)
        | SortKind::Parametric { .. }
        | SortKind::Datatype(_) => None,
    }
}

/// Materialise (interning is idempotent) the deterministic *off-chain* index
/// variable of the unordered array pair `{a, b}` — a Skolem index the pair's
/// disequality constraints keep off both store chains.
fn off_chain_witness(manager: &mut TermManager, a: TermId, b: TermId, domain: SortId) -> TermId {
    let (lo, hi) = if a.raw() <= b.raw() {
        (a.raw(), b.raw())
    } else {
        (b.raw(), a.raw())
    };
    // `OFF_CHAIN_PREFIX` contains a backslash, so the name is unspellable in
    // both SMT-LIB symbol forms (see `oxiz_core::smtlib::ARRAY_OFF_CHAIN_PREFIX`).
    let name = format!("{OFF_CHAIN_PREFIX}{lo}!{hi}");
    manager.mk_var(&name, domain)
}

/// Name prefix of an extensionality witness index.
///
/// Reserved by the backslash it carries, so no SMT-LIB script can declare a
/// constant that interns to the same term — see
/// [`oxiz_core::smtlib::ARRAY_EXT_WITNESS_PREFIX`].
const EXT_WITNESS_PREFIX: &str = oxiz_core::smtlib::ARRAY_EXT_WITNESS_PREFIX;

/// Name prefix of an off-chain Skolem index.
///
/// Reserved by the backslash it carries — and here it is reserved for
/// *soundness*: the off-chain rule constrains the symbol itself, so a script
/// able to spell the name inherits those constraints and a satisfiable formula
/// answers `unsat`.  See [`oxiz_core::smtlib::ARRAY_OFF_CHAIN_PREFIX`].
const OFF_CHAIN_PREFIX: &str = oxiz_core::smtlib::ARRAY_OFF_CHAIN_PREFIX;

/// Whether `term` is a synthetic index variable this module minted for some
/// array pair.
///
/// Such an index belongs to *its* pair: instantiating select congruence for
/// every other pair at it as well is what turned `n` array pairs into `O(n^2)`
/// lemmas, because every witness read registers its index on both its arrays
/// and the congruence family then reads the accumulated set.  Each pair
/// already gets its own witness, so the theory content is unchanged.
fn is_off_chain_index(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|data| match data.kind {
        TermKind::Var(name) => manager.resolve_str(name).starts_with(OFF_CHAIN_PREFIX),
        _ => false,
    })
}

fn is_synthetic_index(term: TermId, manager: &TermManager) -> bool {
    let Some(data) = manager.get(term) else {
        return false;
    };
    let TermKind::Var(name) = data.kind else {
        return false;
    };
    let name = manager.resolve_str(name);
    name.starts_with(EXT_WITNESS_PREFIX) || name.starts_with(OFF_CHAIN_PREFIX)
}

/// Materialise (interning is idempotent) a deterministic extensionality witness
/// index variable for the unordered array pair `{a, b}`.  Using a name derived
/// from the two term ids keeps the witness stable across refinement rounds, so
/// the extensionality lemma for a given pair is asserted exactly once instead of
/// spawning a fresh variable each round.
fn extensionality_witness(
    manager: &mut TermManager,
    a: TermId,
    b: TermId,
    domain: SortId,
) -> TermId {
    let (lo, hi) = if a.raw() <= b.raw() {
        (a.raw(), b.raw())
    } else {
        (b.raw(), a.raw())
    };
    // `EXT_WITNESS_PREFIX` contains a backslash, so the name is unspellable in
    // both SMT-LIB symbol forms (see `oxiz_core::smtlib::ARRAY_EXT_WITNESS_PREFIX`).
    let name = format!("{EXT_WITNESS_PREFIX}{lo}!{hi}");
    manager.mk_var(&name, domain)
}

/// The function symbol the parser gives the SMT-LIB array constant
/// `((as const (Array D R)) d)`.
///
/// There is no dedicated term kind for it: `smtlib/parser/terms.rs`
/// (`Head::Qualified`) turns a qualified identifier into an ordinary
/// uninterpreted application.  The name it interns is
/// [`oxiz_core::smtlib::CONST_ARRAY_FUNC`], a *reserved* symbol containing a
/// backslash: SMT-LIB 2.6 excludes `\` from a simple symbol's character set
/// and forbids it inside a quoted symbol (section 3.1, enforced by the lexer),
/// so no script can declare or apply the same name.  The printers render the
/// application back as `((as const (Array D R)) d)`.
pub(crate) use oxiz_core::smtlib::CONST_ARRAY_FUNC;

/// The default value of an array constant `((as const (Array D R)) d)`, or
/// `None` when `term` is not one.
///
/// # Why the recognition is structural as well as by name (`#P2b-36`)
///
/// `|(as const)|` is a legal quoted SMT-LIB symbol and the lexer strips the
/// bars, so before the reserved name a user-declared function could intern to
/// exactly the string the parser gave array constants — `(declare-fun
/// |(as const)| ((_ BitVec 8)) (Array (_ BitVec 8) (_ BitVec 8)))` parsed and
/// printed as `((as const) #x00)`.  Reading such an application as an array
/// constant would answer `unsat` for a satisfiable formula, which is the same
/// class of defect this axiom exists to remove.  Every structural property of
/// the array constant is therefore checked as well: exactly one argument, an
/// array sort, and an argument whose sort is that array's *range*.
///
/// The ambiguity itself is closed one level up, by the reserved name: the
/// parser interns an array constant under [`CONST_ARRAY_FUNC`], which contains
/// a backslash and so is unspellable in either SMT-LIB symbol form, and it
/// refuses the name outright if one ever reaches it.  A script may therefore
/// declare `|(as const)|` — it is an ordinary function, interned under the
/// ordinary string `(as const)` — while a genuine array constant in the same
/// script is still decided.  The four structural checks stay as belt and
/// braces: they cost one sort lookup and they are what keeps a *builder-API*
/// caller that interns the reserved name by hand from being read as an array
/// constant of the wrong shape.
pub(crate) fn const_array_default(term: TermId, manager: &TermManager) -> Option<TermId> {
    let data = manager.get(term)?;
    let TermKind::Apply { func, args } = &data.kind else {
        return None;
    };
    if args.len() != 1 || manager.resolve_str(*func) != CONST_ARRAY_FUNC {
        return None;
    }
    let SortKind::Array { range, .. } = manager.sorts.get(data.sort)?.kind else {
        return None;
    };
    let default = *args.first()?;
    if manager.get(default)?.sort != range {
        return None;
    }
    Some(default)
}

/// If `term` is a `store`, return `(base, index, value)`.
fn as_store(term: TermId, manager: &TermManager) -> Option<(TermId, TermId, TermId)> {
    match manager.get(term)?.kind {
        TermKind::Store(base, index, value) => Some((base, index, value)),
        _ => None,
    }
}

/// Whether `term` has an array sort.
fn is_array_sorted(term: TermId, manager: &TermManager) -> bool {
    manager
        .get(term)
        .and_then(|d| manager.sorts.get(d.sort))
        .is_some_and(|s| matches!(s.kind, SortKind::Array { .. }))
}

/// The domain (index) sort of `term`'s array sort, if `term` is array-sorted.
/// Whether `term`'s index sort is one the extensionality family decides by
/// enumeration rather than by a Skolem witness index.
///
/// The three lazy phases (3a array-constant witness congruence, 3b chain-index
/// congruence, 3c off-chain Skolem index) all exist to mint or reach an index
/// the eager family did not; for an enumerated pair the eager family reached
/// *every* index there is, so all three have nothing left to say. See
/// [`ARRAY_INDEX_ENUMERATION_LIMIT`].
fn pair_is_enumerated(array: TermId, manager: &mut TermManager) -> bool {
    array_domain(array, manager)
        .and_then(|domain| enumerable_index_values(manager, domain))
        .is_some_and(|values| !values.is_empty())
}

fn array_domain(term: TermId, manager: &TermManager) -> Option<SortId> {
    let sort = manager.get(term)?.sort;
    match manager.sorts.get(sort)?.kind {
        SortKind::Array { domain, .. } => Some(domain),
        _ => None,
    }
}

/// Immediate sub-terms of a term kind that the structural walk descends into
/// for the operators `collect_array_structure` does not handle itself.
///
/// Exhaustive over the ground term language, by delegation to
/// [`super::term_walk::collect_structural_children`] — the single
/// every-sub-term walk the crate keeps.  The hand-written list this replaced
/// named only `not`/`and`/`or`/`distinct`/`=>`/`xor`/`ite`/`Apply`, with
/// `_ => Vec::new()` for the rest: a `select` under `bvadd`, `bvnot`,
/// `concat`, `bvult`, `+` or `<` was invisible to the instantiator, so no
/// read-over-write lemma was ever asserted for it and the read stayed a free
/// leaf — the wrong `sat` of `#P2b-32`.  Any future `TermKind` reaches this
/// walk through `collect_structural_children`, which is what makes the
/// omission unrepeatable.
///
/// Binders are the one deliberate exception, and it is the behaviour the old
/// list had: a `select` under a `forall`, `exists`, `let` or `match` may
/// mention a bound variable, and a ground lemma over a bound variable is an
/// instance of nothing.  This walk stays on the ground fragment.
///
/// That exclusion used to say "quantified array reasoning is MBQI's job", and
/// that sentence was false for as long as it stood.  MBQI hands its instances
/// to `Solver::encode` directly, not to `Solver::assert`, so nothing in the
/// system ever collected their array structure and a read that was first
/// ground after substitution stayed a free value of the element sort — a
/// wrong `sat` from four lines, with a `(get-value)` that contradicted it.
/// The claim is now true by construction rather than by assertion:
/// `Solver::prepare_ground_instance` registers every instance in
/// `Solver::ground_array_roots`, `instantiate_array_axioms` walks that set
/// as a root set beside `self.assertions`, and `Solver::array_refinement_round`
/// runs on the quantified candidate-model path as well as the ground one.  A
/// binder is skipped here because its body is not yet ground, not because
/// someone else is expected to look at it.
pub(crate) fn ground_children(kind: &TermKind, out: &mut Vec<TermId>) {
    match kind {
        TermKind::Forall { .. }
        | TermKind::Exists { .. }
        | TermKind::Let { .. }
        | TermKind::Match { .. } => {}
        _ => super::term_walk::collect_structural_children(kind, out),
    }
}

mod families;
use families::*;

#[cfg(test)]
mod tests;
