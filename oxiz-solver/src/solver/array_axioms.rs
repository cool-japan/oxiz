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
        let roots: Vec<TermId> = self
            .assertions
            .iter()
            .copied()
            .chain(self.array_axiom_instances.iter().copied())
            .collect();

        let mut collected = ArrayStructure::default();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        for &root in &roots {
            collect_array_structure(root, manager, &mut visited, &mut collected);
        }

        if collected.selects.is_empty()
            && collected.eq_pairs.is_empty()
            && collected.stores.is_empty()
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
        build_const_array_reads(manager, &collected, &mut candidates);
        build_extensionality_and_congruence(manager, &collected, &mut candidates);

        // ---- Phase 3: filter (dedup + model) and assert -----------------
        if self.assert_new_instances(&candidates, manager) {
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
        let foreign_pairs = self.foreign_pairs_in_different_classes(&collected);
        if foreign_pairs.is_empty() {
            return false;
        }
        let mut extra: Vec<TermId> = Vec::new();
        for (a, b) in foreign_pairs {
            push_witness_lemma(manager, a, b, &mut extra);
        }
        self.assert_new_instances(&extra, manager)
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
            added = true;
        }

        if std::env::var("OXIZ_DBG_ARRAY").is_ok() {
            eprintln!(
                "round: candidates={} instances={} added={}",
                candidates.len(),
                self.array_axiom_instances.len(),
                added
            );
        }
        added
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
    ) -> Vec<(TermId, TermId)> {
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
    /// `distinct` over array-sorted operands.
    eq_pairs: Vec<(TermId, TermId)>,
    /// `array_variable -> store_term` for every asserted `var = store(...)`.
    aliases: FxHashMap<TermId, TermId>,
    /// Distinct indices read on each array operand (for select congruence).
    read_indices: FxHashMap<TermId, Vec<TermId>>,
    /// `(store_term, index, value)` for every `store` encountered, so the
    /// store's own-index read can be registered even when the script never
    /// spells it (`#P2b-37`).
    stores: Vec<(TermId, TermId, TermId)>,
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
}

impl ArrayStructure {
    /// Record `term` as occupying a foreign position, if it is array-sorted
    /// and not already recorded.  `shared` distinguishes a genuinely shared
    /// position from an array (dis)equality operand — see
    /// [`ArrayStructure::shared_foreign`].
    fn note_foreign(&mut self, term: TermId, manager: &TermManager, shared: bool) {
        if !is_array_sorted(term, manager) {
            return;
        }
        if !self.foreign.contains(&term) {
            self.foreign.push(term);
        }
        if shared {
            self.shared_foreign.insert(term);
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
                out.note_foreign(*value, manager, true);
                stack.push(*value);
                stack.push(*index);
                stack.push(*base);
            }
            TermKind::Eq(lhs, rhs) => {
                // Record an array-sorted equality atom (either polarity: the
                // extensionality / congruence lemmas are valid regardless).
                if lhs != rhs && is_array_sorted(*lhs, manager) && is_array_sorted(*rhs, manager) {
                    out.eq_pairs.push((*lhs, *rhs));
                    out.note_foreign(*lhs, manager, false);
                    out.note_foreign(*rhs, manager, false);
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
                for (position, &lhs) in args.iter().enumerate() {
                    if !is_array_sorted(lhs, manager) {
                        continue;
                    }
                    out.note_foreign(lhs, manager, false);
                    for &rhs in args.iter().skip(position + 1) {
                        if lhs != rhs && is_array_sorted(rhs, manager) {
                            out.eq_pairs.push((lhs, rhs));
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
                for &arg in args {
                    out.note_foreign(arg, manager, true);
                }
                stack.extend(args.iter().rev().copied());
            }
            TermKind::Ite(cond, then_branch, else_branch) => {
                out.note_foreign(*then_branch, manager, true);
                out.note_foreign(*else_branch, manager, true);
                stack.push(*else_branch);
                stack.push(*then_branch);
                stack.push(*cond);
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

/// Push the extensionality witness lemma `a = b ∨ select(a,k) != select(b,k)`
/// for the unordered array pair `{a, b}`, with the deterministic per-pair
/// witness index `k`.
///
/// The lemma is a theorem of the extensional array theory in both directions:
/// asserted `a != b` forces a concrete differing index, and asserted `a = b`
/// leaves it vacuous.
fn push_witness_lemma(
    manager: &mut TermManager,
    a: TermId,
    b: TermId,
    candidates: &mut Vec<TermId>,
) {
    let Some(domain) = array_domain(a, manager) else {
        return;
    };
    let witness = extensionality_witness(manager, a, b, domain);
    let read_a = manager.mk_select(a, witness);
    let read_b = manager.mk_select(b, witness);
    let reads_eq = manager.mk_eq(read_a, read_b);
    let reads_diff = manager.mk_not(reads_eq);
    let eq_ab = manager.mk_eq(a, b);
    let ext = manager.mk_or([eq_ab, reads_diff]);
    candidates.push(ext);
}

/// Build read-over-write instances for every collected `select`, following each
/// read all the way down its store chain.
///
/// The axiom is emitted as its two case-split implications rather than a single
/// `ite`-valued equality, because the arithmetic / EUF theory solvers reduce a
/// guarded equality (`cond ⇒ x = y`) directly, whereas a term-level `ite`
/// operand of an equality would be handed to them opaque.
///
///   * RoW-1: `store_idx = index  ⇒  select_term = stored_val`
///   * RoW-2: `store_idx != index ⇒  select_term = select(base, index)`
fn build_read_over_write(
    manager: &mut TermManager,
    collected: &ArrayStructure,
    candidates: &mut Vec<TermId>,
) {
    for &(select_term, array, index) in &collected.selects {
        emit_read_chain(manager, collected, select_term, array, index, candidates);
    }
}

/// Emit the read-over-write pair for `select(array, index)` and keep descending
/// into the store's base for as long as that base is itself a store (directly,
/// or through an asserted `base = store(..)` alias).
///
/// Each level's pair is a self-contained theorem — it mentions only that
/// level's store and needs only that level's alias equality as a guard — so
/// descending adds no assumption and the lemmas stay valid however the search
/// later assigns the aliases.
fn emit_read_chain(
    manager: &mut TermManager,
    collected: &ArrayStructure,
    select_term: TermId,
    array: TermId,
    index: TermId,
    candidates: &mut Vec<TermId>,
) {
    let mut select_term = select_term;
    let mut array = array;
    // Arrays already reduced on this chain.  An alias cycle would otherwise
    // walk the same two arrays until the depth budget ran out, re-deriving
    // lemmas the dedup set would then discard.
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    for _ in 0..MAX_STORE_CHAIN_DEPTH {
        if !seen.insert(array) {
            return;
        }
        // Resolve this level to a store term, plus the alias equality (if any)
        // that has to guard the lemma.
        let (store_term, alias_eq) = if as_store(array, manager).is_some() {
            (array, None)
        } else if let Some(&aliased) = collected.aliases.get(&array) {
            (aliased, Some(manager.mk_eq(array, aliased)))
        } else {
            return;
        };
        let Some((base, store_idx, stored_val)) = as_store(store_term, manager) else {
            return;
        };
        let (row1, row2) =
            row_implications(manager, select_term, store_idx, stored_val, base, index);
        match alias_eq {
            // An asserted `array = store(...)` makes the axiom apply to the
            // *name*, but only under that equality — guarding keeps the lemma a
            // universally-valid theorem (`array = store(...) ∧ cond ⇒ ...`).
            Some(eq) => {
                let g1 = manager.mk_implies(eq, row1);
                let g2 = manager.mk_implies(eq, row2);
                candidates.push(g1);
                candidates.push(g2);
            }
            None => {
                candidates.push(row1);
                candidates.push(row2);
            }
        }
        // RoW-2 introduced `select(base, index)`; reduce it here rather than
        // waiting for the next refinement round to notice it.
        select_term = manager.mk_select(base, index);
        array = base;
    }
}

/// Build the two read-over-write case-split implications for a
/// `select(store(base, store_idx, stored_val), index)` read.
fn row_implications(
    manager: &mut TermManager,
    select_term: TermId,
    store_idx: TermId,
    stored_val: TermId,
    base: TermId,
    index: TermId,
) -> (TermId, TermId) {
    let idx_eq = manager.mk_eq(store_idx, index);
    // RoW-1: (store_idx = index) ⇒ (select_term = stored_val)
    let hit = manager.mk_eq(select_term, stored_val);
    let row1 = manager.mk_implies(idx_eq, hit);
    // RoW-2: (store_idx != index) ⇒ (select_term = select(base, index))
    let idx_neq = manager.mk_not(idx_eq);
    let base_read = manager.mk_select(base, index);
    let miss = manager.mk_eq(select_term, base_read);
    let row2 = manager.mk_implies(idx_neq, miss);
    (row1, row2)
}

/// Build extensionality and select-congruence instances for every collected
/// array-sorted equality atom.
fn build_extensionality_and_congruence(
    manager: &mut TermManager,
    collected: &ArrayStructure,
    candidates: &mut Vec<TermId>,
) {
    for &(a, b) in &collected.eq_pairs {
        // Extensionality: a = b ∨ select(a,k) != select(b,k), with a fresh but
        // deterministic witness index per unordered pair.
        push_witness_lemma(manager, a, b, candidates);

        // Select congruence: a = b ⇒ select(a,j) = select(b,j) for every index
        // relevant to comparing the two sides.
        let mut indices: Vec<TermId> = Vec::new();
        // The pair's own witness index is one of them (`#P2b-37`).  Without it
        // the two lemmas never meet: extensionality speaks only about `k` and
        // congruence only about the script's indices, so
        // `(= ((as const A) d1) ((as const A) d2))` — two array constants with
        // different defaults, asserted equal, and no `select` anywhere — had
        // no instance that could see both defaults, and answered `sat`.  With
        // the witness read in the congruence family the const-read axiom
        // decides `select(c1,k) = d1` and `select(c2,k) = d2` on the next
        // round and the equality is refuted; in the other polarity the
        // witness lemma itself refutes an asserted `distinct` between two
        // constants with the *same* default.
        if let Some(domain) = array_domain(a, manager) {
            let witness = extensionality_witness(manager, a, b, domain);
            indices.push(witness);
        }
        collect_pair_indices(manager, collected, a, &mut indices);
        collect_pair_indices(manager, collected, b, &mut indices);

        // An index provably *outside* both store chains (`#P2b-37`).
        //
        // `(= (store ((as const A) #b0) i #b1) ((as const A) #b1))` is unsat
        // because the two sides differ at every index other than `i`, and over
        // a two-element index sort such an index exists.  No index in the
        // formula names it, though: every congruence instance above is at an
        // index the script or a store already mentions, and at `i` the two
        // sides agree.  So the pair gets its own Skolem index `d`, asserted
        // different from every store index on either chain — a constraint on a
        // *fresh* symbol that any model can satisfy as long as the index sort
        // has more elements than the chains have writes, which is exactly the
        // cardinality condition checked here.  Without the condition the
        // constraint would be unsatisfiable on a sort too small to hold an
        // off-chain index, and asserting it would report `unsat` for a
        // satisfiable formula.
        let mut chain_indices: Vec<TermId> = Vec::new();
        collect_chain_store_indices(manager, collected, a, &mut chain_indices);
        collect_chain_store_indices(manager, collected, b, &mut chain_indices);
        if !chain_indices.is_empty()
            && let Some(domain) = array_domain(a, manager)
            && index_sort_has_more_than(manager, domain, chain_indices.len())
        {
            let off_chain = off_chain_witness(manager, a, b, domain);
            for &store_index in &chain_indices {
                let same = manager.mk_eq(off_chain, store_index);
                let differs = manager.mk_not(same);
                candidates.push(differs);
            }
            if !indices.contains(&off_chain) {
                indices.push(off_chain);
            }
        }

        for idx in indices {
            let read_a = manager.mk_select(a, idx);
            let read_b = manager.mk_select(b, idx);
            let reads_eq = manager.mk_eq(read_a, read_b);
            let eq_ab = manager.mk_eq(a, b);
            let cong = manager.mk_implies(eq_ab, reads_eq);
            candidates.push(cong);
        }
    }
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
    for _ in 0..MAX_STORE_CHAIN_DEPTH {
        if !seen.insert(current) {
            return;
        }
        if let Some(idxs) = collected.read_indices.get(&current) {
            for &idx in idxs {
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

/// The store indices written along `side`'s store chain, innermost last.
///
/// These are the indices an off-chain Skolem index has to differ from; see the
/// `off_chain_witness` block in [`build_extensionality_and_congruence`].
fn collect_chain_store_indices(
    manager: &TermManager,
    collected: &ArrayStructure,
    side: TermId,
    out: &mut Vec<TermId>,
) {
    let mut current = side;
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    for _ in 0..MAX_STORE_CHAIN_DEPTH {
        if !seen.insert(current) {
            return;
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
    // The `!oxiz!off!` prefix cannot collide with an SMT-LIB source symbol.
    let name = format!("!oxiz!off!{lo}!{hi}");
    manager.mk_var(&name, domain)
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
    // The `!oxiz!ext!` prefix cannot collide with an SMT-LIB source symbol.
    let name = format!("!oxiz!ext!{lo}!{hi}");
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
pub(super) fn const_array_default(term: TermId, manager: &TermManager) -> Option<TermId> {
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

/// Build the array-constant read instances:
/// `select(((as const (Array D R)) d), i) = d` for every collected read whose
/// array operand is an array constant.
///
/// # Why this family was missing (`#P2b-36`)
///
/// An array constant is an opaque `Apply` to every part of the solver, so a
/// read of one was a free leaf: `(= (select ((as const (Array (_ BitVec 8) (_
/// BitVec 8))) #x00) #x00) #x05)` answered `sat` on 0.3.3 and on every tree
/// before this, as did the `Int` spelling, the same read under `bvadd`, and
/// the read wrapped in an uninterpreted function.  The axiom is unconditional
/// — an array constant's value at *every* index is its default — so the
/// instance needs no guard, unlike the alias-guarded read-over-write pairs.
///
/// It composes with the two families around it rather than duplicating them:
/// a read over a `store` chain that bottoms out at an array constant is
/// reduced by RoW-2 to a read *of* the constant, which the next refinement
/// round collects and this family then decides; and `arr = ((as const …) d)`
/// with a read on `arr` is carried across by select congruence to a read of
/// the constant, likewise decided here on the following round.
fn build_const_array_reads(
    manager: &mut TermManager,
    collected: &ArrayStructure,
    candidates: &mut Vec<TermId>,
) {
    for &(select_term, array, _) in &collected.selects {
        let Some(default) = const_array_default(array, manager) else {
            continue;
        };
        let read_is_default = manager.mk_eq(select_term, default);
        candidates.push(read_is_default);
    }
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
/// instance of nothing.  Quantified array reasoning is MBQI's job; this walk
/// stays on the ground fragment.
pub(super) fn ground_children(kind: &TermKind, out: &mut Vec<TermId>) {
    match kind {
        TermKind::Forall { .. }
        | TermKind::Exists { .. }
        | TermKind::Let { .. }
        | TermKind::Match { .. } => {}
        _ => super::term_walk::collect_structural_children(kind, out),
    }
}

#[cfg(test)]
mod budget_honesty_tests {
    use super::*;
    use crate::solver::types::SolverResult;
    use oxiz_core::ast::TermManager;

    /// An exhausted instance budget must never be reported as a model.
    ///
    /// The cap makes `instantiate_array_axioms` return `false`, which is the
    /// same value it returns for "this candidate satisfies every axiom" — the
    /// one answer that licenses `Sat`. Pre-loading the dedup set to the cap
    /// simulates a formula that used the whole budget, and the verdict on a
    /// formula that genuinely needs an array lemma must then be `Unknown`
    /// rather than the `sat` the unchecked `false` would have produced.
    #[test]
    fn an_exhausted_instance_budget_is_unknown_not_sat() {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);

        // `(not (= (store (store a 1 x) 2 y) (store (store a 2 y) 1 x)))` —
        // unsat, and only extensionality can show it.
        let a = tm.mk_var("a", array_sort);
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let one = tm.mk_int(1);
        let two = tm.mk_int(2);
        let lhs = {
            let inner = tm.mk_store(a, one, x);
            tm.mk_store(inner, two, y)
        };
        let rhs = {
            let inner = tm.mk_store(a, two, y);
            tm.mk_store(inner, one, x)
        };
        let eq = tm.mk_eq(lhs, rhs);
        let goal = tm.mk_not(eq);
        solver.assert(goal, &mut tm);

        // Fill the dedup set to the cap with throwaway ids so the very first
        // instantiation call is refused by the budget check.
        for i in 0..MAX_ARRAY_AXIOM_INSTANCES {
            let filler = tm.mk_var(&format!("!filler!{i}"), int_sort);
            solver.array_axiom_instances.insert(filler);
        }

        let verdict = solver.check(&mut tm);
        assert_eq!(
            verdict,
            SolverResult::Unknown,
            "an exhausted array-axiom budget must be reported honestly, never as sat"
        );
        assert!(
            solver.array_axioms_incomplete,
            "the budget exhaustion must be recorded"
        );
    }

    /// The same formula with the budget available is decided, so the gate above
    /// is not simply suppressing every array answer.
    #[test]
    fn an_available_budget_still_decides_the_same_formula() {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let a = tm.mk_var("a", array_sort);
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let one = tm.mk_int(1);
        let two = tm.mk_int(2);
        let lhs = {
            let inner = tm.mk_store(a, one, x);
            tm.mk_store(inner, two, y)
        };
        let rhs = {
            let inner = tm.mk_store(a, two, y);
            tm.mk_store(inner, one, x)
        };
        let eq = tm.mk_eq(lhs, rhs);
        let goal = tm.mk_not(eq);
        solver.assert(goal, &mut tm);

        assert_eq!(
            solver.check(&mut tm),
            SolverResult::Unsat,
            "store commutativity is refutable when the budget is available"
        );
        assert!(
            !solver.array_axioms_incomplete,
            "nothing near the cap was needed"
        );
    }
}

#[cfg(test)]
mod s8_iterative_tests {
    use super::*;
    use oxiz_core::ast::TermManager;

    /// Nesting depth that would overflow the native stack under the previous
    /// recursive walk; the assertion is simply that the call **returns**.
    ///
    /// This depth and [`SMALL_STACK`] were scaled down together by a factor
    /// of 8 (from 60 000 on 1 MiB).  What the test pins is the ~17 bytes of
    /// stack available per level — far under any native frame — not the
    /// absolute depth, and the smaller pair costs a fraction of the memory
    /// the interner has to keep live.  Never raise one without the other.
    const DEEP: usize = 7_500;

    /// Worker stack for the deep-nesting test; see [`DEEP`].
    const SMALL_STACK: usize = 1 << 17;

    /// Build `store(store(...store(a, i, v)..., i, v), i, v)`, `depth` levels.
    fn deep_store_chain(tm: &mut TermManager, depth: usize) -> (TermId, TermId) {
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let base = tm.mk_var("a", array_sort);
        let idx = tm.mk_int(num_bigint::BigInt::from(1));
        let val = tm.mk_int(num_bigint::BigInt::from(7));
        let mut current = base;
        for _ in 0..depth {
            current = tm.mk_store(current, idx, val);
        }
        (current, idx)
    }

    #[test]
    fn s8_collect_array_structure_deep_store_chain_returns() {
        // A 128 KiB stack: the recursive version could not survive `DEEP`
        // frames, so returning at all is the proof of the conversion.
        let handle = std::thread::Builder::new()
            .stack_size(SMALL_STACK)
            .spawn(|| {
                let mut tm = TermManager::new();
                let (deep, idx) = deep_store_chain(&mut tm, DEEP);
                let select = tm.mk_select(deep, idx);
                let mut visited = FxHashSet::default();
                let mut out = ArrayStructure::default();
                collect_array_structure(select, &tm, &mut visited, &mut out);
                out.selects.len()
            })
            .expect("spawn deep-nesting worker");
        assert_eq!(handle.join().ok(), Some(1));
    }

    /// A doubling DAG: without the `visited` set this would expand
    /// exponentially instead of completing immediately.
    #[test]
    fn s8_collect_array_structure_shared_dag_completes() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let mut current = tm.mk_var("x", int_sort);
        for _ in 0..55 {
            current = tm.mk_add(vec![current, current]);
        }
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(current, &tm, &mut visited, &mut out);
        assert!(out.selects.is_empty());
    }

    /// Semantic pin: the walk still records selects, read indices, array
    /// equalities and `var = store(..)` aliases, in the recursive order.
    #[test]
    fn s8_collect_array_structure_records_same_structure() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let a = tm.mk_var("a", array_sort);
        let b = tm.mk_var("b", array_sort);
        let i = tm.mk_int(num_bigint::BigInt::from(1));
        let j = tm.mk_int(num_bigint::BigInt::from(2));
        let v = tm.mk_int(num_bigint::BigInt::from(9));
        let store_a = tm.mk_store(a, i, v);
        let alias = tm.mk_eq(b, store_a);
        let sel_i = tm.mk_select(a, i);
        let sel_j = tm.mk_select(a, j);
        let sel_eq = tm.mk_eq(sel_i, sel_j);
        let both = tm.mk_and(vec![alias, sel_eq]);

        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(both, &tm, &mut visited, &mut out);

        // `b = store(a, i, v)` is recorded as an alias and as an array-sorted
        // equality pair; the two selects are recorded left to right.
        assert_eq!(out.aliases.get(&b), Some(&store_a));
        assert_eq!(out.eq_pairs, vec![(b, store_a)]);
        assert_eq!(
            out.selects,
            vec![(sel_i, a, i), (sel_j, a, j)],
            "select order must match the recursive pre-order"
        );
        assert_eq!(out.read_indices.get(&a), Some(&vec![i, j]));
    }
}

#[cfg(test)]
mod p2b32_walk_tests {
    use super::*;
    use crate::solver::types::SolverResult;
    use oxiz_core::ast::TermManager;

    /// A `select` nested under a bit-vector operator is collected (`#P2b-32`).
    /// The hand-written child list this walk used to have named only the
    /// Boolean connectives, `ite` and `Apply`, so the read under `bvadd`
    /// below was invisible and no read-over-write instance was ever built.
    #[test]
    fn a_select_under_a_bit_vector_operator_is_collected() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let array_sort = tm.sorts.array(bv8, bv8);
        let arr = tm.mk_var("arr", array_sort);
        let i = tm.mk_var("i", bv8);
        let five = tm.mk_bitvec(5, 8);
        let one = tm.mk_bitvec(1, 8);
        let six = tm.mk_bitvec(6, 8);
        let store = tm.mk_store(arr, i, five);
        let read = tm.mk_select(store, i);

        for (name, wrapped) in [
            ("bvadd", tm.mk_bv_add(read, one)),
            ("bvnot", tm.mk_bv_not(read)),
        ] {
            let goal = tm.mk_distinct([wrapped, six]);
            let mut visited = FxHashSet::default();
            let mut out = ArrayStructure::default();
            collect_array_structure(goal, &tm, &mut visited, &mut out);
            assert_eq!(out.selects, vec![(read, store, i)], "under {name}");
        }
        let comparison = tm.mk_bv_ult(six, read);
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(comparison, &tm, &mut visited, &mut out);
        assert_eq!(out.selects, vec![(read, store, i)], "under bvult");
    }

    /// The integer twin: a read under `+`, and the `<` comparison over it.
    #[test]
    fn a_select_under_an_arithmetic_operator_is_collected() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let arr = tm.mk_var("arr", array_sort);
        let i = tm.mk_var("i", int_sort);
        let five = tm.mk_int(5);
        let one = tm.mk_int(1);
        let six = tm.mk_int(6);
        let store = tm.mk_store(arr, i, five);
        let read = tm.mk_select(store, i);
        let sum = tm.mk_add([read, one]);
        let goal = tm.mk_distinct([sum, six]);
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(goal, &tm, &mut visited, &mut out);
        assert_eq!(out.selects, vec![(read, store, i)], "under +");

        let comparison = tm.mk_lt(six, read);
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(comparison, &tm, &mut visited, &mut out);
        assert_eq!(out.selects, vec![(read, store, i)], "under <");
    }

    /// A read under a binder is deliberately *not* collected: a ground lemma
    /// over a bound variable is an instance of nothing, and the old list
    /// stopped at binders too.
    #[test]
    fn a_select_under_a_binder_is_not_collected() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let array_sort = tm.sorts.array(bv8, bv8);
        let arr = tm.mk_var("arr", array_sort);
        let k = tm.mk_var("k", bv8);
        let five = tm.mk_bitvec(5, 8);
        let store = tm.mk_store(arr, k, five);
        let read = tm.mk_select(store, k);
        let body = tm.mk_eq(read, five);
        let quantified = tm.mk_forall([("k", bv8)], body);
        let mut visited = FxHashSet::default();
        let mut out = ArrayStructure::default();
        collect_array_structure(quantified, &tm, &mut visited, &mut out);
        assert!(out.selects.is_empty(), "no ground instance under a binder");
    }

    /// End to end through the builder API: the a3 shape and its integer
    /// twin are refuted, and the miss with a free second index stays `sat`.
    #[test]
    fn nested_read_over_write_is_decided() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let array_sort = tm.sorts.array(bv8, bv8);
        let arr = tm.mk_var("arr", array_sort);
        let i = tm.mk_var("i", bv8);
        let j = tm.mk_var("j", bv8);
        let five = tm.mk_bitvec(5, 8);
        let one = tm.mk_bitvec(1, 8);
        let six = tm.mk_bitvec(6, 8);
        let store = tm.mk_store(arr, i, five);

        let hit = tm.mk_select(store, i);
        let hit_sum = tm.mk_bv_add(hit, one);
        let refuted = tm.mk_distinct([hit_sum, six]);
        let mut solver = Solver::new();
        solver.assert(refuted, &mut tm);
        assert_eq!(solver.check(&mut tm), SolverResult::Unsat, "a3");

        let miss = tm.mk_select(store, j);
        let miss_sum = tm.mk_bv_add(miss, one);
        let satisfiable = tm.mk_distinct([miss_sum, six]);
        let mut solver = Solver::new();
        solver.assert(satisfiable, &mut tm);
        assert_eq!(solver.check(&mut tm), SolverResult::Sat, "a9");

        let int_sort = tm.sorts.int_sort;
        let int_array = tm.sorts.array(int_sort, int_sort);
        let iarr = tm.mk_var("iarr", int_array);
        let n = tm.mk_var("n", int_sort);
        let ifive = tm.mk_int(5);
        let ione = tm.mk_int(1);
        let isix = tm.mk_int(6);
        let istore = tm.mk_store(iarr, n, ifive);
        let iread = tm.mk_select(istore, n);
        let isum = tm.mk_add([iread, ione]);
        let irefuted = tm.mk_distinct([isum, isix]);
        let mut solver = Solver::new();
        solver.assert(irefuted, &mut tm);
        assert_eq!(solver.check(&mut tm), SolverResult::Unsat, "a12");
    }
}
