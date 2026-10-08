//! The ground lemma families the array refinement instantiates.
//!
//! Split out of `array_axioms.rs` to keep that file under the workspace's
//! 2,000-line ceiling.  Each function here *builds* candidate instances into a
//! vector; none of them asserts anything, and none of them consults the model
//! — the scheduling decisions (which pair, which round, whether the assignment
//! already satisfies the family) all live in `Solver::instantiate_array_axioms`
//! and in `Solver::pair_polarity` next door, so a reader can tell "what lemma
//! is this" from "when is it worth asserting" by which file it is in.
//!
//! Every instance built here is a theorem of the extensional array theory, so
//! asserting one never changes satisfiability — it only removes models that
//! violate array semantics, and skipping one only ever costs a round.

use super::*;

/// Push the extensionality witness lemma `a = b ∨ select(a,k) != select(b,k)`
/// for the unordered array pair `{a, b}`, with the deterministic per-pair
/// witness index `k`.
///
/// The lemma is a theorem of the extensional array theory in both directions:
/// asserted `a != b` forces a concrete differing index, and asserted `a = b`
/// leaves it vacuous.
pub(super) fn push_witness_lemma(
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
pub(super) fn build_read_over_write(
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
pub(super) fn emit_read_chain(
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
pub(super) fn row_implications(
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

/// Refute an `n`-ary `distinct` over array-sorted operands whose sort has
/// fewer than `n` elements (`#P2b-38`).
///
/// `(distinct c0 … c9)` over `(Array (_ BitVec 1) (_ BitVec 1))` is
/// unsatisfiable because that sort has exactly four elements — but proving it
/// through the extensionality family means separating all forty-five pairs,
/// which is forty-five fresh witness indices, forty-five reads on each side,
/// and a BV↔EUF partition space large enough that the exchange gives up and
/// the answer is `unknown`.  The pigeonhole argument decides it outright and
/// costs one lemma: `¬(distinct …)` is a *theorem* whenever the operand count
/// exceeds the sort's cardinality, whatever the operands are, so asserting it
/// removes exactly the models the theory has none of.
///
/// Only sorts whose size is *exactly* known take part
/// ([`array_sort_cardinality`]); an unknown or merely bounded-below size
/// yields nothing, which is the direction that can only fail to decide.
pub(super) fn build_cardinality_refutations(
    manager: &mut TermManager,
    collected: &ArrayStructure,
    candidates: &mut Vec<TermId>,
) {
    for &atom in &collected.distinct_atoms {
        let Some(TermKind::Distinct(args)) = manager.get(atom).map(|data| data.kind.clone()) else {
            continue;
        };
        let mut per_sort: FxHashMap<SortId, usize> = FxHashMap::default();
        for &arg in &args {
            if !is_array_sorted(arg, manager) {
                continue;
            }
            let Some(sort) = manager.get(arg).map(|data| data.sort) else {
                continue;
            };
            *per_sort.entry(sort).or_insert(0) += 1;
        }
        let refuted = per_sort.into_iter().any(|(sort, count)| {
            array_sort_cardinality(manager, sort)
                .is_some_and(|size| u128::from(count as u64) > size)
        });
        if refuted {
            let refutation = manager.mk_not(atom);
            candidates.push(refutation);
        }
    }
}

/// The *exact* number of elements of `sort`, or `None` when it is not known
/// exactly (an uninterpreted sort, a datatype, a floating-point sort, an
/// infinite sort) or is too large to represent.
///
/// The distinction from [`index_sort_lower_bound`] is the direction of the
/// error: that one may under-state a size (it guards a rule that needs *at
/// least* so many elements), this one must never over-state it, because the
/// rule it guards refutes a formula outright.  `u128::MAX` therefore stands
/// for "larger than any operand count", not for "unbounded".
pub(super) fn array_sort_cardinality(manager: &TermManager, sort: SortId) -> Option<u128> {
    let kind = &manager.sorts.get(sort)?.kind;
    match kind {
        SortKind::Bool => Some(2),
        SortKind::BitVec(width) => Some(if *width >= 127 {
            u128::MAX
        } else {
            1u128 << *width
        }),
        SortKind::RoundingMode => Some(5),
        SortKind::Array { domain, range } => {
            let domain_size = array_sort_cardinality(manager, *domain)?;
            let range_size = array_sort_cardinality(manager, *range)?;
            if range_size <= 1 {
                return Some(range_size);
            }
            let exponent = u32::try_from(domain_size).ok()?;
            Some(range_size.checked_pow(exponent).unwrap_or(u128::MAX))
        }
        // Infinite, or of a size this function cannot state exactly.
        SortKind::Int
        | SortKind::Real
        | SortKind::String
        | SortKind::FloatingPoint { .. }
        | SortKind::Uninterpreted(_)
        | SortKind::Parameter(_)
        | SortKind::Parametric { .. }
        | SortKind::Datatype(_) => None,
    }
}

/// Every element of `domain`, as ground value terms, when the sort is small
/// enough that writing it out is cheaper than minting a Skolem witness index
/// per array pair — otherwise `None`.
///
/// See [`ARRAY_INDEX_ENUMERATION_LIMIT`] for why this exists and where the
/// bound comes from.  The order is the numeric order of the sort's elements,
/// which is deterministic and independent of hashing.
///
/// Only sorts whose elements this function can *write* are enumerated:
/// `Bool` and a narrow `(_ BitVec w)`.  `array_sort_cardinality` also answers
/// for a nested array sort, whose elements are functions with no ground
/// spelling, so the match here is deliberately narrower than that function's.
pub(super) fn enumerable_index_values(
    manager: &mut TermManager,
    domain: SortId,
) -> Option<Vec<TermId>> {
    let size = array_sort_cardinality(manager, domain)?;
    if size > ARRAY_INDEX_ENUMERATION_LIMIT {
        return None;
    }
    let kind = manager.sorts.get(domain)?.kind.clone();
    match kind {
        SortKind::Bool => Some(vec![manager.mk_false(), manager.mk_true()]),
        SortKind::BitVec(width) => {
            let count = u32::try_from(size).ok()?;
            Some(
                (0..count)
                    .map(|value| manager.mk_bitvec(value, width))
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Build extensionality and select-congruence instances for every collected
/// array-sorted equality atom.
///
/// `polarity` reports what the candidate assignment has already committed to
/// for each pair (see [`Solver::pair_polarity`](super::Solver::pair_polarity)),
/// and each half of the family is emitted only where that commitment leaves it
/// something to say:
///
/// * the **witness lemma** `a = b ∨ select(a,k) != select(b,k)` (or, over an
///   enumerated index sort, the `|D|`-armed disjunction) is skipped for a pair
///   the assignment holds *equal*, where it is satisfied outright, and for a
///   pair the candidate model already separates at some index it published,
///   where this candidate is not claiming the two arrays are one function and
///   the fresh index `k` would only enlarge the circuit;
/// * the **congruence at `k`** is skipped for a pair the assignment holds
///   *apart*, where its antecedent is false.
///
/// Both are theorems of the array theory, so a skip can only defer work: if a
/// later candidate changes its mind about the pair, the missing half is built
/// in that round instead.
///
/// # Why this family is eager where the other four are lazy (`#P2b-46`)
///
/// Phases 3a-3c and the foreign-pair rule each build for *one pair per
/// refinement round*.  This one does not, and the difference is measured
/// rather than assumed.
///
/// The cost this family carries is real: an `n`-ary `distinct` over array
/// terms contributes all `C(n,2)` pairs, so one round emits `C(n,2)` witness
/// lemmas — `C(n,2) · |D|` bit-vector equality atoms over an enumerated index
/// sort — and hands the whole set to a single re-solve.  Twelve pairwise
/// distinct arrays over `(Array (_ BitVec 3) (_ BitVec 1))` make 528 such
/// atoms in **one** refinement round, and the outer search then runs one
/// complete embedded `BvSolver::check` per bit-vector atom propagation:
/// 75,740 of them for 22 s of wall clock, with `combine_bv_with_euf` running
/// exactly *once*.  It is not a budget failure — the loop conflicts far below
/// `ARRAY_REFINEMENT_RESOLVE_CONFLICTS` and builds far below
/// `ARRAY_REFINEMENT_LEMMA_BUDGET` — because the cost is inside that round's
/// re-solve.
///
/// The obvious repair is the discipline the other four phases use, and it was
/// implemented and measured here: one pair per round, only pairs the candidate
/// model fails to separate.  It converges in `O(n)` rounds as intended — seven
/// pairwise distinct arrays at index width 3 need 11 rounds and 11 lemma
/// instances against `C(7,2) = 21` pairs — and it is **much worse**, because a
/// round is a whole re-solve and because a half-built family lets the search
/// satisfy each lemma through its `a = b` arm and then be refuted by the
/// `distinct` in the theory, once per pair per re-solve.  Measured on the same
/// ladder, release, 20 s cap:
///
/// ```text
///   index width 2      n=7      n=9      n=11     n=13     n=15
///   eager             2.7 ms   6.8 ms   17.5 ms  5.8 s    6.0 s
///   one pair/round    4.8 ms   > 20 s   > 20 s   > 20 s   8.0 s
///
///   index width 3      n=5      n=7      n=9      n=11
///   eager             2.0 ms   7.0 ms   28.6 ms  4.7 s
///   one pair/round    1.1 ms   7.5 ms   > 20 s   > 20 s
/// ```
///
/// Eight pairwise distinct arrays at index width 3 burned all 50,000 conflicts
/// of `ARRAY_REFINEMENT_RESOLVE_CONFLICTS` in 61.9 s under the lazy scheme,
/// against 28.6 ms eager at `n = 9`.  So the family stays eager, the residue
/// is reported open (`#P2b-38` strand (b)) rather than claimed, and what
/// bounds it is the deterministic embedded-check budget in `check_core`
/// instead.
pub(super) fn build_extensionality_and_congruence(
    manager: &mut TermManager,
    collected: &ArrayStructure,
    polarity: &PairPolarity,
    candidates: &mut Vec<TermId>,
    witnesses: &mut FxHashSet<TermId>,
) {
    for &(a, b) in &collected.eq_pairs {
        if witnesses.len() >= MAX_ARRAY_EXT_WITNESSES {
            return;
        }
        build_one_pair(manager, polarity, a, b, candidates, witnesses);
    }
}

/// The family of one pair; see [`build_extensionality_and_congruence`].
fn build_one_pair(
    manager: &mut TermManager,
    polarity: &PairPolarity,
    a: TermId,
    b: TermId,
    candidates: &mut Vec<TermId>,
    witnesses: &mut FxHashSet<TermId>,
) {
    let key = unordered(a, b);

    // A small, finite index sort is decided by enumeration rather than by
    // a Skolem witness (see [`ARRAY_INDEX_ENUMERATION_LIMIT`]): both
    // halves of the family below run at the domain's own elements, which
    // every pair shares and which the formula's index set already
    // contains.  No witness is minted for such a pair, so phases 3b and 3c
    // have nothing left to add for it either — `enumerated_pair_indices`
    // is what they consult to know that.
    let enumerated = array_domain(a, manager)
        .and_then(|domain| enumerable_index_values(manager, domain))
        .filter(|values| !values.is_empty());
    if let Some(values) = enumerated {
        // Extensionality over a finite domain:
        // a = b ∨ ⋁_{i ∈ D} select(a,i) != select(b,i).
        if !polarity.held_equal.contains(&key) && !polarity.separated_by_reads.contains(&key) {
            let mut arms: Vec<TermId> = Vec::with_capacity(values.len() + 1);
            arms.push(manager.mk_eq(a, b));
            for &index in &values {
                let read_a = manager.mk_select(a, index);
                let read_b = manager.mk_select(b, index);
                let reads_eq = manager.mk_eq(read_a, read_b);
                arms.push(manager.mk_not(reads_eq));
            }
            candidates.push(manager.mk_or(arms));
        }
        if !polarity.held_apart.contains(&key) {
            push_congruence_at(manager, a, b, &values, candidates);
        }
        return;
    }

    // Extensionality: a = b ∨ select(a,k) != select(b,k), with a fresh but
    // deterministic witness index per unordered pair.
    if !polarity.held_equal.contains(&key) && !polarity.separated_by_reads.contains(&key) {
        push_witness_lemma(manager, a, b, candidates);
        if let Some(domain) = array_domain(a, manager) {
            let witness = extensionality_witness(manager, a, b, domain);
            witnesses.insert(witness);
        }
    }
    if polarity.held_apart.contains(&key) {
        return;
    }

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
    //
    // Decision (7) coverage, stated honestly: unlike the store's own-index
    // read (1b), which *does* have a mutation witness — see
    // `register_store_own_index_reads` — no script is known whose verdict
    // changes when this congruence-at-the-witness instance is reverted.
    // That is an absence of a witness, not a proof of subsumption: the
    // shape it is written for,
    // `(= ((as const A) d1) ((as const A) d2))`, is decided by the witness
    // *lemma* alone whenever one index settles the pair, which every
    // reachable index domain does, and the indirect spelling of the same
    // shape is closed by `build_const_array_witness_cell` instead.
    // A future reader should treat this rule as uncovered rather than as
    // shown-redundant.
    if let Some(domain) = array_domain(a, manager) {
        let witness = extensionality_witness(manager, a, b, domain);
        indices.push(witness);
    }

    push_congruence_at(manager, a, b, &indices, candidates);
}

/// `a = b ⇒ select(a,j) = select(b,j)` for each `j` in `indices`.
///
/// The one place a select-congruence instance is built, shared by the eager
/// witness-index instance in [`build_extensionality_and_congruence`] and the
/// deferred chain-index family in
/// [`Solver::assert_chain_index_congruence`](super::Solver::assert_chain_index_congruence).
pub(super) fn push_congruence_at(
    manager: &mut TermManager,
    a: TermId,
    b: TermId,
    indices: &[TermId],
    candidates: &mut Vec<TermId>,
) {
    for &idx in indices {
        let read_a = manager.mk_select(a, idx);
        let read_b = manager.mk_select(b, idx);
        let reads_eq = manager.mk_eq(read_a, read_b);
        let eq_ab = manager.mk_eq(a, b);
        let cong = manager.mk_implies(eq_ab, reads_eq);
        candidates.push(cong);
    }
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
pub(super) fn build_const_array_reads(
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
    // A constant over an enumerable index sort reads back its default at every
    // element of that sort, not only where the script already spells a read.
    //
    // This is the enumerated counterpart of
    // [`build_const_array_witness_cell`]: the enumerated extensionality family
    // compares a pair at the domain's own elements, so the constant has to be
    // decided *there* for the comparison to say anything.  Emitting it here,
    // eagerly, is what keeps the `(as const)` shapes deciding in one refinement
    // round instead of waiting a round for the reads the family just minted to
    // be collected.
    //
    // # Only where an enumerated comparison actually happens (`#P2b-45`)
    //
    // The enumeration is emitted for a constant whose sort some *equality pair*
    // of the same domain is compared at, and for no other.  A `select` at a
    // ground index is not a free lemma: it is a new EUF term, it cascades a
    // read-over-write instance down every store chain of its sort, and it
    // becomes an entry the model builder has to publish and the `sat` gate has
    // to evaluate.  Emitted unconditionally, those reads turned
    // `round4_pass2_recheck_pins::the_store_own_index_read_is_what_decides_this_script`
    // — a `QF_AUFBV` goal with two array constants, a store chain and *no
    // array equality atom at all*, so nothing is ever compared by enumeration
    // — from `sat` in 21 ms into `unknown`: the axioms saturated one round
    // earlier against a candidate the added reads had changed, and the model
    // gate refused it.  With the rule gated on an actual comparison the same
    // goal answers `sat` in 27 ms, and every `(as const)` shape still decides
    // in the round it did.
    let enumerated_pair_domains: Vec<SortId> = collected
        .eq_pairs
        .iter()
        .filter_map(|&(a, _)| array_domain(a, manager))
        .collect();
    for &c in &collected.const_arrays {
        let (Some(default), Some(domain)) =
            (const_array_default(c, manager), array_domain(c, manager))
        else {
            continue;
        };
        if !enumerated_pair_domains.contains(&domain) {
            continue;
        }
        let Some(values) = enumerable_index_values(manager, domain) else {
            continue;
        };
        for index in values {
            let read = manager.mk_select(c, index);
            let read_is_default = manager.mk_eq(read, default);
            candidates.push(read_is_default);
        }
    }
}

/// Read-side lemmas for an **array-sorted `ite`** (`#P2b-41`, decision (14)).
///
/// For `t = (ite c x y)` of array sort and an index `i`:
///
/// ```text
///     c  =>  select(t, i) = select(x, i)
///   ! c  =>  select(t, i) = select(y, i)
/// ```
///
/// Both are theorems of the array theory for any `i`, and together they decide
/// every read through the `ite` once the search has picked a value for `c`.
///
/// # Why this exists beside the encoder's naming
///
/// `encode::needs_ite_elimination` names an array-sorted `ite` with a fresh
/// variable `v` plus the same two implications, and that is what makes a read
/// written *directly* through the `ite` decidable — `select(v, i)` is an
/// ordinary read of an ordinary array variable, and EUF merges `v` with the
/// branch the condition selects.
///
/// It does not reach the array theory, because this module walks
/// `Solver::assertions`, which holds the terms as the user wrote them.  A read
/// through a `store` over an `ite` is the case that matters:
///
/// ```smt2
/// (assert (= (select (store (ite p a0 a1) #b1 #b0) #b0) #b1))
/// (assert (= (select a0 #b0) #b0))
/// (assert (= (select a1 #b0) #b0))
/// ```
///
/// Read-over-write reduces the read to `select(ite(p, a0, a1), #b0)` — a read
/// of a term that, to this module, was neither a variable nor a `store` nor a
/// constant, and which no rule related to `a0` or `a1`.  The script is
/// unsatisfiable and answered `sat` with the encoder half alone.
///
/// # Which indices
///
/// Every index already read on the `ite` itself (which is where
/// read-over-write leaves the reduced read, one refinement round later), plus —
/// when the index sort is small enough to write out — every element of the
/// domain, so the shape decides in the round it is collected rather than the
/// round after.  Both sets are indices the formula already carries or the
/// enumerated extensionality family already mints, so the partition exchange
/// sees nothing new.
pub(super) fn build_array_ite_reads(
    manager: &mut TermManager,
    collected: &ArrayStructure,
    candidates: &mut Vec<TermId>,
) {
    for &(ite_term, cond, then_branch, else_branch) in &collected.array_ites {
        // The defining equalities themselves, as *array* equality atoms.  They
        // are what puts `{t, then}` and `{t, else}` into the equality-pair set,
        // so extensionality, select congruence and the store-chain families all
        // see the `ite` as an ordinary array term rather than as an opaque one
        // the read-side lemmas below merely sample.
        let not_cond = manager.mk_not(cond);
        let is_then = manager.mk_eq(ite_term, then_branch);
        let is_else = manager.mk_eq(ite_term, else_branch);
        candidates.push(manager.mk_implies(cond, is_then));
        candidates.push(manager.mk_implies(not_cond, is_else));
        let mut indices: Vec<TermId> = collected
            .read_indices
            .get(&ite_term)
            .cloned()
            .unwrap_or_default();
        if let Some(domain) = array_domain(ite_term, manager)
            && let Some(values) = enumerable_index_values(manager, domain)
        {
            for value in values {
                if !indices.contains(&value) {
                    indices.push(value);
                }
            }
        }
        for index in indices {
            let read = manager.mk_select(ite_term, index);
            let then_read = manager.mk_select(then_branch, index);
            let else_read = manager.mk_select(else_branch, index);
            let reads_then = manager.mk_eq(read, then_read);
            let reads_else = manager.mk_eq(read, else_read);
            let not_cond = manager.mk_not(cond);
            candidates.push(manager.mk_implies(cond, reads_then));
            candidates.push(manager.mk_implies(not_cond, reads_else));
        }
    }
}

/// The const-read axiom and select congruence for an array constant at the
/// extensionality witness index of **every** array pair of its sort, not only
/// of the pairs it is itself a member of (`#P2b-37`).
///
/// # The hole this closes
///
/// Every other family meets a constant only at an index some term already
/// spells, and each equality pair is instantiated at its *own* witness.  One
/// variable of indirection is enough to keep two constants apart:
///
/// ```smt2
/// (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))
/// (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))
/// (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))
/// ```
///
/// is unsatisfiable — `a` would have to be two different total functions — but
/// the pairs are `{a, c1}` with witness `k1` and `{a, c2}` with witness `k2`.
/// Congruence gives `select(a,k1) = select(c1,k1) = #b1` and
/// `select(a,k2) = select(c2,k2) = #b0`, at two different indices, and nothing
/// brings the two defaults into one instance.  With no `select` in the script
/// nothing else does either, so the formula answered `sat` on this tree, on
/// the 0.3.4 base and on crates.io 0.3.3 alike; 52 of 361 oracle-decided
/// scripts in a generated array-constant-indirection corpus were wrong the
/// same way.
///
/// Crossing the witnesses closes it: `select(c2,k1) = #b0` and
/// `a = c2 ⇒ select(a,k1) = select(c2,k1)` put both defaults at `k1`, and the
/// pair is refuted there.
///
/// # Why not the one-line `c1 = c2 ⇒ d1 = d2` instead
///
/// That lemma is equally valid and was tried first.  It introduces a *new
/// array-sorted equality atom* into the formula, which joins the BV↔EUF
/// partition-lemma exchange's candidate set; on a formula already carrying
/// two array constants and a store chain the exchange then reached its
/// 512-round budget (`#P2b-29`) and a script that answered `sat` in 28 ms
/// answered `unknown` in 86 ms instead — a completeness regression paid to
/// close a soundness hole.  This rule adds only `select` terms at indices the
/// witness family already minted, so the atom set the exchange sees does not
/// grow.
///
/// # Why this family is built one cell at a time
///
/// Read eagerly, as it was when it landed, the rule is *cubic*: every array
/// constant against every pair's witness index, and then against every pair
/// the constant itself belongs to.  Each cell interns a fresh `select` at a
/// fresh bit-vector witness index, and every such read cascades a
/// read-over-write instance down each link of both store chains and joins the
/// candidate set of the BV↔EUF partition-lemma exchange.  Six declarations and
/// three assertions were enough to make that unbounded: `rc3/slow/m5.smt2`
/// (two array constants, two array-sorted `ite`s, one `select`) answers `sat`
/// in 0.5 ms on the 0.3.4 base and produced **no answer in 400 s** with the
/// eager rule, with no budget stopping it, because the loop spends its time
/// interning and re-solving rather than conflicting.
///
/// So the family is now driven one *cell* per refinement round by
/// [`Solver::assert_const_array_witness_congruence`](super::Solver::assert_const_array_witness_congruence),
/// exactly as the off-chain family (phase 3c) and the foreign-pair rule (phase
/// 4) already are, and only for pairs the candidate assignment has not already
/// decided.  Nothing is lost: every instance here is a theorem of the array
/// theory, so a cell that is skipped this round is built in the round where
/// the assignment still needs it.
///
/// `witness` is the extensionality witness of the pair `(a, b)`; the caller
/// computes it once per pair rather than once per cell.
pub(super) fn build_const_array_witness_cell(
    manager: &mut TermManager,
    constant: TermId,
    witness: TermId,
    pair: (TermId, TermId),
    candidates: &mut Vec<TermId>,
) {
    let Some(default) = const_array_default(constant, manager) else {
        return;
    };
    // The constant reads back its default at this index too.
    let read = manager.mk_select(constant, witness);
    let read_is_default = manager.mk_eq(read, default);
    candidates.push(read_is_default);
    // …and the pair this constant belongs to is compared there.
    let (a, b) = pair;
    push_congruence_at(manager, a, b, &[witness], candidates);
}
