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

        if collected.selects.is_empty() && collected.eq_pairs.is_empty() {
            return false;
        }

        // ---- Phase 2: build candidate ground axiom instances ------------
        let mut candidates: Vec<TermId> = Vec::new();
        build_read_over_write(manager, &collected, &mut candidates);
        build_extensionality_and_congruence(manager, &collected, &mut candidates);

        // ---- Phase 3: filter (dedup + model) and assert -----------------
        // Only instances the candidate model does not already *definitely*
        // satisfy are added.  A `None` evaluation (opaque/undetermined) is
        // treated as unsatisfied so completeness never depends on the model
        // being able to evaluate a select — worst case this degenerates to
        // eager instantiation, which is still sound and complete.
        let mut to_add: Vec<TermId> = Vec::new();
        {
            let model = self.model.as_ref();
            for &inst in &candidates {
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

        added
    }
}

/// Array terms and (dis)equalities gathered from a term-graph walk.
#[derive(Default)]
struct ArrayStructure {
    /// `(select_term, array_operand, index)` for every `select` encountered.
    selects: Vec<(TermId, TermId, TermId)>,
    /// Unordered array-sorted equality atoms `(a, b)` (`a != b` syntactically).
    eq_pairs: Vec<(TermId, TermId)>,
    /// `array_variable -> store_term` for every asserted `var = store(...)`.
    aliases: FxHashMap<TermId, TermId>,
    /// Distinct indices read on each array operand (for select congruence).
    read_indices: FxHashMap<TermId, Vec<TermId>>,
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
                stack.push(*value);
                stack.push(*index);
                stack.push(*base);
            }
            TermKind::Eq(lhs, rhs) => {
                // Record an array-sorted equality atom (either polarity: the
                // extensionality / congruence lemmas are valid regardless).
                if lhs != rhs && is_array_sorted(*lhs, manager) && is_array_sorted(*rhs, manager) {
                    out.eq_pairs.push((*lhs, *rhs));
                }
                // Record a `var = store(...)` alias for alias-aware
                // read-over-write.
                record_alias(*lhs, *rhs, manager, &mut out.aliases);
                record_alias(*rhs, *lhs, manager, &mut out.aliases);
                stack.push(*rhs);
                stack.push(*lhs);
            }
            _ => {
                stack.extend(term_children(&data.kind).into_iter().rev());
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
        if let Some(domain) = array_domain(a, manager) {
            let witness = extensionality_witness(manager, a, b, domain);
            let read_a = manager.mk_select(a, witness);
            let read_b = manager.mk_select(b, witness);
            let reads_eq = manager.mk_eq(read_a, read_b);
            let reads_diff = manager.mk_not(reads_eq);
            let eq_ab = manager.mk_eq(a, b);
            let ext = manager.mk_or([eq_ab, reads_diff]);
            candidates.push(ext);
        }

        // Select congruence: a = b ⇒ select(a,j) = select(b,j) for every index
        // read on either side.
        let mut indices: Vec<TermId> = Vec::new();
        if let Some(idxs) = collected.read_indices.get(&a) {
            for &idx in idxs {
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
        }
        if let Some(idxs) = collected.read_indices.get(&b) {
            for &idx in idxs {
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
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

/// Immediate sub-terms of a term kind that the structural walk should descend
/// into for the operators not handled explicitly by `collect_array_structure`.
/// Only the connectives / shapes that can legitimately contain array sub-terms
/// need to be exhaustive; anything else contributes no children.
fn term_children(kind: &TermKind) -> Vec<TermId> {
    match kind {
        TermKind::Not(a) | TermKind::Neg(a) => vec![*a],
        TermKind::And(args) | TermKind::Or(args) => args.to_vec(),
        TermKind::Distinct(args) => args.to_vec(),
        TermKind::Implies(a, b) | TermKind::Xor(a, b) => vec![*a, *b],
        TermKind::Ite(c, t, e) => vec![*c, *t, *e],
        TermKind::Apply { args, .. } => args.to_vec(),
        _ => Vec::new(),
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
