//! The seam between the *instantiation* paths and the *assertion* pre-passes.
//!
//! # What this module exists to prevent
//!
//! An assertion reaches the SAT core through [`Solver::assert`], which runs a
//! chain of pre-passes over it first — among them
//! [`Solver::eliminate_nonbool_ite`] — and leaves the term in
//! `Solver::assertions`, which is the root set
//! [`Solver::instantiate_array_axioms`] walks when it collects array
//! structure.
//!
//! A *ground instance* — an MBQI instantiation, a blind or finite-domain
//! instantiation, an e-matching lemma — does not.  It reaches the SAT core
//! through [`Solver::encode`] directly, so before this module existed it
//! received neither pre-pass and was never a root of the array-structure walk.
//! That is only invisible while every array term an instance mentions is
//! already ground somewhere in `Solver::assertions`; it is a soundness hole
//! the moment an array term is *first* ground after substitution, because
//! `array_axioms::ground_children` deliberately stops at `Forall`/`Exists`/
//! `Let`/`Match` — a `select` under a binder may mention the bound variable,
//! and a ground lemma over a bound variable is an instance of nothing.
//!
//! So `(forall ((i (_ BitVec 1))) (distinct #b0 (select ((as const …) #b0) i)))`
//! had no read-over-write lemma, no constant-array congruence and no `ite`
//! naming anywhere in the system: the read was a free value of the element
//! sort, and the search satisfied the instance by inventing one.  Four lines
//! answered `sat`, and `(get-value)` then printed `#b1` for a read of the
//! constant-`#b0` array — the solver contradicting itself inside one response.
//! Measured on 400 paired scripts (a quantified assertion beside its own
//! hand-expanded ground twin, which is the same formula written twice): 29
//! wrong `sat` and 130 falsifying models on the quantified side against 0/0/0
//! on the ground side.
//!
//! # The two halves
//!
//! [`Solver::prepare_ground_instance`] is the seam.  Every instantiation path
//! runs its instance through it before `encode`, and it does exactly two
//! things:
//!
//! 1. `eliminate_nonbool_ite`, so an array-sorted (or uninterpreted-sorted)
//!    `ite` that is first ground after substitution is hoisted to an
//!    EUF-visible constant exactly as it would be in an assertion.  Without
//!    this the `#P2b-41` family comes back under a binder.
//! 2. Registration of the instance as a root of the *next*
//!    `collect_array_structure` round, in [`Solver::ground_array_roots`], so
//!    the lazy refinement sees its `select` / `store` / `(as const …)` /
//!    array-`ite` subterms.  The registration is journalled
//!    (`TrailOp::GroundArrayRootAdded`), so a `pop` retracts it together
//!    with the clauses the instance contributed.
//!
//! A third thing was added by re-fix pass 8 (`#P2b-54`), and unlike the two
//! above it is about the instance's own *quantifiers* rather than its array
//! terms.  An instance is asserted as a hard unit clause, so a quantifier on
//! its asserted spine is an unconditional fact and must be registered, and a
//! quantifier anywhere else in it is a conditional occurrence and must be
//! guarded (`encode::quant_guard`).  Neither happened before, so
//! `(forall ((j …)) (forall ((i …)) φ))` — whose instance is
//! `(forall ((i …)) φ[j := c])` — reached `encode` with a free Boolean
//! literal, which is `#P2b-54`'s wrong `sat` one binder in.  Both are gated on
//! one `finite_expand::contains_quantifier` walk, so an instance with no
//! quantifier (the overwhelming majority) pays a single early-exit traversal
//! and nothing else.
//!
//! The other four pre-passes `Solver::assert` runs (`flatten_lookup_spines`,
//! `abstract_compound_bool_args`, `purify_numeric_uf_args`,
//! `collect_polarities`) are deliberately **not** run here.  They are
//! encoding-shape optimisations and non-array purifications whose cost is paid
//! per instance rather than per assertion, and none of them is implicated in
//! the defect above; adding them would change the cost model of every
//! quantified benchmark for no soundness gain.  The binder-sort witness
//! `register_asserted_quantifiers` mints at assert time is likewise **not**
//! minted here: the candidate pool is search state that
//! `MBQIIntegration::restore_search_state` rolls back, so a witness minted
//! mid-`check` would be re-minted on every `check-sat` and kept by none.
//!
//! # The half that is not here
//!
//! Registering a root is worth nothing if no refinement round follows.  The
//! lazy array refinement used to live inside `check_core`'s
//! `if !self.has_quantifiers` branch, so a quantified script never ran it at
//! all; it is now [`Solver::array_refinement_round`], called from both the
//! ground and the quantified candidate-model paths.  See that method.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortKind;
use rustc_hash::{FxHashMap, FxHashSet};

use super::Solver;
use super::array_axioms::ground_children;
use super::encode::finite_expand;
use super::trail::TrailOp;

impl Solver {
    /// Give a ground instance the two pre-passes an assertion receives, and
    /// return the term that should actually be encoded.
    ///
    /// Call this on **every** term an instantiation or lemma path is about to
    /// hand to [`Solver::encode`], before the `encode` call and before any
    /// other pass reads the term, so that the whole system sees the one term
    /// the SAT core's clauses describe.
    ///
    /// It is a cheap no-op on a term with no non-Bool `ite` and no array
    /// structure, which is the overwhelming majority of arithmetic and
    /// datatype lemmas — the walk is a single structural pass with a visited
    /// set and it allocates nothing when it finds nothing.
    ///
    /// # The one instantiation site that must not call this
    ///
    /// `array_axioms::instantiate_array_axioms` inserts its lemma term into
    /// `array_axiom_instances` and journals it *before* encoding it, so the
    /// dedup key is the pre-encode term by construction.  Rewriting the term
    /// between the insert and the `encode` would desynchronise the two: the
    /// next round would fail to recognise the lemma it just asserted and
    /// assert it again, round after round, until the refinement budget ran
    /// out.  Its instances are already roots of the next collection round (it
    /// walks `array_axiom_instances` itself), and they are built out of terms
    /// that have already been through this seam, so they need neither half.
    pub(super) fn prepare_ground_instance(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        let rewritten = self.eliminate_nonbool_ite(term, manager);
        // An instance is asserted as a hard unit clause, so every quantifier
        // on its own asserted spine is an unconditional fact and every other
        // quantifier position in it is exactly the conditional position
        // `encode::quant_guard` exists for.  Before this, a quantifier that
        // was nested inside the instantiated binder — `(forall ((j …))
        // (forall ((i …)) φ))`, whose instance is `(forall ((i …)) φ[j:=c])`
        // — reached `encode` with a free Boolean literal and no registration
        // at all, which is the same wrong `sat` as `#P2b-54` one binder in.
        //
        // Gated on one cheap early-exit walk, because the overwhelming
        // majority of instances carry no quantifier at all and this function
        // runs once per instance per MBQI round: with no quantifier in the
        // term, `quant_guard` returns `None` and the registration registers
        // nothing, so skipping both is exactly equivalent.  The `false` below
        // is what keeps `seed_binder_sort_witnesses` out of the search: a pool
        // entry minted mid-`check` is truncated away by
        // `MBQIIntegration::restore_search_state` on exit, so seeding here
        // would re-mint a fresh symbol on every `check-sat` of an incremental
        // script and never keep one.
        let rewritten = if finite_expand::contains_quantifier(rewritten, manager) {
            let (rewritten, obligations) = self.guard_conditional_quantifiers(rewritten, manager);
            self.register_asserted_quantifiers_with(rewritten, manager, false);
            for obligation in obligations {
                self.assert_quantifier_obligation(obligation, term, manager, 0);
            }
            rewritten
        } else {
            rewritten
        };
        self.register_ground_array_root(rewritten, manager);
        rewritten
    }

    /// The spelling an **assertion**'s encoded root must be registered under
    /// so that [`Solver::instantiate_array_axioms`] collects one array term
    /// per array: `term` with every non-Bool `ite` proxy
    /// [`Solver::eliminate_nonbool_ite`] minted put back.
    ///
    /// # Why a root is not simply the term that was encoded
    ///
    /// The collector's root set is two halves that disagree about spelling.
    /// `self.assertions` holds the **pre**-rewrite term, because a caller
    /// reading its assertions back must see what it asserted; the roots this
    /// module registers held the **post**-rewrite term, because the SAT
    /// core's clauses describe that one.  Where the rewrite chain replaced an
    /// array-sorted `(ite c a b)` with a proxy constant, the same array was
    /// then reachable under two names — and the array theory has no rule that
    /// says they are one object, so it treated them as two: two members of
    /// every pair set, two extensionality witnesses per pair, and a
    /// read-over-write cascade down each.  That is `#P2b-59`: on
    /// `rk9/min/q33_a01.smt2` (two width-8 arrays, `ite`-selected bases, one
    /// `store` each, **no quantifier**) — at index width 8, which is above
    /// the enumeration limit this rule is restricted to — the collector saw
    /// 19 array terms where 13 is the truth, and the refinement took 93
    /// rounds to saturate: no answer in 90 s, against `sat` in 0.34 ms on
    /// `c4b04b7` and on crates.io 0.3.3.
    ///
    /// # Why the `ite` spelling and not the proxy
    ///
    /// Because the `ite` is the spelling the *other* half of the root set
    /// already uses, and because it carries strictly more structure: the
    /// branches are visible, so `build_array_ite_reads` can relate a read
    /// through the `ite` to the same read on each branch.  Nothing is lost on
    /// the encoding side — every lemma this module's caller builds goes back
    /// through `Solver::encode`, which runs `eliminate_nonbool_ite` over it
    /// and so re-derives the proxy for the SAT core.
    ///
    /// # Why only an assertion, and not a ground instance
    ///
    /// Because only an assertion has the *same term* in the root set already.
    /// A ground instance is a root of its own; re-spelling it does not remove
    /// a duplicate, it only hands `build_array_ite_reads` every `ite` an
    /// instantiation grounds, in every MBQI round.  Measured, on the round's
    /// own width-7/8 corpus: registering instances under this spelling too
    /// took `rk6/corpus/qmbqi120/q0075` from `unknown` in 18.4 ms to no answer
    /// in 120 s, and `q0106` from 78.5 ms to 205.5 ms, while leaving
    /// `rk9/min/q33_a01.smt2` — the script the fix is for, which has no
    /// quantifier and so no instance — bit-identical at 66 rounds and 208
    /// lemma instances.  Narrowed on that measurement rather than on taste.
    pub(super) fn array_root_spelling(&self, term: TermId, manager: &mut TermManager) -> TermId {
        if self.ite_elim_aliases.is_empty() {
            return term;
        }
        // Only the proxies whose duplication is expensive: an array whose
        // index sort the extensionality family *enumerates*
        // (`ARRAY_INDEX_ENUMERATION_LIMIT`) mints no Skolem witness, so the
        // second spelling costs at most a constant factor of lemmas over the
        // one shared index set, and putting the `ite` back there instead
        // hands `build_array_ite_reads` a conditional read pair at every
        // element of the domain for every read the enumerated family creates.
        // That is a *loss*, and it was measured rather than guessed: over the
        // 300-pair width-1/2 corpus of `round4_pass5_recheck_pins`, one pair
        // (a `forall` over `(_ BitVec 2)` beside a ground `ite` over two
        // `store`s) went from `sat` in 8.5 ms to `sat` in 2,702 ms, with the
        // verdict unchanged, and the gate's 180 s ceiling turned into a
        // TIMEOUT. Above the enumeration limit — which is where `#P2b-59`
        // lives, at index width 8 — the duplicate is a whole second pair set
        // with its own Skolem witness per pair, and putting the `ite` back is
        // worth 93 refinement rounds.
        let filtered: FxHashMap<TermId, TermId> = self
            .ite_elim_aliases
            .iter()
            .filter(|(proxy, _)| {
                super::array_axioms::array_domain(**proxy, manager).is_some()
                    && !super::array_axioms::pair_is_enumerated(**proxy, manager)
            })
            .map(|(proxy, ite)| (*proxy, *ite))
            .collect();
        if filtered.is_empty() {
            return term;
        }
        manager.substitute(term, &filtered)
    }

    /// Record the term an *assertion* is actually encoded as, when the
    /// pre-pass chain rewrote it into something `self.assertions` does not
    /// contain.
    ///
    /// `Solver::assert` deliberately stores the **pre**-rewrite term in
    /// `self.assertions` (a caller reading assertions back must see what it
    /// asserted), and `instantiate_array_axioms` walks `self.assertions`.  For
    /// an assertion the chain leaves alone the two are the same term and this
    /// is a no-op.  For one it rewrites they are not, and the difference is
    /// not cosmetic: `skolemize_asserted_existentials` turns
    /// `(exists ((i …)) (= (select (store a …) i) …))` into a **ground** body
    /// over a fresh Skolem constant, and that body is exactly the array
    /// structure the collector must see.  Walking the stored term instead
    /// reaches the `Exists` node, stops there (`ground_children`), and collects
    /// nothing.  Measured on the final tree by removing exactly this call and
    /// re-running the 300-script paired corpus in
    /// `round4_pass5_recheck_pins`: **24 wrong `sat` of 300** (0 wrong
    /// `unsat`, 276 agree), together with the `store` and constant-array
    /// guards beside it — so this half of the seam is load-bearing on its own,
    /// not a belt beside the instantiation paths' braces.  (An earlier draft
    /// of this comment said "six of the 300"; that figure is withdrawn — it
    /// does not reproduce and the mutation that produces 24 is recorded as M3
    /// in `TODO.md` `#P2b-47`.)
    pub(super) fn register_encoded_assertion_root(
        &mut self,
        encoded: TermId,
        asserted: TermId,
        manager: &mut TermManager,
    ) {
        // `array_root_spelling` substitutes every above-the-enumeration-limit
        // proxy back to the `ite` it names, so the root the collector sees is
        // spelled the way `self.assertions` spells it.  That — and not the
        // early return below — is `#P2b-59`'s fix: registering the proxy
        // spelling beside the stored assertion is what gave the array theory
        // two names for one array.
        //
        // The `encoded == asserted` guard is the pre-existing cheap exit for a
        // chain that rewrote *nothing*, and it does **not** fire for an
        // assertion whose chain only eliminated `ite`s.
        // `Solver::eliminate_nonbool_ite` returns
        // `(and <rewritten> (=> c (= v t)) (=> (not c) (= v e)) …)` — the two
        // side conditions per eliminated `ite` are conjoined onto the term it
        // returns — so after re-spelling every proxy the root is
        // `(and <the original assertion> (=> c (= (ite c t e) t)) …)`, which
        // is not the stored assertion.  An earlier draft of this comment said
        // the equality was "the point of the fix"; that rationale is
        // **withdrawn** (adversarial recheck pass 10), it cannot hold, and the
        // outcome is unchanged either way because the duplicate spelling is
        // removed by the substitution above and not by this exit.
        let encoded = self.array_root_spelling(encoded, manager);
        if encoded == asserted {
            return;
        }
        self.register_ground_array_root(encoded, manager);
    }

    /// Record `term` as a root for the next `collect_array_structure` round,
    /// when it mentions any array structure at all.
    ///
    /// Also sets [`Solver::has_array_ops`], which is the flag `check_core`
    /// tests before it calls the refinement at all.  Setting it here is not
    /// redundant with `track_theory_vars`: that walk runs inside `encode` and
    /// would set the flag in the same round, but this makes the seam's
    /// precondition local to the seam rather than a property of another pass's
    /// traversal order.  The flag is snapshot-restored by `pop` (see
    /// `ContextState::has_array_ops`), so setting it mid-`check` is the
    /// established pattern — `assert_const_array_witness_congruence` already
    /// does it.
    fn register_ground_array_root(&mut self, term: TermId, manager: &TermManager) {
        if !mentions_array_structure(term, manager) {
            return;
        }
        if !self.ground_array_roots.insert(term) {
            return;
        }
        self.trail.push(TrailOp::GroundArrayRootAdded { term });
        self.has_array_ops = true;
    }
}

/// Whether `term` mentions a `select`, a `store` or any array-sorted sub-term
/// anywhere in its **ground** part.
///
/// Array-sortedness is the test that catches the constant array — `(as const
/// …)` is an ordinary `Apply` under a reserved function symbol, so there is no
/// `TermKind` to match on — and the array-sorted `ite` of `#P2b-41`, and a
/// plain array variable that only ever appears in an equality.  A `select`
/// itself is *not* array-sorted (its sort is the element sort), which is why
/// the two kinds are named explicitly beside the sort test.
///
/// The walk is [`ground_children`], the same one `collect_array_structure`
/// uses, so this predicate agrees with the collector by construction: it
/// answers `true` exactly when registering the term could give the collector
/// something it does not already have.  Descending into a binder would be
/// worse than useless — the collector stops there, so a `select` found under
/// one would register a root that contributes nothing.
fn mentions_array_structure(term: TermId, manager: &TermManager) -> bool {
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
        if matches!(data.kind, TermKind::Select(_, _) | TermKind::Store(_, _, _)) {
            return true;
        }
        if manager
            .sorts
            .get(data.sort)
            .is_some_and(|sort| matches!(sort.kind, SortKind::Array { .. }))
        {
            return true;
        }
        children.clear();
        ground_children(&data.kind, &mut children);
        stack.extend(children.iter().copied());
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The predicate is what decides whether a lemma becomes a collection
    /// root, so it has to agree with `collect_array_structure` on each of the
    /// shapes that reach the array rules — including the one that carries no
    /// `Select`/`Store` node at all.
    #[test]
    fn array_structure_is_recognised_in_every_shape_the_collector_reads() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);

        let a = tm.mk_var("a", array_sort);
        let i = tm.mk_int(1);
        let v = tm.mk_int(2);

        let select = tm.mk_select(a, i);
        assert!(mentions_array_structure(select, &tm), "select");

        let store = tm.mk_store(a, i, v);
        assert!(mentions_array_structure(store, &tm), "store");

        // An array-sorted variable inside an equality: no `Select`, no
        // `Store`, but `push_eq_pair` and the extensionality family read it.
        let b = tm.mk_var("b", array_sort);
        let eq = tm.mk_eq(a, b);
        assert!(mentions_array_structure(eq, &tm), "array equality");

        // Wrapped in Boolean structure, which is what an instantiation result
        // usually looks like.
        let not_eq = tm.mk_not(eq);
        assert!(mentions_array_structure(not_eq, &tm), "under `not`");

        // And under an arithmetic operator, which is the `#P2b-32` shape: a
        // read reached only through `+` was invisible to the old hand-written
        // child list.
        let read = tm.mk_select(a, i);
        let sum = tm.mk_add([read, i]);
        assert!(mentions_array_structure(sum, &tm), "under `+`");
    }

    /// The other half of the same claim: an arithmetic or Boolean lemma must
    /// not become an array root, or every quantified integer problem would
    /// start paying for a refinement round that has nothing to refine.
    #[test]
    fn an_array_free_lemma_is_not_a_root() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let one = tm.mk_int(1);
        let sum = tm.mk_add([x, one]);
        let eq = tm.mk_eq(sum, x);
        assert!(!mentions_array_structure(eq, &tm));
    }

    /// The binder exclusion, stated as a test rather than as a comment: a
    /// `select` that is still under a quantifier is not something the
    /// collector will reach, so registering its enclosing term as a root would
    /// add a root that contributes nothing.
    #[test]
    fn a_select_still_under_a_binder_is_not_ground_structure() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let a = tm.mk_var("a", array_sort);
        let bound = tm.mk_var("i", int_sort);
        let select = tm.mk_select(a, bound);
        let zero = tm.mk_int(0);
        let body = tm.mk_eq(select, zero);
        let forall = tm.mk_forall([("i", int_sort)], body);
        assert!(!mentions_array_structure(forall, &tm));
    }
}
