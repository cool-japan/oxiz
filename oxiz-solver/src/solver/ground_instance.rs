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
//! The other four pre-passes `Solver::assert` runs (`flatten_lookup_spines`,
//! `abstract_compound_bool_args`, `purify_numeric_uf_args`,
//! `collect_polarities`) are deliberately **not** run here.  They are
//! encoding-shape optimisations and non-array purifications whose cost is paid
//! per instance rather than per assertion, and none of them is implicated in
//! the defect above; adding them would change the cost model of every
//! quantified benchmark for no soundness gain.
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
use rustc_hash::FxHashSet;

use super::Solver;
use super::array_axioms::ground_children;
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
        self.register_ground_array_root(rewritten, manager);
        rewritten
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
    /// nothing — six of the 300 paired scripts in
    /// `round4_pass5_recheck_pins` were still wrong `sat` for this reason
    /// after the instantiation paths had been closed, every one of them an
    /// asserted `exists`.
    pub(super) fn register_encoded_assertion_root(
        &mut self,
        encoded: TermId,
        asserted: TermId,
        manager: &TermManager,
    ) {
        if encoded == asserted {
            return;
        }
        self.register_ground_array_root(encoded, manager);
    }

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
