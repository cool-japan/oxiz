//! Turning a theory solver's reason terms into a CDCL(T) conflict clause.
//!
//! Split out of the parent module so `theory_manager.rs` stays under the
//! workspace 2000-line limit.  This is the whole of the translation from
//! "which facts did the theory blame" to "which literals make up the lemma",
//! including the honesty net that refuses to emit a clause whose justification
//! it cannot fully account for.

use super::TheoryManager;
use crate::prelude::*;
use oxiz_core::ast::TermId;
use oxiz_sat::{Lit, Var};
use smallvec::SmallVec;

impl TheoryManager<'_> {
    /// Convert a list of reason term IDs into a theory conflict clause.
    ///
    /// `analyze_theory_conflict` (in `oxiz-sat`) requires every literal of the
    /// conflict clause to be **false** under the current assignment.  A reason
    /// atom may have been assigned either polarity: a disequality (`Eq` assigned
    /// false), a `~(a != b)`, or a Bool application assigned false all store
    /// their term as a theory reason.  We therefore emit, for each reason atom,
    /// the literal opposite to its recorded assignment — `¬var` when the atom is
    /// currently true, `var` when it is currently false — so the literal is
    /// false as required.  Emitting `¬var` unconditionally (the previous
    /// behaviour) produced a *true* literal for negatively-assigned atoms,
    /// yielding an unsound lemma.
    ///
    /// A reason term that names no SAT variable must be *accounted for*, never
    /// dropped.  Omitting one does not weaken the clause, it falsifies it: the
    /// clause then asserts that the surviving literals alone are contradictory.
    /// That is how `(> (f a) (f b)) ∧ (a > b ∨ a = b)` was refuted — the
    /// congruence-derived `f(a) = f(b)` carried `f(a)` as its reason, `f(a)` has
    /// no SAT variable, and the clause reduced to the unit `¬(f(a) > f(b))`,
    /// a level-0 conflict on a satisfiable formula.  Three cases exist, and each
    /// is now distinguished explicitly:
    ///
    /// 1. the term names an atom the SAT core still has assigned — emit its
    ///    currently-false literal;
    /// 2. the term tags an equality the theory layer *derived* and propagated
    ///    between theories — expand it into the EUF explanation recorded in
    ///    [`DerivedReasons`] and resolve those terms in turn;
    /// 3. the term is a registered theory tautology (`10 ≠ 20`, `true ≠ false`,
    ///    two term ids for the same constant) — it depends on no literal and
    ///    contributes nothing.
    ///
    /// Anything else is a bug in whichever caller injected the reason.  It trips
    /// a `debug_assert!` so the next occurrence is loud, and falls back to
    /// [`Self::full_assignment_conflict_clause`] — the negation of the whole
    /// current assignment, which is a valid (if maximally imprecise) lemma
    /// whenever the theories genuinely refute that assignment.
    ///
    /// # `None` is not "no conflict", it is "this conflict is unusable"
    ///
    /// The fallback itself can fail: with nothing assigned there is no literal
    /// to blame, and the negation of the empty assignment is the **empty
    /// clause** — an unconditional top-level refutation.  Emitting that would
    /// turn a lost-justification bug into a silent false `Unsat` in release
    /// builds, where the `debug_assert!`s above are compiled out.  So the
    /// fallback is fallible too, and `None` travels all the way out: see
    /// [`super::TheoryManager::conflict_from_terms`], which converts it into
    /// "abort this conflict and make the solver answer `Unknown`".
    pub(super) fn terms_to_conflict_clause(&self, terms: &[TermId]) -> Option<SmallVec<[Lit; 8]>> {
        let mut conflict: SmallVec<[Lit; 8]> = SmallVec::new();
        let mut emitted: FxHashSet<Var> = FxHashSet::default();
        let mut expanded: FxHashSet<TermId> = FxHashSet::default();
        let mut pending: Vec<TermId> = terms.to_vec();

        while let Some(term) = pending.pop() {
            // (1) The reason names a Boolean atom.
            //
            //     The same liveness requirement as case (2) applies, for a
            //     stronger reason: `analyze_theory_conflict` needs every literal
            //     of the clause to be *false*, and an atom the SAT core has not
            //     assigned has no false literal at all — `false_literal_of`
            //     would fall back to `¬var`, which is merely undefined.  A
            //     theory assertion whose reason atom is unassigned is one that
            //     outlived the trail entry that made it: a scope the manager
            //     could not reach (see `Solver::rebase_theory_state`) or an
            //     explanation read out of a tableau that has since been popped.
            //     Emitting it silently is how the task-#26 leak produced a
            //     false `unsat` without tripping a single net.
            if let Some(&var) = self.term_to_var.get(&term) {
                if !self.reason_literal_is_live(term) {
                    debug_assert!(
                        false,
                        "theory conflict reason {term:?} names SAT variable {var:?}, \
                         which the core has not assigned: the assertion it justifies \
                         outlived the trail entry that made it"
                    );
                    return self.full_assignment_conflict_clause();
                }
                if emitted.insert(var) {
                    conflict.push(self.false_literal_of(var));
                }
                continue;
            }
            // (2) The reason tags a derived, propagated equality: replace it by
            //     the literals that justify it.  `expanded` bounds the walk, so
            //     a justification cycle cannot loop.
            //
            //     An explanation is only usable while every literal it names is
            //     still asserted.  The theory solvers outlive the manager that
            //     drove them (a fresh one is built per MBQI round), so an
            //     equality can survive in the tableau into a round whose trail
            //     no longer contains the literals that derived it.  Expanding
            //     such an explanation would put a literal that is *not* false
            //     into the clause, breaking the invariant
            //     `analyze_theory_conflict` relies on — the same class of bug
            //     as a missing explanation, and reported the same way.
            if let Some(mut justification) = self.derived_reasons.literals(term) {
                if let Some(stale) = justification.find(|&lit| !self.reason_literal_is_live(lit)) {
                    debug_assert!(
                        false,
                        "derived reason {term:?} is explained by {stale:?}, which is \
                         no longer asserted: the explanation outlived the assignment \
                         it was read out of"
                    );
                    return self.full_assignment_conflict_clause();
                }
                if expanded.insert(term)
                    && let Some(justification) = self.derived_reasons.literals(term)
                {
                    pending.extend(justification);
                }
                continue;
            }
            // (3) The reason is a theory tautology and depends on no literal.
            if self.tautological_reasons.contains(&term) {
                continue;
            }
            debug_assert!(
                false,
                "theory conflict reason term {term:?} has no SAT variable, no \
                 recorded explanation and is not a registered tautology: the \
                 conflict clause would silently omit part of its justification"
            );
            return self.full_assignment_conflict_clause();
        }

        // An empty clause is an unconditional top-level refutation.  Reaching
        // one from a non-empty reason set would mean the theory tautologies
        // alone are inconsistent, which they are not — so treat it as the same
        // class of bug and fall back to the conservative lemma.
        if conflict.is_empty() && !terms.is_empty() {
            debug_assert!(
                false,
                "theory conflict over {terms:?} produced an empty clause: that \
                 claims an unconditional refutation from tautologies alone"
            );
            return self.full_assignment_conflict_clause();
        }
        // Belt and braces for the remaining shape, `terms.is_empty()`: a theory
        // that reports a refutation blaming *nothing* has not justified it
        // either, and the clause built from it is again the empty clause.  It is
        // rejected here rather than assumed unreachable, because the cost of
        // being wrong is a refutation of every input.
        if conflict.is_empty() {
            debug_assert!(
                false,
                "theory reported a conflict with an empty reason set: the clause \
                 would be the empty clause, i.e. an unconditional refutation"
            );
            return None;
        }

        Some(conflict)
    }

    /// Is `term` a reason literal the SAT core currently has assigned?
    ///
    /// The single liveness authority for [`Self::terms_to_conflict_clause`]:
    /// both the atom case and the derived-equality expansion consult it, so a
    /// stale justification is reported the same way wherever it enters.
    ///
    /// A term that names no SAT variable at all is not a literal and cannot go
    /// stale — it resolves through the tautology / derived-equality cases
    /// instead, so it counts as live here.  `assigned_level` (not the
    /// generation-stamped polarity table) is the authority: it is the map
    /// pruned on backtrack, so an entry means the assignment still holds.
    /// The polarity table is deliberately *not* pruned and therefore
    /// outlives the assignment it describes; reading it would call a
    /// retracted literal live.
    fn reason_literal_is_live(&self, term: TermId) -> bool {
        match self.term_to_var.get(&term) {
            Some(var) => self.assigned_level.contains_key(var),
            None => true,
        }
    }

    /// The literal of `var` that is **false** under the current assignment.
    ///
    /// `analyze_theory_conflict` (in `oxiz-sat`) requires every literal of a
    /// theory conflict clause to be false, and a reason atom may hold either
    /// polarity: a disequality (`Eq` assigned false), a `~(a != b)`, or a Bool
    /// application assigned false all store their term as a theory reason.
    #[inline]
    fn false_literal_of(&self, var: Var) -> Lit {
        match self.assigned_pol_of(var) {
            // Atom currently true  → its false literal is ¬var.
            Some(true) => Lit::neg(var),
            // Atom currently false → its false literal is var.
            Some(false) => Lit::pos(var),
            // Polarity unknown.  Unreachable from `terms_to_conflict_clause`,
            // which now rejects any reason atom missing from `assigned_level`,
            // and `on_assignment` writes both tables together — so an entry
            // in `assigned_level` implies one in the polarity table.  Kept as
            // a total fallback for `full_assignment_conflict_clause`,
            // matching legacy behaviour rather than introducing a panic path.
            None => Lit::neg(var),
        }
    }

    /// The negation of every literal the SAT core currently has assigned, or
    /// `None` when there is no such literal.
    ///
    /// Every literal is false by construction, so the clause satisfies
    /// `analyze_theory_conflict`'s precondition, and it is entailed whenever the
    /// theories really do refute the current assignment.  It is the weakest
    /// possible lemma — the search makes minimal progress from it — which is
    /// exactly why it is reserved for the "cannot justify a reason" path that
    /// `debug_assert!` declares unreachable.  Ordered by variable index so the
    /// clause, and hence the whole search, stays deterministic.
    ///
    /// # Why it is fallible
    ///
    /// `assigned_level` empty means the SAT core has assigned nothing, and the
    /// negation of an empty assignment is the **empty clause**.  Handing that to
    /// `analyze_theory_conflict` is not a weak lemma, it is the strongest
    /// possible claim: the formula is refuted outright, in every context, no
    /// matter what was asserted.  Yet this function is only ever reached from a
    /// path that has just discovered it *cannot account for the justification*
    /// of a conflict — so the one thing that is certain there is that no
    /// refutation has been established.  In a debug build the `debug_assert!`s
    /// above fire first; in a release build they are gone, and returning the
    /// empty clause would convert a lost-justification bug into a silent false
    /// `Unsat`.
    ///
    /// `None` is therefore the honest answer: no literal can be blamed, so no
    /// lemma exists, and the caller must abort the conflict rather than refute.
    fn full_assignment_conflict_clause(&self) -> Option<SmallVec<[Lit; 8]>> {
        if self.assigned_level.is_empty() {
            return None;
        }
        let mut vars: Vec<Var> = self.assigned_level.keys().copied().collect();
        vars.sort_unstable_by_key(|v| v.index());
        Some(vars.into_iter().map(|v| self.false_literal_of(v)).collect())
    }
}

/// Pins for the honesty net that keeps an **empty** conflict clause
/// unrepresentable.
///
/// An empty clause is not a weak lemma, it is the strongest possible claim:
/// "refuted, unconditionally, whatever was asserted".  Every code path that can
/// produce one here is a path that has just failed to justify a conflict, so
/// producing it would convert a lost-justification bug into a silent false
/// `Unsat` — in release builds, where the `debug_assert!`s are gone.  These
/// tests drive the fallback directly, with no assignment on the table.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::theory_manager::DerivedReasons;
    use crate::solver::types::{Constraint, ParsedArithConstraint, Statistics, TheoryMode};
    use oxiz_core::ast::TermManager;
    use oxiz_sat::Var;
    use oxiz_theories::arithmetic::ArithSolver;
    use oxiz_theories::bv::BvSolver;
    use oxiz_theories::euf::EufSolver;

    /// Every borrow a [`TheoryManager`] needs, owned by the test so the manager
    /// can be built inline.
    struct Fixture {
        terms: TermManager,
        euf: EufSolver,
        arith: ArithSolver,
        bv: BvSolver,
        bv_terms: FxHashSet<TermId>,
        var_to_constraint: FxHashMap<Var, Constraint>,
        var_to_parsed_arith: FxHashMap<Var, ParsedArithConstraint>,
        term_to_var: FxHashMap<TermId, Var>,
        var_to_term: Vec<TermId>,
        derived_reasons: DerivedReasons,
        statistics: Statistics,
        quantifier_uf_funcs: FxHashSet<oxiz_core::interner::Spur>,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                terms: TermManager::new(),
                euf: EufSolver::new(),
                arith: ArithSolver::lra(),
                bv: BvSolver::new(),
                bv_terms: FxHashSet::default(),
                var_to_constraint: FxHashMap::default(),
                var_to_parsed_arith: FxHashMap::default(),
                term_to_var: FxHashMap::default(),
                var_to_term: Vec::new(),
                derived_reasons: DerivedReasons::default(),
                statistics: Statistics::new(),
                quantifier_uf_funcs: FxHashSet::default(),
            }
        }

        fn manager(&mut self) -> TheoryManager<'_> {
            TheoryManager::new(
                &self.terms,
                &mut self.euf,
                &mut self.arith,
                &mut self.bv,
                &self.bv_terms,
                &self.var_to_constraint,
                &self.var_to_parsed_arith,
                &self.term_to_var,
                &self.var_to_term,
                &mut self.derived_reasons,
                TheoryMode::Eager,
                &mut self.statistics,
                0,
                0,
                false,
                false,
                &self.quantifier_uf_funcs,
                None,
            )
        }
    }

    /// With nothing assigned there is no literal to blame, so the "negate the
    /// whole assignment" fallback has no clause to offer.  It must say so
    /// (`None`) rather than hand back the empty clause.
    #[test]
    fn full_assignment_fallback_is_none_when_nothing_is_assigned() {
        let mut fixture = Fixture::new();
        let manager = fixture.manager();

        assert!(
            manager.assigned_level.is_empty(),
            "fixture precondition: a fresh manager has seen no assignment"
        );
        assert_eq!(
            manager.full_assignment_conflict_clause(),
            None,
            "the negation of an empty assignment is the empty clause, which \
             claims an unconditional refutation; the fallback must decline"
        );
    }

    /// Control for the test above: with an assignment on the table the fallback
    /// is a real (if maximally imprecise) lemma, and every literal in it is
    /// false under that assignment.
    #[test]
    fn full_assignment_fallback_negates_the_assignment_when_there_is_one() {
        let mut fixture = Fixture::new();
        let mut manager = fixture.manager();

        let true_var = Var::new(0);
        let false_var = Var::new(1);
        manager.assigned_level.insert(true_var, 0);
        manager.set_assigned_polarity(true_var, true);
        manager.assigned_level.insert(false_var, 0);
        manager.set_assigned_polarity(false_var, false);

        let clause = manager
            .full_assignment_conflict_clause()
            .expect("a non-empty assignment always yields a clause");
        // Ordered by variable index, and each literal is the *false* one.
        assert_eq!(
            clause.as_slice(),
            &[Lit::neg(true_var), Lit::pos(false_var)]
        );
    }

    /// A reason term that names no SAT variable, no recorded explanation and no
    /// registered tautology cannot be justified.  In a debug build the net fires
    /// before anything can be emitted — that is what the `debug_assert!`s are
    /// for, and this pins that they are still armed.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "has no SAT variable")]
    fn an_unjustifiable_reason_trips_the_debug_net() {
        let mut fixture = Fixture::new();
        let orphan = fixture
            .terms
            .mk_var("orphan", fixture.terms.sorts.bool_sort);
        let mut manager = fixture.manager();
        let _ = manager.conflict_from_terms(&[orphan]);
    }

    /// The release-build half of the test above, and the one that matters for
    /// soundness: with the `debug_assert!`s compiled out, an unjustifiable
    /// conflict must **abort** — flagging `unjustified_conflict` so the owning
    /// solver answers `Unknown` — and must never reach the SAT core as a clause,
    /// least of all as the empty one.
    #[cfg(not(debug_assertions))]
    #[test]
    fn an_unjustifiable_conflict_aborts_instead_of_refuting() {
        use oxiz_sat::TheoryCheckResult;

        let mut fixture = Fixture::new();
        let orphan = fixture
            .terms
            .mk_var("orphan", fixture.terms.sorts.bool_sort);
        let mut manager = fixture.manager();

        let result = manager.conflict_from_terms(&[orphan]);
        assert!(
            matches!(result, TheoryCheckResult::Sat),
            "an unjustifiable conflict must be dropped, not turned into a lemma"
        );
        assert!(
            manager.unjustified_conflict(),
            "dropping the conflict must be recorded, or the solver would trust \
             the resulting `Sat`"
        );
    }

    /// A theory that reports a refutation blaming *nothing* is the same defect
    /// arriving from the other side: the clause built from an empty reason set
    /// is the empty clause.  It must be declined, not emitted.
    #[cfg(not(debug_assertions))]
    #[test]
    fn an_empty_reason_set_is_declined() {
        let mut fixture = Fixture::new();
        let manager = fixture.manager();
        assert_eq!(manager.terms_to_conflict_clause(&[]), None);
    }

    /// Debug-build counterpart of the test above.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "empty reason set")]
    fn an_empty_reason_set_trips_the_debug_net() {
        let mut fixture = Fixture::new();
        let manager = fixture.manager();
        let _ = manager.terms_to_conflict_clause(&[]);
    }
}
