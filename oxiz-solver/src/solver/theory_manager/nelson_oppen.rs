//! Bidirectional Nelson-Oppen equality/disequality exchange between the
//! arithmetic and EUF theories.
//!
//! Split out of the parent module so `theory_manager.rs` stays under the
//! workspace 2000-line limit -- see `conflict_clause.rs` for the identical
//! precedent (a self-contained concern lifted into its own child module,
//! `impl TheoryManager<'_>` reopened here rather than in the parent file).
//!
//! [`TheoryManager::propagate_euf_equalities_to_arith`] and
//! [`TheoryManager::model_based_combination`] (both still in the parent
//! module) implement one direction of theory combination: an equality
//! congruence closure *derives* is asserted into the tableau. This module is
//! the other direction -- an equality *arithmetic* entails between two
//! shared terms, propagated into EUF -- plus the disequality analogue, and
//! the bounded candidate-pair search (the "care graph") that keeps both
//! affordable.

use super::TheoryManager;
use crate::prelude::*;
use num_rational::Rational64;
use oxiz_core::ast::TermId;
use oxiz_sat::TheoryCheckResult;
use smallvec::SmallVec;

/// Normalize an unordered term pair into a canonical `(min, max)` form, so
/// `(a, b)` and `(b, a)` collapse to the same care-graph candidate.
#[inline]
fn order_pair(a: TermId, b: TermId) -> (TermId, TermId) {
    if a <= b { (a, b) } else { (b, a) }
}

impl TheoryManager<'_> {
    /// Bidirectional Nelson-Oppen equality/disequality exchange between the
    /// arithmetic and EUF theories, driven to a fixpoint.
    ///
    /// [`Self::propagate_euf_equalities_to_arith`] and
    /// [`Self::model_based_combination`] already implement one direction of
    /// theory combination: an equality congruence closure *derives* is
    /// asserted into the tableau, and two shared terms EUF's current model
    /// puts in one class while arithmetic assigns different values are
    /// resolved the same way. Call that EUF -> arithmetic.
    ///
    /// The other direction was missing entirely: an equality *arithmetic*
    /// entails between two shared terms (e.g. `y = x + 1` together with `x =
    /// 2` entails `y = 3`) is never told to EUF, so a congruence that
    /// entailment should trigger (`f(y) = f(3)`) is silently missed. This is
    /// the textbook non-convex-combination false-`sat`: arithmetic alone has
    /// no notion of congruence, and EUF alone has no notion of "this value is
    /// forced by the other theory's bounds" unless someone tells it.
    ///
    /// Each round: probe a bounded set of candidate term pairs (the *care
    /// graph*, see [`Self::care_graph_candidates`]) for an arithmetic-entailed
    /// equality or disequality, merge/diseq-assert whichever hold into EUF,
    /// then hand any *new* EUF-side agreement back to arithmetic via
    /// [`Self::propagate_euf_equalities_to_arith`] (the pre-existing
    /// direction). Alternates until neither direction produces new
    /// information, then falls back to [`Self::model_based_combination`] for
    /// any residual class/value disagreement the exchange did not resolve.
    ///
    /// Every merge or disequality this induces is a *deduction*:
    /// [`oxiz_theories::arithmetic::ArithSolver::entailed_equal_reason`] /
    /// `entailed_disequal_reason` re-derive the Farkas certificate that forces
    /// it, recorded under the pair's own tag in [`Self::derived_reasons`], so
    /// a conflict that cites the resulting EUF fact expands back to the
    /// literals that actually justify it — it can never manufacture a false
    /// `unsat`.
    pub(super) fn nelson_oppen_combine(&mut self) -> TheoryCheckResult {
        use oxiz_theories::Theory;
        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;

        // A purely-arithmetic problem (no uninterpreted-function structure at
        // all) has no congruence for this exchange to ever trigger, so the
        // whole mechanism below is pure overhead there — skip straight to
        // the pre-existing combination. This is what keeps the exchange from
        // regressing UF-free QF_LIA/QF_LRA benchmarks.
        //
        // Under quantifiers, `care_graph_candidates` below already restricts
        // itself to the one candidate source scoped by `quantifier_uf_funcs`
        // (see that method's doc comment for why that source alone is safe to
        // keep running), so there is no separate `has_quantifiers` check here:
        // a formula in which every function occurs under some binder simply
        // produces an empty candidate set below and this loop falls straight
        // through to `model_based_combination` on its own, at the cost of one
        // cheap pass instead of a branch.
        if !self.euf.has_app_nodes() {
            return self.model_based_combination();
        }

        const MAX_COMBINE_ROUNDS: usize = 8;
        for _ in 0..MAX_COMBINE_ROUNDS {
            let candidates = self.care_graph_candidates();
            if candidates.is_empty() {
                break;
            }

            // Snapshot every candidate term's model value once for this
            // round. `entailed_equal_reason` pushes/pops a scratch simplex
            // scope per probe; re-reading `arith.value` mid-loop would tie
            // this pre-filter's soundness to that scope restoring the
            // assignment array exactly, which it should not have to assume.
            let mut model_value: FxHashMap<TermId, Rational64> = FxHashMap::default();
            for &(a, b) in &candidates {
                for t in [a, b] {
                    if !model_value.contains_key(&t)
                        && let Some(v) = self.arith.value(t)
                    {
                        model_value.insert(t, v);
                    }
                }
            }

            // ---- arithmetic -> EUF, equalities ----
            let mut changed = false;
            for &(x, y) in &candidates {
                let lx = self.euf.intern(x);
                let ly = self.euf.intern(y);
                if self.euf.are_equal(lx, ly) {
                    continue;
                }
                // Cheap sound pre-filter: a pair the *current* model already
                // disagrees on cannot be arithmetically entailed equal (the
                // model is itself a counter-witness), so the probe below is
                // guaranteed to return `None`. Skipping it is what keeps the
                // care graph affordable on a large interface.
                if let (Some(&vx), Some(&vy)) = (model_value.get(&x), model_value.get(&y))
                    && vx != vy
                {
                    continue;
                }
                let Some(reason) = self.arith.entailed_equal_reason(x, y) else {
                    continue;
                };
                self.derived_reasons.record(x, reason);
                if self.euf.merge(lx, ly, x).is_ok() {
                    changed = true;
                }
            }
            if let Some(conflict_terms) = self.euf.check_conflicts() {
                return self.report_theory_conflict(conflict_terms);
            }

            // ---- arithmetic -> EUF, disequalities (cvc5's
            //      `watchedVariableCannotBeZero` analogue) ----
            //
            // A pair EUF now holds equal — from the merges just above, or
            // from congruence closure firing on them independently — that
            // arithmetic's bounds alone already rule out is an immediate
            // cross-theory conflict.
            for &(x, y) in &candidates {
                let lx = self.euf.intern(x);
                let ly = self.euf.intern(y);
                if !self.euf.are_equal(lx, ly) {
                    continue;
                }
                let Some(reason) = self.arith.entailed_disequal_reason(x, y) else {
                    continue;
                };
                self.derived_reasons.record(x, reason);
                self.euf.assert_diseq(lx, ly, x);
                changed = true;
            }
            if let Some(conflict_terms) = self.euf.check_conflicts() {
                return self.report_theory_conflict(conflict_terms);
            }

            if !changed {
                break;
            }

            // ---- EUF -> arithmetic ----
            //
            // The merges above may have put two arithmetic-shared terms in
            // one class with different tableau values; assert the equality
            // and let arithmetic refute it if it must.
            let euf_result = self.propagate_euf_equalities_to_arith();
            if let TheoryCheckResult::Conflict(_) = euf_result {
                self.statistics.theory_conflicts += 1;
                self.statistics.conflicts += 1;
                return euf_result;
            }
            match self.arith.check() {
                Ok(TheoryCheckResultEnum::Sat) | Ok(TheoryCheckResultEnum::Propagate(_)) => {}
                Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) => {
                    return self.report_theory_conflict(conflict_terms);
                }
                Ok(TheoryCheckResultEnum::Unknown) | Err(_) => {
                    self.resource_exhausted = true;
                    return TheoryCheckResult::Sat;
                }
            }
        }

        // Fixpoint reached (or the round bound was hit): fall back to the
        // model-based check for any residual EUF-class / arithmetic-value
        // disagreement the entailed-(dis)equality exchange above did not
        // resolve.
        self.model_based_combination()
    }

    /// Turn a theory-refuted term set into a [`TheoryCheckResult`], bumping
    /// conflict statistics and honouring the conflict-count resource limit
    /// the same way every other conflict site in this file does.
    pub(super) fn report_theory_conflict(
        &mut self,
        conflict_terms: Vec<TermId>,
    ) -> TheoryCheckResult {
        self.statistics.theory_conflicts += 1;
        self.statistics.conflicts += 1;
        if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts {
            self.resource_exhausted = true;
            return TheoryCheckResult::Sat;
        }
        self.conflict_from_terms(&conflict_terms)
    }

    /// Build the Nelson-Oppen *care graph*: the bounded set of shared-term
    /// pairs worth probing this round for an arithmetic entailment.
    ///
    /// Probing every pair of shared terms is O(n²), and on the interface
    /// sizes real QF_UFLIA/QF_UFIDL instances produce that is not
    /// affordable. Only three shapes of pair are ever worth the cost:
    ///
    /// * **difference-constraint pairs** — a `x - y <op> c` atom names
    ///   exactly the pair a single Farkas probe can decide either way;
    /// * **live EUF disequality pairs** — the only pairs whose forced
    ///   equality could manufacture an actual conflict right now;
    /// * **model-equal UF-argument pairs** — two terms that both occur as a
    ///   function-application argument (so merging them could trigger a
    ///   congruence EUF has not seen yet) and that the tableau's current
    ///   model happens to assign the same value — a pair the model already
    ///   disagrees on cannot possibly be entailed equal, so grouping by
    ///   value keeps this close to the number of live congruences instead of
    ///   the square of the interface size.
    ///
    /// Restricted to Int/Real-sorted terms: `track_theory_vars` also interns
    /// bit-vector terms into the arithmetic solver as a bounded-integer
    /// *relaxation*, and an "entailment" read off that partial view would not
    /// actually hold at the bit-vector level — handing it to EUF as a real
    /// equality would be unsound.
    ///
    /// # Under quantifiers
    ///
    /// The difference-constraint and live-disequality sources are not scoped
    /// to any one function symbol -- a pair like `(a, b)` from `a - b = 3`
    /// can involve terms no UF application ever mentions -- so there is no
    /// way to tell whether probing one would perturb an MBQI search the way
    /// `scope_rebase_tests::re_running_the_search_on_an_unchanged_goal_converges`
    /// caught (see `nelson_oppen_combine`'s old doc comment, preserved on the
    /// `TheoryManager::has_quantifiers` field, for the mechanism). Both stay
    /// off whenever `self.has_quantifiers` is true. The model-equal
    /// UF-argument source is different: every candidate it produces already
    /// names two UF-application arguments, and `self.quantifier_uf_funcs`
    /// records exactly the *function symbols* that occur as the head of an
    /// application under a binder (`Solver::purify_numeric_uf_args`'s
    /// per-function gate, mirrored here). Skipping the arguments of those
    /// functions while keeping every other application's is precise rather
    /// than a blanket skip.
    fn care_graph_candidates(&mut self) -> Vec<(TermId, TermId)> {
        let mut candidates: FxHashSet<(TermId, TermId)> = FxHashSet::default();

        if !self.has_quantifiers {
            let one = Rational64::from_integer(1);
            let neg_one = Rational64::from_integer(-1);

            for parsed in self.var_to_parsed_arith.values() {
                if parsed.terms.len() != 2 {
                    continue;
                }
                let (t0, c0) = parsed.terms[0];
                let (t1, c1) = parsed.terms[1];
                let is_difference = (c0 == one && c1 == neg_one) || (c0 == neg_one && c1 == one);
                if is_difference {
                    candidates.insert(order_pair(t0, t1));
                }
            }

            for (a, b) in self.euf.live_diseq_pairs() {
                candidates.insert(order_pair(a, b));
            }
        }

        // Only UF-argument terms can enable a *new* congruence when merged
        // (a UF *result*, or a plain arithmetic term never passed to any
        // function, cannot), so restrict the model-equal probe set to them
        // -- except the arguments of a quantifier-trigger function, which
        // `app_argument_terms_excluding_funcs` leaves out. `quantifier_uf_funcs`
        // is always empty when `has_quantifiers` is false (no quantifier was
        // ever registered to populate it), so this filter is a no-op for
        // every purely ground search. Queried fresh each round (not cached
        // from encode time) because it must also exclude the arguments of a
        // quantifier-trigger function's own MBQI-instantiated ground
        // applications, which only exist in the live e-graph.
        let forbidden_func_ids: FxHashSet<u32> = self
            .quantifier_uf_funcs
            .iter()
            .map(|spur| spur.into_inner().get())
            .collect();
        let uf_args = self
            .euf
            .app_argument_terms_excluding_funcs(&forbidden_func_ids);
        let mut by_value: FxHashMap<Rational64, SmallVec<[TermId; 4]>> = FxHashMap::default();
        for &term in self.arith.interface_terms() {
            if !uf_args.contains(&term) {
                continue;
            }
            if let Some(v) = self.arith.value(term) {
                by_value.entry(v).or_default().push(term);
            }
        }
        const MAX_MODEL_EQUAL_PAIRS: usize = 256;
        let mut added = 0usize;
        'buckets: for group in by_value.values() {
            for i in 0..group.len() {
                for j in (i + 1)..group.len() {
                    if added >= MAX_MODEL_EQUAL_PAIRS {
                        break 'buckets;
                    }
                    candidates.insert(order_pair(group[i], group[j]));
                    added += 1;
                }
            }
        }

        let manager = self.manager;
        let int_sort = manager.sorts.int_sort;
        let real_sort = manager.sorts.real_sort;
        let is_numeric = |t: TermId| -> bool {
            manager
                .get(t)
                .is_some_and(|term| term.sort == int_sort || term.sort == real_sort)
        };
        let mut out: Vec<(TermId, TermId)> = candidates
            .into_iter()
            .filter(|&(a, b)| is_numeric(a) && is_numeric(b))
            .collect();
        // Deterministic order: merge order decides which term becomes the
        // EUF class representative and which tag lands in `derived_reasons`,
        // so leaving this at hash-iteration order would make conflict
        // clauses — and potentially verdicts on some formulas — depend on
        // hashmap iteration order (`model_based_combination` sorts for the
        // identical reason).
        out.sort_unstable();
        out
    }
}
