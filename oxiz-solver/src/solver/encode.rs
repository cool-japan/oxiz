//! Term encoding (Tseitin transformation) for the SMT solver

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use oxiz_sat::{Lit, Var};

use super::Solver;
use super::trail::TrailOp;
use super::types::{ArithConstraintType, Constraint, NamedAssertion, Polarity, UnsatCore};

mod arith_atom_parse;
pub(super) mod bool_euf_encoding;
mod exists_skolem;
pub(crate) mod finite_expand;
mod finite_map_ite;
mod numeric_purification;
mod skolem_candidates;
mod track_theory_vars;

#[cfg(test)]
mod tests;

impl Solver {
    pub(super) fn get_or_create_var(&mut self, term: TermId) -> Var {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }

        let var = self.sat.new_var();
        self.term_to_var.insert(term, var);
        self.trail.push(TrailOp::VarCreated { var, term });

        while self.var_to_term.len() <= var.index() {
            self.var_to_term.push(TermId::new(0));
        }
        self.var_to_term[var.index()] = term;
        var
    }

    /// Remember that assertion `index` was `term`, under `name` if the caller
    /// gave one.
    ///
    /// # Why this is unconditional
    ///
    /// It used to be gated on `produce_unsat_cores`, which made unsat-core
    /// production silently order-dependent: a session that asserted first and
    /// set `:produce-unsat-cores` afterwards got `unsat` and an *empty* core,
    /// because the names were never written down — and nothing along the way
    /// reported that the option had arrived too late to be honoured.
    ///
    /// The gate belongs on *producing* a core, where it still is
    /// (`Solver::build_unsat_core`, `Solver::build_unsat_core_trivial_false`
    /// and `Solver::minimize_unsat_core` all check the flag), not on
    /// remembering what the caller asserted.  What it saved
    /// was one `Vec` push and one trail entry per assertion — the SAT-level work
    /// of core tracking was never behind this branch — so nothing is paid for
    /// the sessions that never ask for a core beyond the assertion names they
    /// themselves supplied.
    ///
    /// `named_assertions` is only ever searched by `index` (never assumed dense
    /// or aligned with `assertions`), so filling it in for every assertion
    /// changes no lookup.
    fn record_assertion_identity(&mut self, term: TermId, name: Option<String>, index: usize) {
        let na_index = self.named_assertions.len();
        self.named_assertions.push(NamedAssertion {
            term,
            name,
            index: index as u32,
        });
        self.trail
            .push(TrailOp::NamedAssertionAdded { index: na_index });
    }

    /// Record `term`'s Tseitin encoding in [`Solver::encoded_terms`], journalling
    /// the write so [`Solver::pop`] can take back exactly the entries whose
    /// defining clauses the matching `sat.pop()` retracts.
    ///
    /// The journal entry carries the displaced value, which is what makes the
    /// retraction *precise* rather than destructive: an entry widened from
    /// `Positive` to `Both` inside a scope has only the extra implication
    /// direction retracted with that scope, so `pop` must put the narrower
    /// pre-scope coverage back rather than forget the term altogether.
    ///
    /// A write that changes nothing is not journalled — re-encoding a term at
    /// the coverage it already has emits duplicate clauses but no new memo
    /// state, and a trail entry for it would only make `pop` do redundant work.
    fn memoize_encoding(&mut self, term: TermId, lit: Lit, polarity: Polarity) {
        let previous = self.encoded_terms.insert(term, (lit, polarity));
        if previous == Some((lit, polarity)) {
            return;
        }
        self.trail
            .push(TrailOp::EncodedTermAdded { term, previous });
    }

    /// Attach a theory constraint to `var`, journalling the write only when
    /// this scope is the one that introduced it.
    ///
    /// The encoder is re-entrant across assertion scopes: a term first encoded
    /// at an outer level keeps its SAT variable (`get_or_create_var` hits the
    /// cache), so asserting it again inside a `push` re-runs this code for a
    /// variable the outer scope already owns.  Journalling that repeat write
    /// would make the matching `pop` delete a constraint that is still active,
    /// leaving the atom without any theory meaning — the solver then loses the
    /// refutation that depends on it (`(or (= x 1) (= x 2)) ∧ (= x 5)` stopped
    /// being provably `unsat` after such a scope).  Recording only the first
    /// write keeps the trail entry paired with the scope that owns the fact.
    pub(super) fn record_constraint(&mut self, var: Var, constraint: Constraint) {
        if self.var_to_constraint.insert(var, constraint).is_none() {
            self.trail.push(TrailOp::ConstraintAdded { var });
        }
    }

    /// Mark `term` (a Bool-sorted `Var`) as appearing in a UF-application
    /// argument position, so the `TermKind::Var` arm of `encode_depth_uncached`
    /// registers it for Bool completion (`Constraint::BoolApp`) once it is
    /// encoded.
    ///
    /// Bool completion merges a Bool-sorted EUF class with the canonical
    /// true/false node by SAT-assigned value -- exactly what lets congruence
    /// fire over a UF application that takes Bool arguments. Registering
    /// *every* Bool variable for it unconditionally would charge that EUF
    /// bookkeeping to every propositional variable in the problem, most of
    /// which never appear near a UF application at all. Restricting it to
    /// the terms `abstract_compound_bool_args` actually found in argument
    /// position keeps the cost proportional to the UF-with-Bool-arguments
    /// shape that needs it.
    ///
    /// # Retroactive registration
    ///
    /// `encode_depth_uncached`'s `TermKind::Var` arm only consults
    /// `bool_uf_arg_terms` the *first* time `term` is encoded -- a later
    /// occurrence hits the `encoded_terms` memo and never re-runs the arm.
    /// If `term` was asserted (and so already encoded) by an *earlier*
    /// assertion, before a *later* one discovers it in argument position,
    /// the constraint would otherwise never be registered at all: `b1`
    /// asserted directly in one `assert` call, then used as `f`'s argument in
    /// a second one, must still be completed. So when `term` already has a
    /// SAT variable, register the constraint here, immediately, rather than
    /// waiting for an encode that has already happened.
    pub(super) fn mark_bool_uf_arg(&mut self, term: TermId) {
        if self.bool_uf_arg_terms.insert(term) {
            self.trail.push(TrailOp::BoolUfArgAdded { term });
            if let Some(&var) = self.term_to_var.get(&term) {
                self.record_constraint(var, Constraint::BoolApp(term));
            }
        }
    }

    /// Assert a term
    ///
    /// Strengthening the assertion stack invalidates the last verdict: a model
    /// found before this call need not satisfy `term`, so serving it afterwards
    /// would report a "model" of a formula it falsifies.  See
    /// `Solver::invalidate_results` (private) for the rule and for why the unsat
    /// core goes with it.
    pub fn assert(&mut self, term: TermId, manager: &mut TermManager) {
        let index = self.assertions.len();
        self.assertions.push(term);
        self.trail.push(TrailOp::AssertionAdded { index });
        self.invalidate_fp_cache();
        self.invalidate_results();

        // Check if this is a boolean constant first
        if let Some(t) = manager.get(term) {
            match t.kind {
                TermKind::False => {
                    // Mark that we have a false assertion
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    self.record_assertion_identity(term, None, index);
                    return;
                }
                TermKind::True => {
                    // True is always satisfied, no need to encode
                    self.record_assertion_identity(term, None, index);
                    return;
                }
                _ => {}
            }
        }

        // Overflow guard (soundness): a term nested far deeper than the encoder
        // can safely handle would overflow the native call stack in one of the
        // recursive passes below (simplification, polarity collection, or the
        // Tseitin encoder).  Detect excessive depth with an explicit-stack scan
        // and, when exceeded, skip every deep recursive pass for this assertion
        // and flag the incomplete encoding so `check` answers `Unknown` rather
        // than crashing the process.
        if self.term_exceeds_encode_depth(term, manager) {
            self.encode_depth_exceeded = true;
            self.record_assertion_identity(term, None, index);
            return;
        }

        // Apply simplification if enabled
        let simplified = if self.config.simplify {
            self.simplifier.simplify(term, manager)
        } else {
            term
        };

        // Replace bounded-integer quantifiers by their exactly equivalent
        // ground expansion so the ground solver decides them directly.
        let expanded = self
            .finite_expand_assertion(simplified, manager)
            .unwrap_or(simplified);

        // Replace the existentials this assertion states unconditionally by
        // their Skolemization, so the ground solver *searches* for a witness
        // instead of MBQI guessing one.
        let term_to_encode = self.skolemize_asserted_existentials(expanded, manager);

        // Check again if simplification produced a constant
        if let Some(t) = manager.get(term_to_encode) {
            match t.kind {
                TermKind::False => {
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    return;
                }
                TermKind::True => {
                    // Simplified to true, no need to encode
                    return;
                }
                _ => {}
            }
        }

        // Check for datatype constructor mutual exclusivity
        // If we see (= var Constructor), track it and check for conflicts
        if let Some(t) = manager.get(term_to_encode).cloned() {
            if let TermKind::Eq(lhs, rhs) = &t.kind {
                if let Some((var_term, constructor)) =
                    self.extract_dt_var_constructor(*lhs, *rhs, manager)
                {
                    if let Some(&existing_con) = self.dt_var_constructors.get(&var_term) {
                        if existing_con != constructor {
                            // Variable constrained to two different constructors - UNSAT
                            if !self.has_false_assertion {
                                self.has_false_assertion = true;
                                self.trail.push(TrailOp::FalseAssertionSet);
                            }
                            return;
                        }
                    } else {
                        self.dt_var_constructors.insert(var_term, constructor);
                        self.trail
                            .push(TrailOp::DtVarConstructorAdded { term: var_term });
                    }
                }
            }
        }

        // Eliminate non-Bool `ite` subterms and abstract compound Bool
        // UF-arguments into the equisatisfiable, congruence-closure-friendly
        // form `eliminate_nonbool_ite`/`abstract_compound_bool_args`
        // document.  Deliberately *after* the True/False re-check above (a
        // constant needs no rewriting) but *before* polarity collection and
        // MBQI registration, so those two passes -- and everything from here
        // on, including `self.assertions`' role as the thing get-assertions
        // and the array/string/BV/FP/datatype axiom passes walk -- see the
        // one term that is actually encoded, not a structure the SAT core's
        // clauses have since diverged from.  `self.assertions` itself still
        // stores the *pre*-rewrite `term` (see `record_assertion_identity`
        // below): the rewrite is purely an encoding-time concern; a caller
        // reading assertions back should see what it asserted.
        // Collapse right-leaning `(ite (= idx k1) v1 (ite (= idx k2) v2 …))`
        // lookup spines into one result constant plus flat key implications
        // *before* the generic mux eliminator below gets to them: each
        // nesting level would otherwise earn its own fresh mux variable,
        // turning an O(1)-guard-depth lookup into an O(n)-deep implication
        // chain (see `finite_map_ite`'s module doc).
        let term_to_encode = self.flatten_lookup_spines(term_to_encode, manager);
        let term_to_encode = self.eliminate_nonbool_ite(term_to_encode, manager);
        let term_to_encode = self.abstract_compound_bool_args(term_to_encode, manager);
        // Purify numeric (Int/Real) uninterpreted-function-application
        // arguments into fresh shared variables: `track_theory_vars` does
        // not intern a UF argument on its own, so a constant/compound
        // argument such as `f(3)` or `f(fmt1 + 1)` is never an arithmetic
        // interface term and an arithmetic-derived equality to it can never
        // reach EUF (see `numeric_purification` for the full picture — this
        // is the non-convex QF_UFLIA/QF_UFIDL false-`sat` root cause).
        let term_to_encode = self.purify_numeric_uf_args(term_to_encode, manager);

        // Collect polarity information if polarity-aware encoding is enabled
        if self.polarity_aware {
            self.collect_polarities(term_to_encode, Polarity::Positive, manager);
        }

        // Hand MBQI only the quantifiers this assertion actually entails.
        self.register_asserted_quantifiers(term_to_encode, manager);

        // Encode the assertion immediately.  Every numeric `Eq` atom inside
        // `term_to_encode` receives its trichotomy from
        // `Solver::add_numeric_trichotomy` during this call — see that method
        // for why the split lives on the atom rather than in a syntactic
        // pre-pass over the assertion.
        let lit = self.encode(term_to_encode, manager);
        self.sat.add_clause([lit]);

        self.record_assertion_identity(term, None, index);
    }

    /// Assert a named term (for unsat core tracking)
    ///
    /// Invalidates the last verdict for the same reason as [`Solver::assert`].
    pub fn assert_named(&mut self, term: TermId, name: &str, manager: &mut TermManager) {
        let index = self.assertions.len();
        self.assertions.push(term);
        self.trail.push(TrailOp::AssertionAdded { index });
        self.invalidate_fp_cache();
        self.invalidate_results();

        // Check if this is a boolean constant first
        if let Some(t) = manager.get(term) {
            match t.kind {
                TermKind::False => {
                    // Mark that we have a false assertion
                    if !self.has_false_assertion {
                        self.has_false_assertion = true;
                        self.trail.push(TrailOp::FalseAssertionSet);
                    }
                    self.record_assertion_identity(term, Some(name.to_string()), index);
                    return;
                }
                TermKind::True => {
                    // True is always satisfied, no need to encode
                    self.record_assertion_identity(term, Some(name.to_string()), index);
                    return;
                }
                _ => {}
            }
        }

        // Overflow guard (soundness): see `assert`.  Skip all deep recursive
        // passes for a pathologically deep term and flag the incomplete
        // encoding so `check` answers `Unknown` instead of overflowing.
        if self.term_exceeds_encode_depth(term, manager) {
            self.encode_depth_exceeded = true;
            self.record_assertion_identity(term, Some(name.to_string()), index);
            return;
        }

        // Replace bounded-integer quantifiers by their exactly equivalent
        // ground expansion so the ground solver decides them directly.
        let expanded = self.finite_expand_assertion(term, manager).unwrap_or(term);

        // Replace the existentials this assertion states unconditionally by
        // their Skolemization (see `assert`).
        let term_to_encode = self.skolemize_asserted_existentials(expanded, manager);

        // See `Solver::assert` for why this runs here: after skolemization,
        // before polarity collection / MBQI registration, and without
        // touching `self.assertions` (still `term`, pushed above). Lookup-spine
        // flattening runs first for the same reason it does in `assert` — see
        // that call site's comment.
        let term_to_encode = self.flatten_lookup_spines(term_to_encode, manager);
        let term_to_encode = self.eliminate_nonbool_ite(term_to_encode, manager);
        let term_to_encode = self.abstract_compound_bool_args(term_to_encode, manager);
        // See `Solver::assert` for why this runs here too: named assertions
        // go through the same pre-pass chain, and a numeric UF argument in a
        // named assertion needs purification just as much as an anonymous
        // one does.
        let term_to_encode = self.purify_numeric_uf_args(term_to_encode, manager);

        // Collect polarity information if polarity-aware encoding is enabled
        if self.polarity_aware {
            self.collect_polarities(term_to_encode, Polarity::Positive, manager);
        }

        // Hand MBQI only the quantifiers this assertion actually entails.
        self.register_asserted_quantifiers(term_to_encode, manager);

        // Encode the assertion immediately.  As in `Solver::assert`, the
        // numeric `Eq` atoms get their trichotomy from
        // `Solver::add_numeric_trichotomy` inside this `encode` call.
        let lit = self.encode(term_to_encode, manager);
        self.sat.add_clause([lit]);

        self.record_assertion_identity(term, Some(name.to_string()), index);
    }

    /// Get the unsat core (after check() returned Unsat)
    #[must_use]
    pub fn get_unsat_core(&self) -> Option<&UnsatCore> {
        self.unsat_core.as_ref()
    }

    /// Rewrite every bounded-integer quantifier of `term` into its exactly
    /// equivalent finite conjunction / disjunction, or `None` when nothing in
    /// `term` qualifies.
    ///
    /// The rewrite is an equivalence, not a strengthening (see
    /// [`finite_expand`]): the whole substituted body — guard included — is
    /// emitted for every point of the interval, and the interval provably
    /// contains the entire region where the guard (`forall`) or the body
    /// (`exists`) can be true.  So the expanded assertion is interchangeable
    /// with the original at any polarity, and the quantifier disappears
    /// altogether instead of being handed to MBQI.  That is exactly what lets
    /// the *ground* array / arithmetic solver decide it, which is how an
    /// `(exists ((i Int)) (and (<= 0 i) (<= i 9) (= (select a i) v)))` over a
    /// pinned array finds its witness.
    ///
    /// A quantifier that does not fit the fragment is left untouched and keeps
    /// its normal MBQI path, so declining costs completeness only.
    /// Replace the existentials `term` asserts unconditionally by their
    /// Skolemization, returning `term` unchanged when it has none.
    ///
    /// See [`exists_skolem`] for why the rewrite is an equisatisfiability
    /// (never a strengthening) and why it is confined to the positive
    /// top-level conjunct spine.
    fn skolemize_asserted_existentials(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        let mut next_id = self.next_skolem_id;
        let rewritten = exists_skolem::skolemize_asserted_existentials(term, manager, &mut next_id);
        self.next_skolem_id = next_id;
        rewritten.unwrap_or(term)
    }

    fn finite_expand_assertion(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let budget = self.config.finite_expansion_budget;
        if budget == 0 || !finite_expand::contains_quantifier(term, manager) {
            return None;
        }
        self.refresh_entailed_int_constants(manager);
        let entailed = core::mem::take(&mut self.entailed_int_consts);
        let expanded = finite_expand::expand_finite_quantifiers(term, manager, budget, &entailed);
        self.entailed_int_consts = entailed;
        expanded
    }

    /// Bring [`Solver::entailed_int_consts`] up to date with the assertion
    /// stack: for every term some **top-level** assertion pins to an integer
    /// literal, record the value every model must give it.
    ///
    /// Each assertion's top-level `And` spine is walked, because a conjunct of
    /// an unconditionally asserted conjunction is itself unconditionally
    /// asserted.  Nothing below a disjunction, negation, implication or
    /// quantifier is collected — those equalities are conditional and would not
    /// hold in every model, so using one as a quantifier bound could expand
    /// over the wrong interval.
    ///
    /// Only the assertions added since the last call are folded in, and `pop`
    /// resets both the map and the watermark (see [`Solver::pop`]), so the map
    /// is always exactly "consequences of the live assertion set" and the total
    /// scanning cost stays linear in the number of assertions.
    ///
    /// This is what lets `(assert (= n 5))` make `(< i n)` a *concrete* bound
    /// for [`Solver::finite_expand_assertion`]; without it a symbolic bound
    /// would leave the quantifier unexpanded.
    fn refresh_entailed_int_constants(&mut self, manager: &TermManager) {
        if self.entailed_int_consts_upto >= self.assertions.len() {
            return;
        }
        let mut entailed = core::mem::take(&mut self.entailed_int_consts);
        let int_sort = manager.sorts.int_sort;
        let mut stack: Vec<TermId> = Vec::new();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        for &assertion in &self.assertions[self.entailed_int_consts_upto..] {
            stack.push(assertion);
            while let Some(current) = stack.pop() {
                if !visited.insert(current) {
                    continue;
                }
                let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
                    continue;
                };
                match kind {
                    TermKind::And(args) => stack.extend(args.iter().copied()),
                    TermKind::Eq(lhs, rhs) => {
                        let pinned = match (
                            manager.get(lhs).map(|t| t.kind.clone()),
                            manager.get(rhs).map(|t| t.kind.clone()),
                        ) {
                            (Some(TermKind::IntConst(value)), _) => Some((rhs, value)),
                            (_, Some(TermKind::IntConst(value))) => Some((lhs, value)),
                            _ => None,
                        };
                        if let Some((symbol, value)) = pinned
                            && manager.get(symbol).is_some_and(|t| t.sort == int_sort)
                        {
                            entailed.entry(symbol).or_insert(value);
                        }
                    }
                    _ => {}
                }
            }
        }
        self.entailed_int_consts = entailed;
        self.entailed_int_consts_upto = self.assertions.len();
    }

    /// Register with MBQI and the E-matching engine every quantifier that the
    /// assertion `term` asserts **unconditionally**.
    ///
    /// MBQI turns a registered universal into ground instances that it adds to
    /// the SAT core as hard unit clauses — and, when an instance evaluates to
    /// `false`, as the empty clause — so a quantifier the assertion set does not
    /// entail must never be registered. `(not (forall ((x Int)) (P x)))` is
    /// `∃x. ¬P(x)`, not a universal fact; registering it refuted the satisfiable
    /// `(not (forall ((x Int)) (P x))) ∧ (not (P 5))`. The same held for a
    /// quantifier inside a disjunct, an implication, an `ite` branch, or a
    /// Bool-sorted equality's operand.
    ///
    /// [`Solver::encode`] cannot make this call: it is the Tseitin transform and
    /// visits every sub-term at every polarity. So the decision is made here, on
    /// the asserted spine, with [`super::term_walk::asserted_children`] — the
    /// shared definition of "unconditionally asserted" also used by the
    /// `check_*.rs` definite-conflict collectors.
    ///
    /// Skipping a non-entailed quantifier costs only completeness: `check`
    /// still sees `has_quantifiers`, so it answers `Unknown` rather than
    /// guessing.
    fn register_asserted_quantifiers(&mut self, term: TermId, manager: &mut TermManager) {
        let mut stack: Vec<(TermId, bool)> = vec![(term, true)];
        let mut visited: FxHashSet<(TermId, bool)> = FxHashSet::default();

        while let Some((current, positive)) = stack.pop() {
            if !visited.insert((current, positive)) {
                continue;
            }
            let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
                continue;
            };
            if positive {
                match &kind {
                    TermKind::Forall { patterns, body, .. } => {
                        let triggers: Vec<TermId> =
                            patterns.iter().flat_map(|p| p.iter().copied()).collect();
                        self.register_asserted_forall(current, *body, triggers, manager);
                    }
                    TermKind::Exists { patterns, body, .. } => {
                        let triggers: Vec<TermId> =
                            patterns.iter().flat_map(|p| p.iter().copied()).collect();
                        self.mbqi.add_quantifier(current, manager);
                        for trigger in triggers {
                            self.mbqi.collect_ground_terms(trigger, manager);
                        }
                        self.collect_quantifier_uf_funcs(*body, manager);
                    }
                    _ => {}
                }
            }
            stack.extend(super::term_walk::asserted_children(&kind, positive));
        }
    }

    /// Record every uninterpreted-function symbol occurring as the head of
    /// an `Apply` reachable from `term` into `self.quantifier_uf_funcs`.
    ///
    /// Called on a quantifier's *body* (not its patterns alone) as each
    /// quantifier is registered, so `purify_numeric_uf_args` can tell a
    /// function that genuinely occurs under a binder apart from one that
    /// merely coexists with an unrelated quantifier elsewhere in the
    /// script -- see [`Solver::purify_numeric_uf_args`]'s doc comment for
    /// why that distinction is what keeps the fix sound regardless of
    /// assertion order. Unlike `collect_ground_subterms`, this walks *into*
    /// nested binders (`get_children` descends through `Forall`/`Exists`/
    /// `Let`) since a function applied under a further-nested quantifier is
    /// exactly as e-matching-relevant as one applied directly. Iterative
    /// with an explicit stack and a `visited` set, matching every other DAG
    /// walk in this file.
    fn collect_quantifier_uf_funcs(&mut self, term: TermId, manager: &TermManager) {
        let mut stack: Vec<TermId> = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(t) = manager.get(current) else {
                continue;
            };
            if let TermKind::Apply { func, .. } = &t.kind {
                self.quantifier_uf_funcs.insert(*func);
            }
            stack.extend(oxiz_core::ast::get_children(&t.kind));
        }
    }

    /// Register one unconditionally asserted universal, Skolemizing a nested
    /// existential body (`∀x. ∃y. φ(x,y)` → `∀x. φ(x, sk(x))`) first so that
    /// MBQI sees a plain universal.
    fn register_asserted_forall(
        &mut self,
        term: TermId,
        body: TermId,
        triggers: Vec<TermId>,
        manager: &mut TermManager,
    ) {
        self.collect_quantifier_uf_funcs(body, manager);

        let body_is_exists = manager
            .get(body)
            .is_some_and(|t| matches!(t.kind, TermKind::Exists { .. }));

        if body_is_exists {
            #[cfg(feature = "std")]
            {
                // Seeded from the solver-wide counter: a fresh context always
                // starts at `sk!0` / `skf!0`, so two Skolemized quantifiers
                // would otherwise share one witness symbol — a strengthening
                // that can turn `sat` into `unsat`.
                let mut sk_ctx =
                    crate::skolemization::SkolemizationContext::with_first_id(self.next_skolem_id);
                let skolem_result = sk_ctx.skolemize(manager, term);
                self.next_skolem_id = sk_ctx.skolem_count();
                if let Ok(skolemized) = skolem_result {
                    self.mbqi.add_quantifier(skolemized, manager);
                    let _ = self.ematch_engine.register_quantifier(skolemized, manager);

                    // Also collect Skolem function application terms from the
                    // Skolemized body as MBQI candidates.  These terms (e.g.
                    // sk(x)) must appear in the candidate pool so that other
                    // universal quantifiers can be instantiated with them.
                    self.collect_skolem_candidates(skolemized, manager);
                } else {
                    // Skolemization failed — fall back to original
                    self.mbqi.add_quantifier(term, manager);
                    let _ = self.ematch_engine.register_quantifier(term, manager);
                }
            }
            #[cfg(not(feature = "std"))]
            {
                self.mbqi.add_quantifier(term, manager);
                let _ = self.ematch_engine.register_quantifier(term, manager);
            }
        } else {
            self.mbqi.add_quantifier(term, manager);
            // Register with E-matching engine for trigger-based instantiation
            let _ = self.ematch_engine.register_quantifier(term, manager);
        }

        // Collect ground terms from patterns as candidates
        for trigger in triggers {
            self.mbqi.collect_ground_terms(trigger, manager);
        }
    }

    /// Encode a term into SAT clauses using Tseitin transformation.
    ///
    /// Thin wrapper over [`Solver::encode_depth`] that starts the recursion at
    /// depth 0.  The depth counter guards against native-stack overflow on
    /// adversarially deep formulas (see [`ENCODE_DEPTH_LIMIT`](super::ENCODE_DEPTH_LIMIT)).
    pub(super) fn encode(&mut self, term: TermId, manager: &mut TermManager) -> Lit {
        self.encode_depth(term, manager, 0)
    }

    /// Depth-tracked recursive Tseitin encoder: memo check, depth guard, then
    /// the arm dispatch in [`Solver::encode_depth_uncached`].
    ///
    /// # Memoisation (`Solver::encoded_terms`)
    ///
    /// Each term's clauses are emitted at most once per polarity coverage:
    /// without the memo, a shared sub-term of the hash-consed DAG was
    /// re-descended once per *edge*, which is `2^n` re-encodes and `2^n`
    /// duplicate clauses on a doubling DAG (each level referencing the
    /// previous twice) — a hang at roughly depth 40.  The assert-time
    /// pre-check `term_exceeds_encode_depth` cannot catch that input: it
    /// measures depth and deliberately prunes shared nodes.
    ///
    /// A cached entry is only reused when the polarity it was encoded under
    /// covers the polarity the current occurrence needs (see the field doc on
    /// [`Solver::encoded_terms`]): `And`/`Or` under `polarity_aware` emit only
    /// one implication direction, every other arm is polarity-independent.  On
    /// a widening miss the term is re-encoded, which appends the missing
    /// direction; re-emitting the direction that already exists only
    /// duplicates clauses and cannot change the encoded semantics.
    ///
    /// The memo is consulted *before* the depth guard: a hit means the term's
    /// full encoding already exists in the SAT core, so returning it is always
    /// complete, whereas the pre-memo code would have set
    /// [`Solver::encode_depth_exceeded`] even for an already-encoded term.
    ///
    /// # Depth guard
    ///
    /// When the structural recursion exceeds [`ENCODE_DEPTH_LIMIT`](super::ENCODE_DEPTH_LIMIT) we stop
    /// descending, set [`Solver::encode_depth_exceeded`], and return a fresh
    /// literal for the sub-term.  The truncated encoding is deliberately
    /// incomplete: `check` observes the flag and answers `Unknown` rather than
    /// crashing the process with a stack overflow or trusting a partial model.
    /// Nothing is memoised on this path — the term was *not* encoded, and a
    /// later shallower occurrence must still get a real encoding.
    pub(super) fn encode_depth(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
        depth: u32,
    ) -> Lit {
        // The polarity the And/Or arms would emit clauses under right now.
        // This must be the same lookup those arms perform; the map is not
        // mutated while an encode is in flight, so the two reads agree.
        let needed = if self.polarity_aware {
            self.polarities
                .get(&term)
                .copied()
                .unwrap_or(Polarity::Both)
        } else {
            Polarity::Both
        };
        if let Some(&(lit, cached)) = self.encoded_terms.get(&term) {
            if cached == Polarity::Both || cached == needed {
                return lit;
            }
        }
        if depth > super::ENCODE_DEPTH_LIMIT {
            self.encode_depth_exceeded = true;
            let var = self.get_or_create_var(term);
            return Lit::pos(var);
        }
        let lit = self.encode_depth_uncached(term, manager, depth);
        // The And/Or arms have already inserted their entry with the polarity
        // they actually used; every other arm's clause set is
        // polarity-independent, so `Both` is the correct coverage for it.
        if !self.encoded_terms.contains_key(&term) {
            self.memoize_encoding(term, lit, Polarity::Both);
        }
        lit
    }

    /// The arm dispatch of the Tseitin encoder.  Only called by
    /// [`Solver::encode_depth`], which owns the memo lookup and the depth
    /// guard; recursive descent goes back through `encode_depth` so every
    /// sub-term gets both.
    fn encode_depth_uncached(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
        depth: u32,
    ) -> Lit {
        // Clone the term data to avoid borrowing issues
        let Some(t) = manager.get(term).cloned() else {
            let var = self.get_or_create_var(term);
            return Lit::pos(var);
        };

        match &t.kind {
            TermKind::True => {
                let var = self.get_or_create_var(manager.mk_true());
                self.sat.add_clause([Lit::pos(var)]);
                Lit::pos(var)
            }
            TermKind::False => {
                let var = self.get_or_create_var(manager.mk_false());
                self.sat.add_clause([Lit::neg(var)]);
                // `encode` must return a literal whose truth value *equals* the
                // term's.  The unit clause above pins `var := false`, so the
                // literal standing for `false` is `pos(var)` — `neg(var)` would
                // evaluate to `true` and invert every nested occurrence of the
                // constant (mirrors the `True` arm, which pins and returns
                // `pos(var)`).
                Lit::pos(var)
            }
            TermKind::Var(_) => {
                let var = self.get_or_create_var(term);
                // Bool completion: a Bool variable known to appear in a
                // UF-application argument position (marked by
                // `abstract_compound_bool_args`) must merge with the
                // canonical true/false EUF node by SAT-assigned value, so
                // congruence can fire over the application that uses it. Not
                // unconditional -- see `Solver::mark_bool_uf_arg` for why.
                if t.sort == manager.sorts.bool_sort && self.bool_uf_arg_terms.contains(&term) {
                    self.record_constraint(var, Constraint::BoolApp(term));
                }
                // Track theory terms for model extraction
                let is_int = t.sort == manager.sorts.int_sort;
                let is_real = t.sort == manager.sorts.real_sort;

                if is_int || is_real {
                    // Track arithmetic terms
                    if !self.arith_terms.contains(&term) {
                        self.arith_terms.insert(term);
                        self.trail.push(TrailOp::ArithTermAdded { term });
                        // Register with arithmetic solver
                        self.arith.intern(term);
                    }
                } else if let Some(sort) = manager.sorts.get(t.sort)
                    && sort.is_bitvec()
                    && !self.bv_terms.contains(&term)
                {
                    self.bv_terms.insert(term);
                    self.trail.push(TrailOp::BvTermAdded { term });
                    // Register with BV solver if not already registered
                    if let Some(width) = sort.bitvec_width() {
                        self.bv.new_bv(term, width);
                    }
                }
                Lit::pos(var)
            }
            TermKind::Not(arg) => {
                let arg_lit = self.encode_depth(*arg, manager, depth + 1);
                arg_lit.negate()
            }
            TermKind::And(args) => {
                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut arg_lits: Vec<Lit> = Vec::new();
                for &arg in args {
                    arg_lits.push(self.encode_depth(arg, manager, depth + 1));
                }

                // Get polarity for optimization
                let polarity = if self.polarity_aware {
                    self.polarities
                        .get(&term)
                        .copied()
                        .unwrap_or(Polarity::Both)
                } else {
                    Polarity::Both
                };

                // result => all args (needed when result is positive)
                // ~result or arg1, ~result or arg2, ...
                if polarity != Polarity::Negative {
                    for &arg in &arg_lits {
                        self.sat.add_clause([result.negate(), arg]);
                    }
                }

                // all args => result (needed when result is negative)
                // ~arg1 or ~arg2 or ... or result
                if polarity != Polarity::Positive {
                    let mut clause: Vec<Lit> = arg_lits.iter().map(|l| l.negate()).collect();
                    clause.push(result);
                    self.sat.add_clause(clause);
                }

                // Record the polarity these clauses were emitted under; a later
                // occurrence needing the other direction must re-encode (see
                // `encode_depth`).  An overwriting write (not `or_insert`) so a
                // widening re-encode upgrades a previously narrower entry.
                self.memoize_encoding(term, result, polarity);

                result
            }
            TermKind::Or(args) => {
                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut arg_lits: Vec<Lit> = Vec::new();
                for &arg in args {
                    arg_lits.push(self.encode_depth(arg, manager, depth + 1));
                }

                // Get polarity for optimization
                let polarity = if self.polarity_aware {
                    self.polarities
                        .get(&term)
                        .copied()
                        .unwrap_or(Polarity::Both)
                } else {
                    Polarity::Both
                };

                // result => some arg (needed when result is positive)
                // ~result or arg1 or arg2 or ...
                if polarity != Polarity::Negative {
                    let mut clause: Vec<Lit> = vec![result.negate()];
                    clause.extend(arg_lits.iter().copied());
                    self.sat.add_clause(clause);
                }

                // some arg => result (needed when result is negative)
                // ~arg1 or result, ~arg2 or result, ...
                if polarity != Polarity::Positive {
                    for &arg in &arg_lits {
                        self.sat.add_clause([arg.negate(), result]);
                    }
                }

                // Same polarity bookkeeping as the `And` arm above.
                self.memoize_encoding(term, result, polarity);

                result
            }
            TermKind::Xor(lhs, rhs) => {
                let lhs_lit = self.encode_depth(*lhs, manager, depth + 1);
                let rhs_lit = self.encode_depth(*rhs, manager, depth + 1);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (lhs xor rhs)
                // result <=> (lhs and ~rhs) or (~lhs and rhs)

                // result => (lhs or rhs)
                self.sat.add_clause([result.negate(), lhs_lit, rhs_lit]);
                // result => (~lhs or ~rhs)
                self.sat
                    .add_clause([result.negate(), lhs_lit.negate(), rhs_lit.negate()]);

                // (lhs and ~rhs) => result
                self.sat.add_clause([lhs_lit.negate(), rhs_lit, result]);
                // (~lhs and rhs) => result
                self.sat.add_clause([lhs_lit, rhs_lit.negate(), result]);

                result
            }
            TermKind::Implies(lhs, rhs) => {
                let lhs_lit = self.encode_depth(*lhs, manager, depth + 1);
                let rhs_lit = self.encode_depth(*rhs, manager, depth + 1);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (~lhs or rhs)
                // result => ~lhs or rhs
                self.sat
                    .add_clause([result.negate(), lhs_lit.negate(), rhs_lit]);

                // (~lhs or rhs) => result
                // lhs or result, ~rhs or result
                self.sat.add_clause([lhs_lit, result]);
                self.sat.add_clause([rhs_lit.negate(), result]);

                result
            }
            TermKind::Ite(cond, then_br, else_br) => {
                let cond_lit = self.encode_depth(*cond, manager, depth + 1);
                let then_lit = self.encode_depth(*then_br, manager, depth + 1);
                let else_lit = self.encode_depth(*else_br, manager, depth + 1);

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                // result <=> (cond ? then : else)
                // cond and result => then
                self.sat
                    .add_clause([cond_lit.negate(), result.negate(), then_lit]);
                // cond and then => result
                self.sat
                    .add_clause([cond_lit.negate(), then_lit.negate(), result]);

                // ~cond and result => else
                self.sat.add_clause([cond_lit, result.negate(), else_lit]);
                // ~cond and else => result
                self.sat.add_clause([cond_lit, else_lit.negate(), result]);

                result
            }
            TermKind::Eq(lhs, rhs) => {
                // Check if this is a boolean equality or theory equality
                let lhs_term = manager.get(*lhs);
                let is_bool_eq = lhs_term.is_some_and(|t| t.sort == manager.sorts.bool_sort);

                if is_bool_eq {
                    // Boolean equality: encode as iff
                    let lhs_lit = self.encode_depth(*lhs, manager, depth + 1);
                    let rhs_lit = self.encode_depth(*rhs, manager, depth + 1);

                    let result_var = self.get_or_create_var(term);
                    let result = Lit::pos(result_var);

                    // result <=> (lhs <=> rhs)
                    // result => (lhs => rhs) and (rhs => lhs)
                    self.sat
                        .add_clause([result.negate(), lhs_lit.negate(), rhs_lit]);
                    self.sat
                        .add_clause([result.negate(), rhs_lit.negate(), lhs_lit]);

                    // (lhs <=> rhs) => result
                    self.sat.add_clause([lhs_lit, rhs_lit, result]);
                    self.sat
                        .add_clause([lhs_lit.negate(), rhs_lit.negate(), result]);

                    // ALSO register the equality as a theory constraint, so
                    // EUF learns `lhs = lhs` too, not just the SAT-level iff.
                    // The iff gate above is complete for pure-boolean
                    // propagation but cannot express *congruence*: if some
                    // Bool-returning `g` is applied to `lhs`/`rhs` elsewhere,
                    // deriving `g(lhs) = g(rhs)` needs EUF to know `lhs = rhs`
                    // as an asserted fact, which only `Constraint::Eq`
                    // delivers. Without this, a Bool-sorted equality between
                    // two UF arguments never reaches congruence closure and
                    // `QF_UF` problems built on Bool-sorted uninterpreted
                    // predicates can go false-`sat`. The two mechanisms are
                    // complementary, not redundant: the iff gate pins SAT
                    // values, `Constraint::Eq` pins the EUF class.
                    self.record_constraint(result_var, Constraint::Eq(*lhs, *rhs));
                    self.track_theory_vars(*lhs, manager);
                    self.track_theory_vars(*rhs, manager);

                    result
                } else {
                    // Theory equality: create a fresh boolean variable
                    // Store the constraint for theory propagation
                    let var = self.get_or_create_var(term);
                    self.record_constraint(var, Constraint::Eq(*lhs, *rhs));

                    // Track theory variables for model extraction
                    self.track_theory_vars(*lhs, manager);
                    self.track_theory_vars(*rhs, manager);

                    // Pre-parse arithmetic equality for ArithSolver
                    // Only for Int/Real sorts, not BitVec
                    let is_arith = lhs_term.is_some_and(|t| {
                        t.sort == manager.sorts.int_sort || t.sort == manager.sorts.real_sort
                    });
                    if is_arith {
                        // We use Le type as placeholder since equality will be asserted
                        // as both Le and Ge
                        if let Some(parsed) = self.parse_arith_comparison(
                            *lhs,
                            *rhs,
                            ArithConstraintType::Le,
                            term,
                            manager,
                        ) {
                            self.var_to_parsed_arith.insert(var, parsed);
                        }

                        // Give this atom its trichotomy `(a = b) OR (a < b) OR
                        // (a > b)` right here, at the one place every numeric
                        // `Eq` atom is guaranteed to pass through.
                        //
                        // The theory layer has no other way to hear about a
                        // *negative* numeric equality: `TheoryManager::
                        // process_constraint`'s `Constraint::Eq` arm reaches
                        // `ArithSolver::assert_eq` only under `is_positive`,
                        // and its negative branch speaks to EUF and BV alone
                        // (`ArithSolver` has no `assert_neq` -- a disequality
                        // is not a convex constraint and the tableau cannot
                        // hold one). So an `Eq` the SAT core assigns *false*
                        // constrains nothing at all: the LP is free to hand
                        // both sides the same value and the search reports a
                        // model that violates the very disequality it just
                        // committed to. That is the false-`sat` on the
                        // `QF_LIA`/`QF_IDL`/`QF_UFLIA`/`QF_AUFLIA` family.
                        //
                        // The clause is a *tautology* over a total order, so
                        // emitting it unconditionally is sound in any Boolean
                        // context -- it adds no constraint of its own, it only
                        // makes the case split explicit for the SAT core.
                        // (Emitting the unguarded split `(a < b) OR (a > b)`
                        // instead would NOT be: for `(or p (not (= x 0)))` it
                        // would force `x != 0` even in models that satisfy the
                        // formula through `p`.) Once the context forces
                        // `(a = b)` false, unit propagation leaves
                        // `(a < b) OR (a > b)` and whichever strict atom the
                        // core picks *does* reach the tableau through the
                        // ordinary `Constraint::Lt`/`Gt` path.
                        //
                        // Doing it here rather than in a syntactic pre-pass
                        // over the assertion is what makes it complete. The
                        // four walks this replaced (`add_arith_diseq_split`,
                        // `add_arith_trichotomy_clause`,
                        // `add_arith_eq_trichotomy` and the never-called
                        // `add_arith_diseq_splits_for_sat_model`, all now
                        // deleted) looked for `Not(Eq(..))`/`Distinct(..)` and
                        // enumerated only a handful of connectives, so an `Eq`
                        // reachable only through a shape they did not
                        // enumerate (`Xor`, an `Implies` antecedent, a nested
                        // `Eq`, a `Let`) stayed a free Boolean. `encode_depth`
                        // is the single funnel every numeric `Eq` atom --
                        // asserted, MBQI-instantiated or axiom-generated --
                        // must pass to get a SAT variable at all, so attaching
                        // the split to the atom removes the dependence on
                        // *where* it sits.
                        //
                        // Removing those walks also removed an unbounded cost:
                        // each ran with only a *per-call* `visited` set and no
                        // cross-call ledger, so every one of the five MBQI /
                        // E-matching call sites re-emitted literal-identical
                        // trichotomy clauses on every round, and
                        // `oxiz_sat::Solver::add_clause` does not deduplicate.
                        // Each of those sites called `encode` on the very same
                        // term one line earlier, so this arm had already
                        // emitted the clause they duplicated.
                        //
                        // `encoded_terms` memoises this arm per term, so the
                        // three clauses are emitted once per atom, and the
                        // literals reuse atoms `encode` creates anyway.
                        self.add_numeric_trichotomy(term, var, *lhs, *rhs, manager, depth);
                    }

                    // A non-Bool `ite` in either operand denotes a
                    // conditional value EUF cannot see through on its own
                    // (it interns the `ite` as an opaque leaf). Add the
                    // narrow forward-implication clauses so that, once the
                    // SAT core pins the condition, the corresponding branch
                    // equality is forced and reaches congruence closure. This
                    // is a *complement* to `Solver::eliminate_nonbool_ite`
                    // (the whole-assertion pre-pass in `assert`/
                    // `assert_named`), not a duplicate of it: this arm also
                    // runs on lemmas reached through `Solver::encode`
                    // directly -- MBQI instantiation and the array/datatype/
                    // arithmetic axiom passes -- which never go through
                    // `assert` at all.
                    self.encode_nonbool_ite_equality(var, *lhs, *rhs, manager, depth);

                    Lit::pos(var)
                }
            }
            TermKind::Distinct(args) => {
                // Encode distinct as pairwise disequalities
                // distinct(a,b,c) <=> (a!=b) and (a!=c) and (b!=c)
                if args.len() <= 1 {
                    // trivially true
                    let var = self.get_or_create_var(manager.mk_true());
                    return Lit::pos(var);
                }

                let result_var = self.get_or_create_var(term);
                let result = Lit::pos(result_var);

                let mut diseq_lits = Vec::new();
                for i in 0..args.len() {
                    for j in (i + 1)..args.len() {
                        let eq = manager.mk_eq(args[i], args[j]);
                        let eq_lit = self.encode_depth(eq, manager, depth + 1);
                        diseq_lits.push(eq_lit.negate());
                    }
                }

                // result => all disequalities
                for &diseq in &diseq_lits {
                    self.sat.add_clause([result.negate(), diseq]);
                }

                // all disequalities => result
                let mut clause: Vec<Lit> = diseq_lits.iter().map(|l| l.negate()).collect();
                clause.push(result);
                self.sat.add_clause(clause);

                result
            }
            TermKind::Let { bindings, body } => {
                // For encoding, we can substitute the bindings into the body
                // This is a simplification - a more sophisticated approach would
                // memoize the bindings
                let substituted = *body;
                for (name, value) in bindings.iter().rev() {
                    // In a full implementation, we'd perform proper substitution
                    // For now, just encode the body directly
                    let _ = (name, value);
                }
                self.encode_depth(substituted, manager, depth + 1)
            }
            // Theory atoms (arithmetic, bitvec, arrays, UF)
            // These get fresh boolean variables - the theory solver handles the semantics
            TermKind::IntConst(_) | TermKind::RealConst(_) | TermKind::BitVecConst { .. } => {
                // Constants are theory terms, not boolean formulas
                // Should not appear at top level in boolean context
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Neg(_)
            | TermKind::Add(_)
            | TermKind::Sub(_, _)
            | TermKind::Mul(_)
            | TermKind::Div(_, _)
            | TermKind::Mod(_, _) => {
                // Arithmetic terms - should not appear at boolean top level
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Lt(lhs, rhs) => {
                // Arithmetic predicate: lhs < rhs
                let var = self.get_or_create_var(term);
                self.record_constraint(var, Constraint::Lt(*lhs, *rhs));
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Lt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Le(lhs, rhs) => {
                // Arithmetic predicate: lhs <= rhs
                let var = self.get_or_create_var(term);
                self.record_constraint(var, Constraint::Le(*lhs, *rhs));
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Le, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Gt(lhs, rhs) => {
                // Arithmetic predicate: lhs > rhs
                let var = self.get_or_create_var(term);
                self.record_constraint(var, Constraint::Gt(*lhs, *rhs));
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Gt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Ge(lhs, rhs) => {
                // Arithmetic predicate: lhs >= rhs
                let var = self.get_or_create_var(term);
                self.record_constraint(var, Constraint::Ge(*lhs, *rhs));
                // Parse and store linear constraint for ArithSolver
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Ge, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvConcat(_, _)
            | TermKind::BvExtract { .. }
            | TermKind::BvNot(_)
            | TermKind::BvAnd(_, _)
            | TermKind::BvOr(_, _)
            | TermKind::BvXor(_, _)
            | TermKind::BvAdd(_, _)
            | TermKind::BvSub(_, _)
            | TermKind::BvMul(_, _)
            | TermKind::BvShl(_, _)
            | TermKind::BvLshr(_, _)
            | TermKind::BvAshr(_, _) => {
                // Bitvector terms - should not appear at boolean top level
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::BvUdiv(_, _)
            | TermKind::BvSdiv(_, _)
            | TermKind::BvUrem(_, _)
            | TermKind::BvSrem(_, _) => {
                // Bitvector arithmetic terms (division/remainder)
                // Mark that we have arithmetic BV ops for conflict checking
                self.has_bv_arith_ops = true;
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::BvUlt(lhs, rhs) => {
                // Bitvector unsigned less-than: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.record_constraint(var, Constraint::Lt(*lhs, *rhs));
                // Parse as arithmetic constraint (bitvector as bounded integer)
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Lt, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvUle(lhs, rhs) => {
                // Bitvector unsigned less-than-or-equal: treat as integer comparison
                let var = self.get_or_create_var(term);
                self.record_constraint(var, Constraint::Le(*lhs, *rhs));
                if let Some(parsed) =
                    self.parse_arith_comparison(*lhs, *rhs, ArithConstraintType::Le, term, manager)
                {
                    self.var_to_parsed_arith.insert(var, parsed);
                }
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvSlt(lhs, rhs) => {
                // Bitvector *signed* less-than.  The `Constraint::Lt` recorded
                // here is consumed by the BV theory path in
                // `TheoryManager::process_constraint`, which recovers the
                // signedness from this term's `TermKind` and asserts a proper
                // two's-complement `assert_slt` into the BV solver.
                //
                // We deliberately do NOT populate `var_to_parsed_arith`: that
                // path parses BV operands as plain *unsigned* non-negative
                // integers and asserts them into the linear ArithSolver.  For a
                // signed comparison that is wrong — mixing signed and unsigned
                // orders over the same shared integer variable yields spurious
                // UNSAT (e.g. `(bvslt x #b0000) ∧ (bvult #b0100 x)` is SAT with
                // x = 9, but the unsigned arith parse derives x < 0 ∧ x > 4).
                // Signed BV comparisons therefore stay purely in the BV solver.
                let var = self.get_or_create_var(term);
                self.record_constraint(var, Constraint::Lt(*lhs, *rhs));
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::BvSle(lhs, rhs) => {
                // Bitvector *signed* less-than-or-equal.  As for `BvSlt`, the
                // recorded `Constraint::Le` is handled with correct signed
                // (two's-complement) semantics by the BV theory path.  We do NOT
                // create a `var_to_parsed_arith` entry, because the linear-arith
                // parse treats BV operands as unsigned non-negative integers and
                // would mix signed/unsigned orders in one integer space,
                // producing spurious UNSAT.  Signed BV comparisons stay purely
                // in the BV solver.
                let var = self.get_or_create_var(term);
                self.record_constraint(var, Constraint::Le(*lhs, *rhs));
                // Track theory variables for model extraction
                self.track_theory_vars(*lhs, manager);
                self.track_theory_vars(*rhs, manager);
                Lit::pos(var)
            }
            TermKind::Select(_, _) | TermKind::Store(_, _, _) => {
                // Array operations - theory terms
                self.has_array_ops = true;
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::Apply { .. } => {
                // Uninterpreted function application - theory term
                let var = self.get_or_create_var(term);
                // Register Bool-valued function applications as theory
                // constraints so that EUF congruence closure can detect
                // conflicts when the SAT solver assigns opposite polarities
                // to congruent applications (e.g., t(m)=true, t(co)=false,
                // but m=co implies t(m)=t(co)).
                if t.sort == manager.sorts.bool_sort {
                    self.record_constraint(var, Constraint::BoolApp(term));
                }
                Lit::pos(var)
            }
            // Quantifiers are *registered* with MBQI / E-matching by
            // `register_asserted_quantifiers`, which runs on the asserted spine
            // before this encoder does.  This pass is the Tseitin transform and
            // is polarity-blind by construction, so it must not decide which
            // quantifiers are facts: MBQI turns a registered universal into
            // ground unit clauses, and registering one that sits behind a
            // polarity boundary refuted satisfiable formulas such as
            // `(not (forall ((x Int)) (P x))) ∧ (not (P 5))`.
            TermKind::Forall { .. } | TermKind::Exists { .. } => {
                self.has_quantifiers = true;
                // Create a boolean variable for the quantifier
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // String operations - theory terms and predicates
            TermKind::StringLit(_)
            | TermKind::StrConcat(_, _)
            | TermKind::StrLen(_)
            | TermKind::StrSubstr(_, _, _)
            | TermKind::StrAt(_, _)
            | TermKind::StrReplace(_, _, _)
            | TermKind::StrReplaceAll(_, _, _)
            | TermKind::StrReplaceRe(_, _, _)
            | TermKind::StrReplaceReAll(_, _, _)
            | TermKind::StrToInt(_)
            | TermKind::IntToStr(_)
            | TermKind::StrToCode(_)
            | TermKind::StrFromCode(_)
            | TermKind::StrInRe(_, _) => {
                // String terms - theory solver handles these
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            TermKind::StrContains(_, _)
            | TermKind::StrPrefixOf(_, _)
            | TermKind::StrSuffixOf(_, _)
            | TermKind::StrLt(_, _)
            | TermKind::StrLe(_, _)
            | TermKind::StrIndexOf(_, _, _) => {
                // String predicates - theory atoms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point constants and special values
            TermKind::FpLit { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpNaN { .. } => {
                // FP constants - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point operations
            TermKind::FpAbs(_)
            | TermKind::FpNeg(_)
            | TermKind::FpSqrt(_, _)
            | TermKind::FpRoundToIntegral(_, _)
            | TermKind::FpAdd(_, _, _)
            | TermKind::FpSub(_, _, _)
            | TermKind::FpMul(_, _, _)
            | TermKind::FpDiv(_, _, _)
            | TermKind::FpRem(_, _)
            | TermKind::FpMin(_, _)
            | TermKind::FpMax(_, _)
            | TermKind::FpFma(_, _, _, _) => {
                // FP operations - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point predicates
            TermKind::FpLeq(_, _)
            | TermKind::FpLt(_, _)
            | TermKind::FpGeq(_, _)
            | TermKind::FpGt(_, _)
            | TermKind::FpEq(_, _)
            | TermKind::FpIsNormal(_)
            | TermKind::FpIsSubnormal(_)
            | TermKind::FpIsZero(_)
            | TermKind::FpIsInfinite(_)
            | TermKind::FpIsNaN(_)
            | TermKind::FpIsNegative(_)
            | TermKind::FpIsPositive(_) => {
                // FP predicates - theory atoms that return bool
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Floating-point conversions
            TermKind::FpToFp { .. }
            | TermKind::FpToSBV { .. }
            | TermKind::FpToUBV { .. }
            | TermKind::FpToReal(_)
            | TermKind::RealToFp { .. }
            | TermKind::SBVToFp { .. }
            | TermKind::UBVToFp { .. } => {
                // FP conversions - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Datatype operations
            TermKind::DtConstructor { .. }
            | TermKind::DtTester { .. }
            | TermKind::DtSelector { .. } => {
                // Datatype operations - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
            // Match expressions on datatypes
            TermKind::Match { .. } => {
                // Match expressions - theory terms
                let var = self.get_or_create_var(term);
                Lit::pos(var)
            }
        }
    }

    /// Attach the trichotomy `(a = b) OR (a < b) OR (a > b)` to a numeric `Eq`
    /// **atom** that [`Solver::encode_depth`] has just given the SAT variable
    /// `eq_var`.
    ///
    /// This is the **single owner** of arithmetic-disequality enforcement.  It
    /// replaced four overlapping syntactic walks — `add_arith_diseq_split`,
    /// `add_arith_trichotomy_clause`, `add_arith_eq_trichotomy` and the never
    /// called `add_arith_diseq_splits_for_sat_model` — which were deleted with
    /// it, not merely demoted.  See the call site in the `TermKind::Eq` arm for
    /// the full rationale; in short:
    ///
    /// * `ArithSolver` has no `assert_neq` and cannot have a useful one — `a
    ///   != b` is not convex, so it has no representation as tableau bounds.
    ///   The standard CDCL(T) answer is to make the case split explicit and
    ///   let the SAT core choose the disjunct, which is exactly this clause.
    /// * Because it is a *tautology* over a total order it is sound in every
    ///   Boolean context, so it can be emitted from the atom itself without
    ///   knowing anything about the polarity the atom will be used under.  It
    ///   adds no constraint; it only ensures that when the core does force
    ///   `(a = b)` false, some strict atom becomes unit and reaches the
    ///   tableau through the ordinary `Constraint::Lt`/`Constraint::Gt` path.
    ///
    /// # Incrementality
    ///
    /// The three clauses go into the SAT core at the current scope, so
    /// `Solver::pop` retracts them with `sat.pop()` like every other clause
    /// this arm emits.  Emission is guarded by
    /// [`Solver::numeric_trichotomy_atoms`], journalled as
    /// [`TrailOp::NumericTrichotomyAdded`], so the mark is dropped by the same
    /// `pop` that drops the clause and the next encode re-emits both together.
    ///
    /// The guard is *required*, not an optimisation.  The Tseitin memo cannot
    /// serve as one: it is retracted per entry on `pop` and deliberately
    /// bypassed when a term is re-encoded under a widened polarity, so this arm
    /// runs again for an atom whose trichotomy is already in the database.
    /// Without the ledger that appended a literal-identical clause over
    /// identical variables — which `oxiz_sat::Solver::add_clause` does not
    /// deduplicate — once per `(push)(pop)` pair, growing without bound.
    /// `scope_rebase_tests::a_no_op_push_pop_between_checks_does_not_re_encode_
    /// the_goal` caught exactly that.
    ///
    /// A ledger *here* is safe in a way one beside the tableau would not be:
    /// it is keyed by term and scoped to the assertion stack, the same stack
    /// `sat.pop()` unwinds.  A lazily-recorded disequality set living next to
    /// the arithmetic solver would instead have to stay in step with the
    /// theory solvers' scope stack, which tracks CDCL *decision levels* rather
    /// than assertion scopes (see `Solver::pop`) — and that mismatch is the
    /// class of bug this whole change exists to remove.
    ///
    /// Skipped when `mk_eq` folds the pair to a constant (syntactically equal
    /// operands): the clause would be trivially satisfied and carry nothing.
    fn add_numeric_trichotomy(
        &mut self,
        eq_term: TermId,
        eq_var: Var,
        lhs: TermId,
        rhs: TermId,
        manager: &mut TermManager,
        depth: u32,
    ) {
        // A folded `Eq` never reaches this arm (the encoder dispatches on
        // `TermKind::Eq`), but `mk_lt`/`mk_gt` below would still build atoms
        // for an identical pair, and `(a = a) OR (a < a) OR (a > a)` carries
        // nothing.
        if lhs == rhs {
            return;
        }
        if !self.numeric_trichotomy_atoms.insert(eq_term) {
            return;
        }
        self.trail
            .push(TrailOp::NumericTrichotomyAdded { term: eq_term });
        let lt_term = manager.mk_lt(lhs, rhs);
        let gt_term = manager.mk_gt(lhs, rhs);
        // Encode through `encode_depth` (not `encode`) so the two strict atoms
        // inherit this atom's depth budget rather than restarting at zero:
        // `encode_depth_exceeded` must stay honest on a deep instantiation.
        let lt_lit = self.encode_depth(lt_term, manager, depth + 1);
        let gt_lit = self.encode_depth(gt_term, manager, depth + 1);
        self.sat.add_clause([Lit::pos(eq_var), lt_lit, gt_lit]);
    }
}
