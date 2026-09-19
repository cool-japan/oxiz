//! The core CDCL(T)/MBQI search loop, split out of `mod.rs` once it grew
//! past what that already-large file (module declarations, the `Solver`
//! struct, construction, and every other public entry point) had room for
//! under the workspace's 2000-line-per-file ceiling.
//!
//! `pub(super)` rather than private: `Solver::check_with_arith_refinement`
//! (still in `mod.rs`) is `check_core`'s only caller.

use super::array_refinement::ArrayRefinementStep;
use super::*;

impl Solver {
    /// Get a SAT variable for a term, then check satisfiability
    pub(super) fn check_core(&mut self, manager: &mut TermManager) -> SolverResult {
        // Per-search case-split round budget: each CDCL(T) search gets its
        // own allowance to spend on non-convex-LIA refinement rounds. Unlike
        // `case_split_terms` (trail-scoped: see the field doc) this resets on
        // every search, not only on `pop`.
        self.case_split_rounds = 0;
        // Per-search, exactly like the round counter above: each `check` gets
        // its own chance to run the refinement, so a previous search's skipped
        // round must not condemn this one's verdict.
        self.case_split_skipped_targets = false;
        // Check for trivial unsat (false assertion)
        if self.has_false_assertion {
            self.build_unsat_core_trivial_false();
            return SolverResult::Unsat;
        }

        if self.assertions.is_empty() {
            return SolverResult::Sat;
        }

        // Honesty gate (soundness): if the Tseitin encoder refused a
        // sub-formula because it was pathologically deep, the encoding is
        // incomplete and any model built over it is untrustworthy.
        //
        // This gate is deliberately the *first* thing after the two trivial
        // verdicts above.  `assert` sets the flag by skipping the deep term
        // entirely (see `encode.rs`), and every stage between here and the
        // CDCL(T) loop — the axiom instantiators, the five early-conflict
        // collectors, and the nonlinear/FP/string model attempts — walks those
        // same assertion terms.  Several of those walks recurse natively, so
        // running any of them on a term already known to exceed the encoder's
        // safe depth crashes the process instead of reaching this answer: a
        // flat `(str.++ x1 … x5000)` aborted here via
        // `check_string_constraints` -> `eval_ground_bool`, on a 1 MiB stack,
        // long before the gate was consulted.
        //
        // Cost of the earlier position: one of those collectors could have
        // refuted the assertion set outright, and an `Unsat` derived from a
        // partial encoding is still sound.  That precision is given up
        // knowingly — it only applies to inputs that carry an assertion deeper
        // than `ENCODE_DEPTH_LIMIT`, which the gate was already going to
        // answer `Unknown` for unless a collector happened to refute them
        // first, and "answers `Unknown`" beats "aborts the process".
        if self.encode_depth_exceeded {
            return SolverResult::Unknown;
        }

        // Supply the defining axioms of every internalised `div` / `mod` /
        // numeric-`ite` term before any stage inspects the arithmetic atoms:
        // without them those terms are free variables and both the honesty gate
        // and the CDCL(T) loop below would reason about a formula that has lost
        // the terms' semantics.
        self.instantiate_arith_axioms(manager);

        // Supply the defining axioms of every datatype term as well.  Without
        // them a selector, a tester and a constructor application are three
        // unrelated free symbols to the CDCL(T) core, and even
        // `(= (head l) 10) ∧ (= (head l) 11)` came back `sat`.
        self.instantiate_dt_axioms(manager);

        // Check string constraints for early conflict detection
        if self.check_string_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check floating-point constraints for early conflict detection
        if self.check_fp_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check datatype constraints for early conflict detection
        if self.check_dt_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check array constraints for early conflict detection
        if self.check_array_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Check bitvector constraints for early conflict detection
        if self.check_bv_constraints(manager) {
            return SolverResult::Unsat;
        }

        // For NIA/NRA logics: dispatch all assertions to the full polynomial
        // solver first (NiaSolver or NlsatSolver). This gives a definitive
        // SAT/UNSAT for most benchmark problems without the CDCL(T) loop.
        if let Some(nl_result) = self.dispatch_nl_solver(manager) {
            match nl_result {
                SolverResult::Sat => return SolverResult::Sat,
                SolverResult::Unsat => return SolverResult::Unsat,
                SolverResult::Unknown => {}
            }
        }

        // Check nonlinear arithmetic constraints for early conflict detection
        // (static pattern matching, complementary to the dispatch above).
        if self.check_nonlinear_constraints(manager) {
            return SolverResult::Unsat;
        }

        // Positive FP path (completeness without sacrificing soundness): before
        // conceding `Unknown` on FP atoms below, try to construct and *verify* a
        // concrete floating-point model.  `try_fp_model_sat` pins every FP-sorted
        // term to a bit-exact IEEE-754 value and only reports success when every
        // assertion evaluates to `true` under it, so the resulting `Sat` is a
        // genuine model witness rather than a guess.
        if self.fp_atoms_need_theory(manager) && self.try_fp_model_sat(manager) {
            return SolverResult::Sat;
        }

        // Honesty gate (soundness): there is no complete String / FP theory
        // wired into the CDCL(T) core — `encode.rs` maps string and FP atoms to
        // fresh SAT variables, and the checks above only detect a fixed set of
        // definite conflicts.  If any such atom survives without a proven
        // conflict, we must answer `Unknown` instead of letting the SAT core
        // treat it as a free Boolean, which would report a spurious `Sat` for
        // formulas like `(= s "abc") ∧ (str.contains s "xyz")` or
        // `fp.lt x y ∧ fp.lt y x`.
        if self.string_atoms_need_theory(manager) {
            // Before conceding, try to construct and verify a concrete string
            // model. A verified witness is a sound `Sat` certificate; otherwise
            // keep the honest `Unknown`.
            if self.ground_string_model_sat(manager) {
                return SolverResult::Sat;
            }
            return SolverResult::Unknown;
        }
        if self.fp_atoms_need_theory(manager) {
            return SolverResult::Unknown;
        }

        // Honesty gate (soundness): an arithmetic comparison / equality atom that
        // could not be turned into a linear constraint (it contains Div/Mod, a
        // nonlinear product, or an out-of-range constant) has no theory
        // constraint attached — `encode.rs` left it as a free Boolean.  Trusting
        // the SAT layer to guess a truth value for such an atom yields a
        // spurious Sat/Unsat.  If the nonlinear dispatch above could not decide
        // the problem and such an atom survives, answer `Unknown`.
        if self.arith_atoms_need_theory(manager) {
            return SolverResult::Unknown;
        }

        // The array refinement's deterministic work counters describe *this*
        // check, not the script's history: a second `(check-sat)` gets the
        // same budget as the first.  See [`Statistics::array_refinement_rounds`].
        self.statistics.array_refinement_rounds = 0;
        self.statistics.array_lemma_instances = 0;
        self.statistics.bv_embedded_checks = 0;

        // Check resource limits before starting
        if self.config.max_conflicts > 0 && self.statistics.conflicts >= self.config.max_conflicts {
            return SolverResult::Unknown;
        }
        if self.config.max_decisions > 0 && self.statistics.decisions >= self.config.max_decisions {
            return SolverResult::Unknown;
        }

        // Wall-clock deadline for the whole check.  `timeout_ms == 0` means "no
        // timeout".  Computed exactly once, here, and then handed to every
        // consumer below: the MBQI round boundary in the search loop, every
        // `TheoryManager` this function builds (including the ones it rebuilds
        // for a refinement round -- deriving it inside `TheoryManager::new`
        // used to restart the clock on each, granting a script with `N` rounds
        // `N * timeout_ms`), the outer SAT solver, and the bit-vector solver's
        // embedded one.
        //
        // Placed here, *above* the pure-equality fast path, for two reasons:
        // that path runs its own `self.sat.solve()` and must be budgeted like
        // every other solve on this engine (a budget-induced `Unknown` there
        // costs a wasted probe and falls through -- `eq_skeleton`'s `Unknown`
        // arm returns `None`, never a verdict), and because the ceilings below
        // are relative to the engine's *current* counters, leaving them
        // un-refreshed until after that solve would let the previous check's
        // exhausted ceiling silently abort it.
        //
        // Not `cfg`-gated on `std`: `oxiz_time::Instant` exists on every
        // target, and on one whose clock is frozen
        // (`wasm32-unknown-unknown` / `no_std`) `now()` is a constant t = 0, so
        // no deadline built from it can ever be reached and `:timeout` is the
        // documented no-op `oxiz_time`'s crate docs describe.
        let deadline: Option<oxiz_time::Instant> = if self.config.timeout_ms > 0 {
            oxiz_time::Instant::now()
                .checked_add(core::time::Duration::from_millis(self.config.timeout_ms))
        } else {
            None
        };

        // Arm the two SAT engines this check will run (finding U-Z12).
        //
        // Until this existed, `(set-option :max-conflicts N)` bounded only
        // *theory* conflicts (`Statistics::conflicts`, incremented solely next
        // to a `theory_conflicts += 1`), `(set-option :max-decisions N)` was
        // wired to nothing at all, and `(set-option :timeout N)` was polled
        // only between MBQI rounds and at the entry of a theory callback -- all
        // of them outside the one place a bit-blasted goal actually spends its
        // time, `BvSolver::check`'s embedded `solve()`.
        //
        // After this, `N` is *three independent budgets of `N`*, one per kind
        // of work, all re-armed once per check:
        //   * outer Boolean conflicts / decisions -- `oxiz_sat::SolverStats`
        //     on `self.sat`, bounded here;
        //   * embedded bit-blasting conflicts -- the total across every probe
        //     and repair round of this check, bounded by `BvSolver`'s own
        //     `conflicts_spent` accumulator;
        //   * theory conflicts -- `Statistics::conflicts`, unchanged, still
        //     compared inside `TheoryManager`.
        // They are separate counters because they count different work; a
        // single shared counter would need the embedded solver to report into
        // the outer `Statistics`.  `:timeout`, by contrast, is one wall-clock
        // deadline shared by all of them.
        //
        // The ceilings are relative to the counters' current values because
        // `oxiz_sat::SolverStats` is cumulative across `solve()` calls and the
        // outer solver is never reset between checks (only backtracked to the
        // root): a raw `Some(N)` would give the second `(check-sat)` of a
        // script whatever the first left over, and eventually nothing.
        let conflict_budget = (self.config.max_conflicts > 0).then_some(self.config.max_conflicts);
        let decision_budget = (self.config.max_decisions > 0).then_some(self.config.max_decisions);
        // Read into locals first: `stats()` borrows `self.sat` immutably while
        // the setters borrow it mutably.
        let conflicts_so_far = self.sat.stats().conflicts;
        let decisions_so_far = self.sat.stats().decisions;
        self.sat
            .set_max_conflicts(conflict_budget.map(|n| conflicts_so_far.saturating_add(n)));
        self.sat
            .set_max_decisions(decision_budget.map(|n| decisions_so_far.saturating_add(n)));
        self.sat.set_deadline(deadline);
        // `set_budget` re-arms the bit-vector solver's total allowance, so the
        // budget's period is one call to this function.
        self.bv.set_budget(conflict_budget, deadline);

        // Pure Equality Logic fast path: static transitivity clauses (see
        // `eq_skeleton`'s module doc) make plain SAT a complete decision
        // procedure for a formula built only from Boolean connectives over
        // equalities between uninterpreted-sort constants, so a disjunctive
        // equality chain that would make CDCL(T)'s theory round-trips blow up
        // exponentially is instead decided by the SAT core alone. Every gate
        // above this point already ruled out `False`/empty/too-deep/nonlinear
        // inputs, none of which this narrower check needs to repeat: it
        // self-gates independently by walking the assertions and declining
        // (returning `None`, changing nothing) at the first construct outside
        // its grammar. A confirmed verdict returns immediately; anything else
        // — impure input, or a `Sat` this module's own re-verification could
        // not confirm — falls through to the ordinary search below unchanged.
        if let Some(verdict) = self.try_pure_equality_fast_path(manager) {
            self.debug_check_invariants("check_core: after pure-equality fast path");
            return verdict;
        }

        // Seam 1 of 2: rebuild all three incremental theory solvers from the
        // live assertion set before this check starts searching.
        //
        // The previous `check` on this solver ended either `Sat` — in which case
        // it never backtracked and left the theory solvers several decision
        // scopes deep, holding that check's branch facts — or `Unsat`, which
        // returns from `solve_with_theory` without unwinding either.  Nothing
        // between two `check` calls pops those scopes: `Solver::pop` is the only
        // other place that clears them, and a script need never call it.  An
        // interposed `(check-sat)` could therefore change the answer of the next
        // one, which is exactly what `tests/scope_leak_hazard.rs` demonstrates.
        //
        // See `rebase_theory_state` for why this is a reset-and-replay rather
        // than a scope unwind, and for the BV solver's own (older) reason to be
        // reset here: its base-level unit facts are not wired into
        // `Solver::push` / `pop` and would leak across a user scope as well.
        self.rebase_theory_state();

        // Run SAT solver with theory integration
        let mut theory_manager = TheoryManager::new(
            manager,
            &mut self.euf,
            &mut self.arith,
            &mut self.bv,
            &self.bv_terms,
            &self.var_to_constraint,
            &self.var_to_parsed_arith,
            &self.term_to_var,
            &self.var_to_term,
            &mut self.derived_reasons,
            self.config.theory_mode,
            &mut self.statistics,
            self.config.max_conflicts,
            self.config.max_decisions,
            self.has_bv_arith_ops,
            self.has_quantifiers,
            &self.quantifier_uf_funcs,
            deadline,
        );

        // MBQI loop for quantified formulas
        let max_mbqi_iterations = 100;
        let mut mbqi_iteration = 0;

        // Lazy array-axiom refinement rounds (see `array_refinement`); the
        // round cap is [`MAX_ARRAY_REFINEMENT_ROUNDS`].
        let mut array_refinement_rounds = 0usize;
        // Deterministic budget for the *re-solves* the array refinement
        // triggers (`#P2b-38` strand (c)), armed when the first array lemma is
        // asserted: the value of `SolverStats::conflicts` past which this check
        // answers `Unknown`.  See [`ARRAY_REFINEMENT_RESOLVE_CONFLICTS`].
        let mut array_resolve_conflict_ceiling: Option<u64> = None;

        // How much SAT work this check had done before the search started.
        // The two repair gates below (`case_split_affordable`,
        // `blocking_affordable`) are relative to it, so that "the first solve
        // was cheap" is a statement about propagations performed and not about
        // seconds elapsed — see [`REFINEMENT_WORK_CEILING_PROPAGATIONS`].
        let propagations_at_entry = self.sat.stats().propagations;

        loop {
            // Enforce the wall-clock timeout between MBQI rounds.  Mid-`solve`
            // enforcement lives in the theory callbacks (see TheoryManager) and,
            // since U-Z12, inside both SAT engines themselves.
            if let Some(d) = deadline {
                if oxiz_time::Instant::now() >= d {
                    return SolverResult::Unknown;
                }
            }
            // The array refinement's own budget, in conflicts rather than in
            // seconds (decision (9)).  Checked here, at the same round
            // boundary the wall-clock version used, so an exhausted budget
            // answers `Unknown` and never a verdict; the model goes with it for
            // the reason the round-budget exit below gives.
            if let Some(ceiling) = array_resolve_conflict_ceiling {
                // Two counters, because a refinement loop can run away in two
                // different ways and a single one does not see both.
                //
                // * Conflicts bound a loop that *searches*: the re-solves
                //   branch and backtrack, and the conflict count climbs.
                // * Lemma instances bound a loop that only *builds*.  The
                //   const-array / chain-index families can enlarge the circuit
                //   round after round while the search itself stays
                //   conflict-free — `rc3/slow/m5.smt2` (six declarations, three
                //   assertions) ran 400 s with no answer and never accrued the
                //   50,000 conflicts the ceiling asked for, because the work
                //   was all in interning and re-solving a growing circuit, not
                //   in conflict analysis.  Only an explicit `:timeout` stopped
                //   it, which is exactly the machine-dependence decision (9)
                //   removes.
                //
                // Both are monotone counts of work performed, so they are
                // identical on an idle and on a loaded machine.
                if self.sat.stats().conflicts >= ceiling {
                    self.model = None;
                    self.unsat_core = None;
                    return SolverResult::Unknown;
                }
            }
            let sat_result = self.sat.solve_with_theory(&mut theory_manager);
            // If a genuine theory conflict was suppressed because the conflict
            // limit was hit, the theory manager reported `Sat` to the SAT solver
            // to force it to stop searching.  That `Sat` is a resource-exhaustion
            // signal, NOT a proof of satisfiability: the model on the table may
            // violate a theory constraint whose conflict we refused to report.
            // We must answer `Unknown` rather than trust such a `Sat`.
            //
            // `unjustified_conflict` is the same shape for a different cause:
            // a theory refuted the assignment but the manager could not build a
            // clause for it (no reason literal could be blamed), so the conflict
            // was aborted instead of being emitted as the empty clause.  A
            // dropped conflict never justifies `Sat` either.
            let resource_exhausted =
                theory_manager.resource_exhausted() || theory_manager.unjustified_conflict();
            match sat_result {
                SatResult::Unsat => {
                    // Soundness gate: a model-blocking clause (see
                    // `model_blocking`) is a restriction of the search space,
                    // not a consequence of the assertions — the gate that
                    // triggered it fires on the *evaluator's* width limit as
                    // well as on a genuine violation.  "No model outside the
                    // excluded region" is therefore not `unsat`, and there is
                    // no core to hand over: the resolution proof rests on
                    // clauses no assertion entails, so `unsat_core` is cleared
                    // rather than built.
                    //
                    // Accepted cost: a *genuine* refutation found after
                    // blocking started — including the empty clause MBQI adds
                    // for a definitively-false instantiation — also surfaces as
                    // `Unknown`.  That is precision, not soundness, and the
                    // verdict lattice stays `Unknown -> {Sat, Unknown}`.
                    if self.blocking_clauses_present() {
                        self.model = None;
                        self.unsat_core = None;
                        return SolverResult::Unknown;
                    }
                    self.build_unsat_core();
                    // After a theory/Boolean conflict has been turned into an
                    // unsat core: the core must name assertions that still
                    // exist in this context (see `check_unsat_core`).
                    self.debug_check_invariants("check_core: after unsat-core construction");
                    return SolverResult::Unsat;
                }
                SatResult::Unknown => {
                    return SolverResult::Unknown;
                }
                SatResult::Sat => {
                    if resource_exhausted {
                        // A real theory conflict was dropped at the conflict
                        // limit; never fabricate Sat over a suppressed conflict.
                        self.unsat_core = None;
                        return SolverResult::Unknown;
                    }
                    // If no quantifiers, we're done
                    if !self.has_quantifiers {
                        self.build_model(manager);
                        // ORDER (issue #40): the two repair paths below run
                        // *before* the `model_refutes_assertions` gate, which
                        // used to sit right here and bail out.
                        //
                        // Both repairs exist precisely to fix a candidate model
                        // that does not hold up — the case-split lemmas force
                        // the search to branch on a value the LP was free to
                        // collide, the array lemmas retract a `select` the
                        // candidate got wrong — so bailing out first made them
                        // unreachable for the very models they were written
                        // for.  Every gate they sit behind is unchanged; only
                        // the order is.
                        //
                        // `self.model` therefore stays *set* through both:
                        // `instantiate_array_axioms` reads it to decide which
                        // axiom instances the candidate already satisfies, and
                        // a `None` there is read as "nothing is satisfied",
                        // degenerating the round into eager instantiation of
                        // every candidate instance — up to 256 rounds of that.
                        // Clearing moved to the final `Unknown` exit below.
                        //
                        // Nothing observable from outside distinguishes the two
                        // orders, so the test build records which one is live
                        // (see the field doc).
                        #[cfg(test)]
                        self.repair_paths_saw_model.push(self.model.is_some());

                        // Non-convex LIA refinement: a numeric UF-argument
                        // term pinned to a small finite domain by
                        // arithmetic bounds is invisible to Nelson-Oppen
                        // equality sharing (no single value is entailed), so
                        // the CDCL(T) core has no atom to branch its value on
                        // and a genuine `unsat` can come back a spurious
                        // `sat`. Emit an explicit `(or (= t v0) ...)` lemma
                        // for each such term and re-solve. Gated on the search
                        // so far having been cheap, since the refinement
                        // re-solves the whole problem from scratch — see
                        // [`REFINEMENT_WORK_CEILING_PROPAGATIONS`], the
                        // deterministic ceiling that replaced the wall-clock
                        // one this gate used to read (decision (9)).
                        let case_split_affordable = self
                            .sat
                            .stats()
                            .propagations
                            .saturating_sub(propagations_at_entry)
                            < REFINEMENT_WORK_CEILING_PROPAGATIONS;
                        // The affordability test must not simply short-circuit
                        // the call away: `split_narrow_int_domains` is what
                        // discovers whether this candidate has unbranched
                        // shared-term domains, and it records that in
                        // `case_split_skipped_targets` for the honesty gate in
                        // `check`. With a plain `affordable && split(..)` the
                        // gate never heard about a candidate whose refinement
                        // the ceiling declined, so the verdict silently changed
                        // with the ceiling.
                        //
                        // When the round is unaffordable the targets are still
                        // *counted* (marking the `Sat` unverified) but no lemma
                        // is asserted and no re-solve is owed.
                        if !case_split_affordable {
                            self.note_unaffordable_case_split();
                        }
                        if case_split_affordable && self.split_narrow_int_domains(manager) {
                            // Re-solve with the freshly asserted case-split
                            // lemmas from a clean state, exactly as the
                            // array-lemma path below does: `add_clause` left
                            // the SAT core at the candidate model's trail,
                            // and the incremental theory solvers still hold
                            // that model's facts (only level-scoped `pop` is
                            // available, no surgical undo), so rebase to root
                            // before re-driving them from a fresh
                            // `TheoryManager`.
                            self.rebase_theory_state();
                            theory_manager = TheoryManager::new(
                                manager,
                                &mut self.euf,
                                &mut self.arith,
                                &mut self.bv,
                                &self.bv_terms,
                                &self.var_to_constraint,
                                &self.var_to_parsed_arith,
                                &self.term_to_var,
                                &self.var_to_term,
                                &mut self.derived_reasons,
                                self.config.theory_mode,
                                &mut self.statistics,
                                self.config.max_conflicts,
                                self.config.max_decisions,
                                self.has_bv_arith_ops,
                                self.has_quantifiers,
                                &self.quantifier_uf_funcs,
                                deadline,
                            );
                            continue;
                        }
                        // Lazy array-axiom instantiation: the syntactic array
                        // pre-checks and EUF congruence do not implement a
                        // complete array decision procedure, so a candidate `Sat`
                        // may violate read-over-write / extensionality.  Watch the
                        // array terms in this candidate model and assert every
                        // axiom instance it does not already satisfy as a lemma,
                        // then re-solve.  Only genuine array models survive.
                        // Lazy array-axiom refinement (see
                        // `solver::array_refinement`).  One round asserts every
                        // axiom instance this candidate model violates and
                        // prepares a fresh search; rebuilding the theory
                        // manager and looping is this call site's job, because
                        // the manager holds `&mut` borrows the round cannot
                        // exist across.
                        match self.array_refinement_round(
                            manager,
                            &mut array_refinement_rounds,
                            &mut array_resolve_conflict_ceiling,
                            conflict_budget,
                            conflicts_so_far,
                            deadline,
                        ) {
                            ArrayRefinementStep::OutOfBudget => {
                                return SolverResult::Unknown;
                            }
                            ArrayRefinementStep::Resolve => {
                                theory_manager = TheoryManager::new(
                                    manager,
                                    &mut self.euf,
                                    &mut self.arith,
                                    &mut self.bv,
                                    &self.bv_terms,
                                    &self.var_to_constraint,
                                    &self.var_to_parsed_arith,
                                    &self.term_to_var,
                                    &self.var_to_term,
                                    &mut self.derived_reasons,
                                    self.config.theory_mode,
                                    &mut self.statistics,
                                    self.config.max_conflicts,
                                    self.config.max_decisions,
                                    self.has_bv_arith_ops,
                                    self.has_quantifiers,
                                    &self.quantifier_uf_funcs,
                                    deadline,
                                );
                                continue;
                            }
                            ArrayRefinementStep::NoLemma => {}
                        }
                        // Soundness gate: never return `Sat` for a model that
                        // provably violates an assertion (see
                        // `model_refutes_assertions`).  This backstops the SAT
                        // core: if it commits an inconsistent trail and reports a
                        // full assignment that falsifies a Boolean clause the
                        // theory layer cannot observe, we answer `Unknown`
                        // instead of a wrong `Sat`.
                        //
                        // Recomputed here, at the point of use, rather than
                        // hoisted above the two repairs: neither repair mutates
                        // `self.model`, but computing it where it is consumed
                        // is what makes "recompute after every re-solve" fall
                        // out of the loop structure instead of being a rule to
                        // remember.
                        if self.model_refutes_assertions(manager) {
                            // Bounded blocking (issue #40): this one assignment
                            // did not hold up, which says nothing about the
                            // *next* one.  Exclude it and re-solve rather than
                            // concede on the first unlucky candidate.  See
                            // `model_blocking` for why the resulting clause is
                            // a search restriction rather than a lemma, and for
                            // the `Unsat` downgrade that pays for it.
                            //
                            // Gated on the same deterministic ceiling the
                            // case-split refinement uses, and for the same
                            // reason: a round is a full re-solve from scratch,
                            // affordable only when the search so far was cheap.
                            let blocking_affordable = self
                                .sat
                                .stats()
                                .propagations
                                .saturating_sub(propagations_at_entry)
                                < REFINEMENT_WORK_CEILING_PROPAGATIONS;
                            if self.block_refuted_model_and_rebase(blocking_affordable) {
                                theory_manager = TheoryManager::new(
                                    manager,
                                    &mut self.euf,
                                    &mut self.arith,
                                    &mut self.bv,
                                    &self.bv_terms,
                                    &self.var_to_constraint,
                                    &self.var_to_parsed_arith,
                                    &self.term_to_var,
                                    &self.var_to_term,
                                    &mut self.derived_reasons,
                                    self.config.theory_mode,
                                    &mut self.statistics,
                                    self.config.max_conflicts,
                                    self.config.max_decisions,
                                    self.has_bv_arith_ops,
                                    self.has_quantifiers,
                                    &self.quantifier_uf_funcs,
                                    deadline,
                                );
                                continue;
                            }
                            // Nothing left to try: the budget is spent, the
                            // feature is off, or the assignment projects onto
                            // no mapped variable.  This is the exit that now
                            // owns clearing the model (see the ORDER note
                            // above).
                            self.model = None;
                            self.unsat_core = None;
                            return SolverResult::Unknown;
                        }
                        // The gate's other refusal (`#P2b-27`): an assertion
                        // it could not evaluate *because a Boolean variable
                        // has no model entry* is not "no opinion" — the
                        // published model prints a default for that
                        // variable, and may falsify the assertion with it.
                        // Nothing to block and re-solve here (the variable
                        // was never assigned, so the candidate would come
                        // back unchanged): the honest answer is `unknown`.
                        if self.model_leaves_a_boolean_undetermined(manager) {
                            self.model = None;
                            self.unsat_core = None;
                            return SolverResult::Unknown;
                        }
                        self.unsat_core = None;
                        self.debug_check_invariants("check_core: before returning sat");
                        return SolverResult::Sat;
                    }

                    // Build partial model for MBQI
                    self.build_model(manager);

                    // Lazy array-axiom refinement, on the *quantified* path.
                    //
                    // This call is the other half of the seam
                    // `solver::ground_instance` documents.  The round used to
                    // exist only in the `!self.has_quantifiers` branch above,
                    // so a script with one `forall` in it never ran a single
                    // array rule — not over the terms an MBQI instance
                    // grounds, and not even over the ones its own ground
                    // assertions spell out.  `(assert (= (select a #b1) #b0))`
                    // beside `(assert (forall ((i …)) (= (select (store a #b0
                    // #b1) i) #b1)))` is unsatisfiable and answered `sat`,
                    // with a `(get-value)` that contradicted itself.
                    //
                    // It runs *before* `certify_quantified_sat` and before all
                    // three of the loop's `Sat` exits, because a candidate that
                    // violates read-over-write is not a model and must not be
                    // offered to the certifier as one.  A candidate that
                    // satisfies every applicable instance costs one collection
                    // walk and returns `NoLemma`.
                    match self.array_refinement_round(
                        manager,
                        &mut array_refinement_rounds,
                        &mut array_resolve_conflict_ceiling,
                        conflict_budget,
                        conflicts_so_far,
                        deadline,
                    ) {
                        ArrayRefinementStep::OutOfBudget => {
                            return SolverResult::Unknown;
                        }
                        ArrayRefinementStep::Resolve => {
                            theory_manager = TheoryManager::new(
                                manager,
                                &mut self.euf,
                                &mut self.arith,
                                &mut self.bv,
                                &self.bv_terms,
                                &self.var_to_constraint,
                                &self.var_to_parsed_arith,
                                &self.term_to_var,
                                &self.var_to_term,
                                &mut self.derived_reasons,
                                self.config.theory_mode,
                                &mut self.statistics,
                                self.config.max_conflicts,
                                self.config.max_decisions,
                                self.has_bv_arith_ops,
                                self.has_quantifiers,
                                &self.quantifier_uf_funcs,
                                deadline,
                            );
                            continue;
                        }
                        ArrayRefinementStep::NoLemma => {}
                    }

                    // NOTE (soundness): each of the three `Sat` exits below is
                    // guarded by `quantified_model_refutes_ground_assertions`
                    // — the quantified counterpart of the ground branch's
                    // `model_refutes_assertions` gate above.  See that method
                    // for the wrong-`sat` it closes and for why it is narrower
                    // than the ground gate.
                    //
                    // The guard is repeated at each exit rather than hoisted
                    // to here, on purpose: the MBQI branches that `continue`
                    // the loop must NOT be gated.  A model that falsifies a
                    // ground assertion mid-loop is a candidate MBQI is still
                    // working on, and the instantiation lemmas it is about to
                    // add can drive the search to a different model — or to a
                    // genuine `Unsat`, which is a strictly better answer than
                    // the `Unknown` an early gate would have produced.  Only a
                    // model the solver is about to *report* needs verifying.
                    //
                    // The verdict is `Unknown` rather than a resumed search:
                    // at these exits the loop has reached its fixpoint, so
                    // nothing would change on a further round and re-solving
                    // would not terminate.

                    // Certified `sat`: for the fragments `mbqi::model_certify`
                    // covers, a *total* interpretation of every symbol can be
                    // constructed from this candidate model and checked
                    // against every assertion — quantified ones included, over
                    // their whole infinite domain.  When that check passes we
                    // hold a model in the ordinary semantic sense, so `sat`
                    // follows outright and MBQI has nothing left to add.  When
                    // it does not, nothing changes: the certifier declines and
                    // the instantiation loop below runs exactly as before.
                    if self.certify_quantified_sat(manager) {
                        if self.quantified_model_refutes_ground_assertions(manager) {
                            self.model = None;
                            self.unsat_core = None;
                            return SolverResult::Unknown;
                        }
                        self.unsat_core = None;
                        self.debug_check_invariants(
                            "check_core: before returning sat (certified model)",
                        );
                        return SolverResult::Sat;
                    }

                    // Run MBQI to check quantified formulas
                    let model_assignments = self
                        .model
                        .as_ref()
                        .map(|m| m.assignments().clone())
                        .unwrap_or_default();

                    let mbqi_result = self.mbqi.check_with_model(&model_assignments, manager);
                    match mbqi_result {
                        MBQIResult::NoQuantifiers => {
                            if self.quantified_model_refutes_ground_assertions(manager) {
                                self.model = None;
                                self.unsat_core = None;
                                return SolverResult::Unknown;
                            }
                            self.unsat_core = None;
                            self.debug_check_invariants(
                                "check_core: before returning sat (no quantifiers)",
                            );
                            return SolverResult::Sat;
                        }
                        MBQIResult::Satisfied => {
                            // All quantifiers satisfied by the current model.
                            if self.quantified_model_refutes_ground_assertions(manager) {
                                self.model = None;
                                self.unsat_core = None;
                                return SolverResult::Unknown;
                            }
                            self.unsat_core = None;
                            self.debug_check_invariants(
                                "check_core: before returning sat (mbqi fixpoint)",
                            );
                            return SolverResult::Sat;
                        }
                        MBQIResult::InstantiationLimit => {
                            // Too many instantiations - return unknown
                            return SolverResult::Unknown;
                        }
                        MBQIResult::Conflict {
                            quantifier: _,
                            reason,
                        } => {
                            // Turn the reason into a blocking clause — but only
                            // if *every* reason term names a literal.  Skipping
                            // the ones that do not would not weaken the clause,
                            // it would strengthen it into a claim the reason
                            // never made: that the surviving literals alone are
                            // contradictory.  When the reason cannot be
                            // expressed we add nothing and let the bounded MBQI
                            // loop run out, which costs a round rather than
                            // correctness.
                            let lits: Option<Vec<Lit>> = reason
                                .iter()
                                .map(|&t| self.term_to_var.get(&t).map(|&v| Lit::neg(v)))
                                .collect();
                            if let Some(lits) = lits
                                && !lits.is_empty()
                            {
                                self.sat.add_clause(lits);
                            }
                            // Continue loop
                        }
                        MBQIResult::NewInstantiations(instantiations) => {
                            // Collect ground sub-terms (especially Skolem
                            // applications) from instantiation results so they
                            // become MBQI candidates in subsequent rounds.
                            for inst in &instantiations {
                                self.collect_ground_candidates_from_term(inst.result, manager);
                            }

                            // Collect domain/disequality info for pigeonhole
                            let mut ph_domains: FxHashMap<TermId, (i64, i64)> =
                                FxHashMap::default();
                            let mut ph_diseqs: Vec<(TermId, TermId)> = Vec::new();

                            // Add instantiation lemmas
                            for inst in instantiations {
                                // If the instantiation result is definitively False
                                // (e.g., a nested Exists with no valid witness), add an
                                // empty clause to signal immediate UNSAT.
                                let is_false_result = manager
                                    .get(inst.result)
                                    .is_some_and(|t| matches!(t.kind, TermKind::False));
                                if is_false_result {
                                    self.sat.add_clause([] as [Lit; 0]);
                                    break;
                                }
                                // Scan for pigeonhole patterns (recurses into Implies)
                                // The seam: an MBQI instance is ground, so it
                                // gets the pre-passes an assertion gets and
                                // becomes a root of the next array-structure
                                // collection round.  See
                                // `solver::ground_instance` for the wrong `sat`
                                // that skipping this produced.
                                let prepared = self.prepare_ground_instance(inst.result, manager);
                                self.scan_for_pigeonhole(
                                    prepared,
                                    manager,
                                    &mut ph_domains,
                                    &mut ph_diseqs,
                                );
                                let lit = self.encode(prepared, manager);
                                let ok = self.sat.add_clause([lit]);
                                let _ = ok;
                                self.add_int_domain_clauses(prepared, manager);
                            }
                            // Add pigeonhole exclusion clauses
                            if !ph_diseqs.is_empty() && !ph_domains.is_empty() {
                                self.add_pigeonhole_exclusions_from(
                                    &ph_domains,
                                    &ph_diseqs,
                                    manager,
                                );
                            }

                            // E-matching phase: find additional instantiations via trigger patterns
                            let ematch_lemmas =
                                self.ematch_engine.match_round(manager).unwrap_or_default();
                            let mut new_clauses_added = 0usize;
                            let mut ematch_unsat = false;
                            for lemma in ematch_lemmas {
                                // Same seam as the MBQI instances above: an
                                // e-matching lemma is a ground instance too.
                                let lemma = self.prepare_ground_instance(lemma, manager);
                                let lit = self.encode(lemma, manager);
                                if self.sat.add_clause([lit]) {
                                    new_clauses_added += 1;
                                } else {
                                    ematch_unsat = true;
                                    break;
                                }
                            }
                            if ematch_unsat || new_clauses_added > 0 {
                                // SAT solver will process newly added clauses on next iteration
                            }
                            // Continue loop
                        }
                        MBQIResult::Unknown => {
                            // Some evaluations produced symbolic residuals.
                            // Generate blind instantiations (simplified) once
                            // to seed the solver with ground lemmas for array
                            // theory reasoning (pigeonhole, bounds, etc.).
                            if !self.mbqi.blind_tried() {
                                self.mbqi.mark_blind_tried();
                                // Clear dedup cache so that blind instantiations with
                                // corrected substitution results are not filtered out
                                // as duplicates of earlier (broken) engine results.
                                self.mbqi.clear_dedup_cache();
                                let blind = self.mbqi.generate_blind_instantiations(manager);
                                let mut ph_domains: FxHashMap<TermId, (i64, i64)> =
                                    FxHashMap::default();
                                let mut ph_diseqs: Vec<(TermId, TermId)> = Vec::new();
                                for inst in blind {
                                    let is_false = manager
                                        .get(inst.result)
                                        .is_some_and(|t| matches!(t.kind, TermKind::False));
                                    if is_false {
                                        self.sat.add_clause([] as [Lit; 0]);
                                        break;
                                    }
                                    // Track domains and disequalities for pigeonhole
                                    let _ = manager.get(inst.result);
                                    // Same seam as the model-driven instances:
                                    // a blind instantiation is ground too.
                                    let prepared =
                                        self.prepare_ground_instance(inst.result, manager);
                                    self.scan_for_pigeonhole(
                                        prepared,
                                        manager,
                                        &mut ph_domains,
                                        &mut ph_diseqs,
                                    );
                                    let lit = self.encode(prepared, manager);
                                    let _ = self.sat.add_clause([lit]);
                                    self.add_int_domain_clauses(prepared, manager);
                                }
                                // Add pigeonhole exclusion clauses directly
                                // from the collected domains and disequalities.
                                self.add_pigeonhole_exclusions_from(
                                    &ph_domains,
                                    &ph_diseqs,
                                    manager,
                                );
                            }
                            // After 2 Unknown rounds, try finite instantiation:
                            // for quantifiers with bounded integer guards like
                            // (i >= 0 && i <= 3), enumerate all values and add
                            // ground instances directly.
                            if mbqi_iteration == 2 {
                                let finite_insts =
                                    self.mbqi.generate_finite_domain_instantiations(manager);
                                if !finite_insts.is_empty() {
                                    let mut ph_d: FxHashMap<TermId, (i64, i64)> =
                                        FxHashMap::default();
                                    let mut ph_q: Vec<(TermId, TermId)> = Vec::new();
                                    for inst in &finite_insts {
                                        let simplified =
                                            self.mbqi.deep_simplify(inst.result, manager);
                                        // Skip tautologies
                                        if manager
                                            .get(simplified)
                                            .is_some_and(|t| matches!(t.kind, TermKind::True))
                                        {
                                            continue;
                                        }
                                        // Same seam as the other three
                                        // instantiation paths: a finite-domain
                                        // instance is ground too.
                                        let simplified =
                                            self.prepare_ground_instance(simplified, manager);
                                        self.scan_for_pigeonhole(
                                            simplified, manager, &mut ph_d, &mut ph_q,
                                        );
                                        let lit = self.encode(simplified, manager);
                                        let _ = self.sat.add_clause([lit]);
                                        self.add_int_domain_clauses(simplified, manager);
                                    }
                                    if !ph_q.is_empty() && !ph_d.is_empty() {
                                        self.add_pigeonhole_exclusions_from(&ph_d, &ph_q, manager);
                                    }
                                }
                            }
                            if mbqi_iteration >= 10 {
                                // After exhausting blind and finite domain
                                // instantiation attempts, MBQI still could not
                                // *verify* that the candidate model satisfies
                                // every quantifier (each round returned
                                // `Unknown`, i.e. symbolic residuals remained).
                                //
                                // Blindly returning Sat here would be unsound:
                                // any UNSAT quantified formula whose refutation
                                // needs an instantiation outside the enumerated
                                // candidates would be wrongly declared
                                // satisfiable.  Z3 returns `unknown` in exactly
                                // this situation.
                                //
                                // We may still soundly answer Sat in one case:
                                // when every quantifier is *trivially valid* —
                                // its body simplifies to `True` in every model
                                // (e.g. `forall x. f(x) = f(x)`).  Such
                                // quantifiers add no constraint, so the model the
                                // SAT/theory layer already found satisfies the
                                // whole formula.  Otherwise the honest answer is
                                // Unknown — never fabricate Sat for an unverified
                                // quantifier.
                                self.unsat_core = None;
                                if self.quantifiers_trivially_valid(manager) {
                                    self.build_model(manager);
                                    // Same ground-model gate as the other
                                    // quantified `Sat` exits: "every quantifier
                                    // is vacuous" says nothing about the ground
                                    // assertions, and it is the ground part
                                    // that carries the wrong-`sat` this gate
                                    // closes.  Note the `build_model` above —
                                    // this exit rebuilds the model rather than
                                    // reusing the one made earlier in the
                                    // round, so the gate must run *after* it.
                                    if self.quantified_model_refutes_ground_assertions(manager) {
                                        self.model = None;
                                        return SolverResult::Unknown;
                                    }
                                    self.debug_check_invariants(
                                        "check_core: before returning sat (trivially valid)",
                                    );
                                    return SolverResult::Sat;
                                }
                                return SolverResult::Unknown;
                            }
                            // Continue MBQI loop
                        }
                    }

                    mbqi_iteration += 1;
                    #[cfg(test)]
                    {
                        self.mbqi_round_clauses.push(self.sat.num_clauses());
                    }
                    if mbqi_iteration >= max_mbqi_iterations {
                        return SolverResult::Unknown;
                    }

                    // MBQI round boundary: this round encoded fresh
                    // instantiation / e-matching lemmas through `encode`, each
                    // of which may allocate SAT variables and extend the
                    // Tseitin memo.  Check before the next round consumes them.
                    self.debug_check_invariants("check_core: mbqi round boundary");

                    // Seam 2 of 2: rebuild the theory solvers before the next
                    // round searches.
                    //
                    // The round that just finished ended `Sat`, so it never
                    // backtracked and left one theory scope open per decision it
                    // took, holding that branch's facts.  The lemmas encoded
                    // just above exist to *retract* that very branch, and the
                    // fresh manager below would assert their consequences on top
                    // of the facts they contradict — in scopes it cannot reach,
                    // because it numbers its own `level_stack` from zero.  That
                    // is the task-#26 false `unsat`.
                    //
                    // What the next round is entitled to survives untouched: the
                    // ground assertions and every kept instantiation / e-matching
                    // lemma live in the SAT clause database with their unit
                    // consequences committed at the root, and the replay below
                    // re-derives the theory state from exactly those.  Nothing is
                    // re-encoded, so no clause is duplicated.
                    self.rebase_theory_state();
                    theory_manager = TheoryManager::new(
                        manager,
                        &mut self.euf,
                        &mut self.arith,
                        &mut self.bv,
                        &self.bv_terms,
                        &self.var_to_constraint,
                        &self.var_to_parsed_arith,
                        &self.term_to_var,
                        &self.var_to_term,
                        &mut self.derived_reasons,
                        self.config.theory_mode,
                        &mut self.statistics,
                        self.config.max_conflicts,
                        self.config.max_decisions,
                        self.has_bv_arith_ops,
                        self.has_quantifiers,
                        &self.quantifier_uf_funcs,
                        deadline,
                    );
                }
            }
        }
    }
}

/// Boolean conflicts the array-axiom refinement's re-solves may accrue, counted
/// from the round that asserts the first array lemma (`#P2b-38` strand (c)).
///
/// # Why a conflict count and not a clock
///
/// This budget used to be a wall-clock floor of two minutes, and that made the
/// *verdict* a property of the machine.  Measured on one release binary and one
/// twelve-line script with no `:timeout`: run alone it answered `sat` at
/// 77.5 s; with ten copies in flight all ten answered `unknown` at the 120 s
/// floor; with six copies all six answered `sat` at 114-116 s.  `oxiz` verdicts
/// are consumed as verification evidence by `cargo-formal`, which requires the
/// same verdict on every machine, so a budget that reads the clock is not a
/// budget this solver may use to *decide* anything.
///
/// `SolverStats::conflicts` is the replacement: monotone, advanced by the
/// search itself rather than by the scheduler, and identical on an idle and a
/// loaded machine.  A user `:timeout` is unaffected and remains the only
/// wall-clock limit there is.
///
/// # Calibration
///
/// Measured on this tree (2026-09-18), total conflicts per script — an
/// over-estimate of the refinement's share, which starts counting later: the
/// whole 217-script `bench/` corpus peaks at 1,408, on a datatype goal that
/// reaches no array lemma at all; 1,200 exhaustive array/UF scripts peak at 65;
/// 1,200 random mixed scripts at 5; 700 new-shape scripts at 4,098; 400
/// array-constant scripts at 1; 300 datatype scripts at 0; an 80-script sample
/// of the `n`-ary-`distinct`-over-store-chains corpus peaks at 8,106, and the
/// twelve-line script that motivated the budget answers `sat` at 558.
///
/// Fifty thousand is six times the largest of those, so nothing that decides
/// today comes near it, and it still turns an unbounded search into an
/// `unknown` that reproduces byte for byte on an idle and on a loaded machine.
/// Verified live rather than assumed: dropping the constant to 1 in an
/// isolated tree copy turns
/// `round4_pass2_recheck_pins::n_ary_distinct_over_store_chains_is_decided`
/// red, so the ceiling really does reach the search.
///
/// It is not, on its own, a bound on the loop: see
/// [`ARRAY_REFINEMENT_LEMMA_BUDGET`] for the case it does not see.
pub(super) const ARRAY_REFINEMENT_RESOLVE_CONFLICTS: u64 = 50_000;

/// Array-axiom lemma instances one `check` may assert before it answers
/// `Unknown` (`#P2b-38` strand (c), second currency).
///
/// # Why a second counter
///
/// [`ARRAY_REFINEMENT_RESOLVE_CONFLICTS`] bounds a refinement loop that
/// *searches*.  It does not bound one that only *builds*: a family that mints
/// a fresh index per pair per round enlarges the circuit every round while the
/// re-solves stay conflict-free, so the conflict count never moves and the
/// ceiling never fires.  That is not hypothetical — `rc3/slow/m5.smt2` (six
/// declarations, three assertions, answered `sat` in 0.5 ms by the 0.3.4 base)
/// ran 400 s under `/usr/bin/time` on the previous tree with no answer and no
/// budget stopping it; only an explicit `:timeout` did.
///
/// Counted in [`crate::solver::Statistics::array_lemma_instances`], which
/// `Solver::assert_new_instances` advances once per lemma that reaches the SAT
/// core, and checked at the same round boundary the conflict ceiling is.
/// Budget exhaustion answers `Unknown` and clears the model; it never produces
/// a verdict.
///
/// # Calibration
///
/// Measured on this tree (2026-09-18), lemma instances per script over the
/// corpora this round uses: the 217-script `bench/` corpus peaks at 116; the
/// four in-tree `n`-ary-`distinct` scripts at 158; the 2,350-script generated
/// campaign at 372; `c20`/`c21` at 12 and 9.  Ten thousand is more than
/// twenty-five times the largest of those, so nothing that decides today comes
/// near it, while the runaway shapes cross it in well under a second.
pub(super) const ARRAY_REFINEMENT_LEMMA_BUDGET: u64 = 10_000;

/// SAT propagations, counted from the entry of this `check`, past which the
/// two *repair* refinements below decline to re-solve: the non-convex-LIA
/// case split (`split_narrow_int_domains`) and bounded model blocking
/// (`block_refuted_model_and_rebase`).
///
/// # Why this is not a clock
///
/// Both repairs re-solve the whole problem from scratch, so both are only
/// affordable when the search so far has been cheap; both used to read
/// `Instant::elapsed()` against `int_case_split::REFINEMENT_TIME_CEILING_MS`
/// (two minutes), *whether or not the caller set a `:timeout`*.  A gate that
/// reads the clock and then decides a verdict makes the verdict a property of
/// the machine: the unaffordable branch of the first marks its `Sat`
/// unverified, and the unaffordable branch of the second falls through to
/// `Unknown`.  Reachability is measured, not argued — with this ceiling set to
/// 1 in an isolated tree copy, 7 of the 217 `bench/` scripts flip `sat` to
/// `unknown`.
///
/// Propagations are the replacement currency: monotone, advanced by the search
/// rather than by the scheduler, and cheap to read.  Decision (9) leaves an
/// explicit `:timeout` as the only wall clock in the solver.
///
/// # Calibration
///
/// Measured on this tree (2026-09-18): of the 217 `bench/` scripts the largest
/// propagation count at the first candidate model is 1.4 million, and the
/// seven scripts the mutation above flips peak at 41 thousand.  One hundred
/// million is seventy times the largest, so every script that decides today
/// stays affordable, while a search that has already propagated a hundred
/// million times is not one a from-scratch re-solve will rescue.
const REFINEMENT_WORK_CEILING_PROPAGATIONS: u64 = 100_000_000;
