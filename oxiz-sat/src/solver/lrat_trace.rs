//! Online LRAT (Linear RAT) proof tracing.
//!
//! Where [`Solver::drat`] streams a self-justifying DRAT proof (every add
//! line is checkable by a RUP/RAT *search* against the clauses known so
//! far, no extra information needed), an LRAT proof instead ships an
//! explicit **hint chain** with every addition: the ids of clauses that,
//! unit-propagated in that order after the new clause's negation is
//! assumed, reach a conflict. A checker given the hints only ever needs
//! forward propagation — see [`oxiz_proof::lrat_check`] for the reader side
//! of that contract.
//!
//! # Building the hint chain: transitive antecedent closure
//!
//! The clause being registered — call it `C` (empty, for the final UNSAT
//! line) — is proved by assuming its negation and showing that assumption,
//! together with clauses already known to a checker, forces a conflict via
//! unit propagation. `C`'s own literals are handed to the checker for free
//! (that is what "assume the negation" means); everything else the
//! derivation actually used has to be an explicit hint.
//!
//! A first attempt at this — replay the *entire* current trail in order,
//! citing every propagated literal's reason clause — is unsound. Two
//! problems, both found by this port's own end-to-end tests:
//!
//! 1. **Unrelated decisions leak in.** The trail can hold decisions from
//!    other, unrelated branches of the search that happen to still be
//!    assigned. A reason clause cited as a hint may contain one of *their*
//!    literals as a "silently discharged" antecedent, but that decision is
//!    not part of `C`'s negation and has no clause backing it — the checker
//!    has no way to know its value, and rejects the proof.
//! 2. **Level-0 antecedents go missing.** [`Solver::analyze`] deliberately
//!    never marks a level-0 variable `seen` (level-0 facts are unconditional
//!    background truths, irrelevant to the *decision-level* structure 1-UIP
//!    resolution reasons about) — but a checker's per-line check has no
//!    persistent state between addition lines: a level-0 fact used as a
//!    "this literal is just false" antecedent inside some cited clause has
//!    to be independently re-derived within *this* hint chain too, or the
//!    checker sees it as unassigned.
//!
//! The fix is to compute exactly the antecedent closure the derivation
//! actually needs, via [`Self::lrat_build_hint_chain`]: start a worklist
//! from the falsified clause's own literals, and for every variable that is
//! *not* already free via `C`, look up its reason clause and push that
//! clause's *other* literals onto the worklist too — recursing until every
//! variable reached either lands in `C` (free) or bottoms out at a clause
//! with no further antecedents (a unit). This mirrors what
//! [`Solver::analyze`] itself does for same-decision-level resolution (its
//! `seen` walk), extended to also recurse into level-0 antecedents that
//! `analyze` skips, and naturally never touches an unrelated decision at
//! all: a variable analyze() would have added directly to `C` (any literal
//! at a level other than the conflict's own) is, by construction, in `C`'s
//! own variable set, so the worklist stops there rather than continuing
//! past it into whatever *that* decision happened to imply.
//!
//! Hints are then emitted in trail order (ascending) restricted to the
//! variables the worklist actually marked needed: trail order is always a
//! valid replay order, since a clause only becomes a reason once every
//! other literal in it was already resolved earlier in the same trail.
//!
//! A plain decision that is reached this way (no clause of any kind backs
//! it, and it is not one of `C`'s own literals) would mean the derivation
//! is not actually decision-independent — [`Self::lrat_build_hint_chain`]'s
//! worklist is structured so this cannot happen for a genuine 1-UIP-derived
//! clause (see the reasoning above); if it somehow did, that variable is
//! silently skipped rather than corrupting the proof with a bogus hint, and
//! the resulting chain — now missing a needed antecedent — fails to verify,
//! which is the fail-safe direction.
//!
//! # What this cannot cover
//!
//! [`Reason::Theory`] literals have no clausal justification by
//! construction (a theory oracle asserted them, not a clause becoming
//! unit), so no hint chain can ever cite one. [`Solver::solve_with_theory`]
//! force-disables LRAT tracing at entry for exactly this reason — see its
//! doc comment.
//!
//! # What is gated off instead of covered
//!
//! Bounded variable elimination, equivalent-literal substitution, gate
//! congruence, failed-literal probing, on-the-fly hyper-binary resolution,
//! and the `enable_inprocessing` subsumption/strengthening pipeline all
//! refuse to run at all while LRAT tracing is active (see each mechanism's
//! own gate, e.g. [`Solver::bounded_variable_elimination`]) rather than
//! attempt a hint chain through machinery this module was not built to
//! cover — the coverage matrix in this port's report lists each one and why.

use super::*;

impl Solver {
    /// Enable online LRAT proof tracing to `path`.
    ///
    /// Must be called before the first [`Solver::add_clause`] (or
    /// [`Solver::add_clause_dimacs`]): every original clause's LRAT id is
    /// assigned at the moment it is added, in the order it is added, so a
    /// clause added before tracing was enabled has no id a later hint could
    /// ever reference. Returns an error rather than silently producing an
    /// incomplete proof if the solver already has clauses or variables.
    pub fn enable_lrat_proof(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        if self.num_vars > 0 || self.clauses.iter_ids().next().is_some() {
            return Err(std::io::Error::other(
                "enable_lrat_proof must be called before any add_clause: original clauses \
                 need an LRAT id assigned at insertion time, in insertion order",
            ));
        }
        let mut writer = crate::proof::LratWriter::new();
        writer.enable(path)?;
        self.lrat = Some(writer);
        Ok(())
    }

    /// Disable LRAT proof tracing (flushing any buffered output).
    pub fn disable_lrat_proof(&mut self) {
        if let Some(mut writer) = self.lrat.take() {
            let _ = writer.flush();
        }
    }

    /// Returns `true` when LRAT proof tracing is currently enabled.
    ///
    /// Also `false` once a proof has been finalized (see
    /// `lrat_emit_empty_from`, a private method not part of this crate's
    /// public API): the writer is closed the instant the proof's conclusion
    /// is written, so tracing is no longer actually happening even if the
    /// caller never called [`Self::disable_lrat_proof`] explicitly.
    #[must_use]
    pub fn lrat_proof_enabled(&self) -> bool {
        self.lrat.is_some()
    }

    /// Whether *any* proof format (DRAT or LRAT) is being traced right now.
    /// Every inprocessing mechanism whose hint coverage this port does not
    /// (yet) reach checks this before running.
    pub(super) fn proof_tracing_active(&self) -> bool {
        self.drat.is_some() || self.lrat.is_some()
    }

    /// Reserve the next original-clause id (no-op, returns `None`, when LRAT
    /// tracing is off). Called once per [`Solver::add_clause`] invocation,
    /// before any tautology/empty/unit-clause special-casing, so the
    /// reserved sequence matches "one id per `add_clause` call, in call
    /// order" — exactly how a caller who also writes a DIMACS file alongside
    /// these calls would number that file's clause lines.
    ///
    /// Uses [`crate::proof::LratWriter::reserve_original_id`] rather than
    /// [`crate::proof::LratWriter::add_original_clause`]: the LRAT format
    /// keeps original clauses implicit (numbered from the accompanying CNF,
    /// never written into the proof stream itself — only derived clauses
    /// and deletions are), so this must not emit a line.
    pub(super) fn lrat_register_original(&mut self) -> Option<u64> {
        let writer = self.lrat.as_mut()?;
        Some(writer.reserve_original_id())
    }

    /// Record that `cid` (a real database clause, just inserted) was
    /// registered with LRAT id `id`.
    pub(super) fn lrat_set_clause_id(&mut self, cid: ClauseId, id: u64) {
        let idx = cid.0 as usize;
        if self.clause_lrat_id.len() <= idx {
            self.clause_lrat_id.resize(idx + 1, 0);
        }
        self.clause_lrat_id[idx] = id;
    }

    /// Record that `var`'s current level-0 value is justified by the
    /// original unit clause registered as LRAT id `id`. Needed because
    /// [`Solver::add_clause`]'s unit-clause fast path installs the fact via
    /// [`Trail::assign_decision`] (no [`ClauseId`] at all — nothing was
    /// inserted into the clause database), so [`Self::lrat_hint_chain`]
    /// cannot recover a justification for it the way it does for a real
    /// [`Reason::Propagation`].
    pub(super) fn lrat_set_unit_justification(&mut self, var: Var, id: u64) {
        let idx = var.index();
        if self.lrat_unit_id.len() <= idx {
            self.lrat_unit_id.resize(idx + 1, 0);
        }
        self.lrat_unit_id[idx] = id;
    }

    /// Clear `var`'s recorded unit justification, if any.
    ///
    /// Called from [`Solver::pop`] and [`Solver::restore_to_trail_size`] for
    /// every variable their rollback unassigns. Without this a stale entry
    /// left behind by a *retracted* unit fact would silently survive to
    /// justify whatever this same variable happens to be reassigned to next
    /// — a different value, reached through a different mechanism, with no
    /// relation at all to the clause id the stale entry names. A hint chain
    /// built from [`Self::lrat_build_hint_chain`] reads `lrat_unit_id` purely
    /// by variable index against whatever the trail currently says, so it has
    /// no way to detect that mismatch on its own; the id must actually be
    /// gone.
    pub(super) fn lrat_clear_unit_justification(&mut self, var: Var) {
        if let Some(slot) = self.lrat_unit_id.get_mut(var.index()) {
            *slot = 0;
        }
    }

    /// The LRAT id registered for `cid`, if any (`0` is the "never
    /// registered" sentinel `LratWriter` never assigns to a real clause).
    pub(super) fn lrat_id_of(&self, cid: ClauseId) -> Option<u64> {
        self.clause_lrat_id
            .get(cid.0 as usize)
            .copied()
            .filter(|&id| id != 0)
    }

    /// Mark every variable transitively needed to justify `seed_lits` (the
    /// literals of the clause fully falsified right now) becoming empty,
    /// stopping at any variable already free via `given` (`C`'s own
    /// variables — empty when deriving the unconditional empty clause
    /// directly). See the module doc comment for why this closure, rather
    /// than a blind trail replay, is what soundness requires.
    fn lrat_mark_needed_antecedents(&self, seed_lits: &[Lit], given: &[Lit]) -> Vec<bool> {
        let mut needed: Vec<bool> = vec![false; self.num_vars];
        let mut worklist: Vec<Var> = seed_lits.iter().map(|l| l.var()).collect();
        while let Some(v) = worklist.pop() {
            if let Some(given_lit) = given.iter().find(|l| l.var() == v) {
                // Free: `v` is one of `C`'s own variables, so the checker's
                // "assume `C`'s negation" step already pins its value —
                // *provided* that pinned value actually agrees with what
                // this trail says `v` is right now. It always should: both
                // `given` (built by `analyze()` from literals currently
                // false on the trail) and every literal this worklist ever
                // pushes (an "other" literal of some `Reason::Propagation`
                // clause, likewise false on the trail by definition of
                // "unit") describe the same trail-false polarity convention.
                // A debug-only check rather than a silent trust: a mismatch
                // here would mean some future call site started handing
                // `given`/`seed_lits` values that no longer share that
                // convention, which would silently corrupt hint chains
                // instead of failing loudly.
                debug_assert!(
                    self.trail.lit_value(*given_lit).is_false(),
                    "lrat_mark_needed_antecedents: {:?} sits in the clause \
                     being proved but is not actually false on the current \
                     trail — the \"free via C\" shortcut requires C's \
                     negation to reproduce this trail's own assignment",
                    v
                );
                continue; // free: `v` is one of `C`'s own variables.
            }
            let idx = v.index();
            if idx >= needed.len() {
                needed.resize(idx + 1, false);
            }
            if needed[idx] {
                continue; // already processed.
            }
            needed[idx] = true;
            if let Reason::Propagation(cid) = self.trail.reason(v)
                && let Some(clause) = self.clauses.get(cid)
            {
                for &lit in &clause.lits {
                    if lit.var() != v {
                        worklist.push(lit.var());
                    }
                }
            }
            // `Reason::Decision` bottoms out here: either `lrat_unit_id`
            // backs it (a single-literal "clause", no further antecedents)
            // or it is a genuine decision that should be unreachable for a
            // sound 1-UIP-derived clause (see the module doc comment) — in
            // either case there is nothing further to recurse into.
            // `Reason::Theory` likewise has no antecedents to offer.
        }
        needed
    }

    /// Build a RUP hint chain sufficient to justify `seed_lits` (the
    /// literals of the clause fully falsified right now — the conflict
    /// clause's literals, or a freshly-registered original clause's own
    /// literals when [`Solver::add_clause`] discovers a contradiction
    /// directly) becoming empty, given that `given`'s variables (`C`'s own,
    /// empty for the unconditional-empty-clause case) are free.
    ///
    /// Skips `final_id == 0` (nothing to justify — see [`Self::lrat_id_of`])
    /// rather than emit a literal `0` into the hint list, which would be
    /// misread as the list's own terminator by any reader of the proof.
    fn lrat_build_hint_chain(&self, seed_lits: &[Lit], given: &[Lit], final_id: u64) -> Vec<u64> {
        let needed = self.lrat_mark_needed_antecedents(seed_lits, given);
        let mut hints = Vec::new();
        for &lit in self.trail.assignments() {
            let var = lit.var();
            let idx = var.index();
            if idx >= needed.len() || !needed[idx] {
                continue;
            }
            match self.trail.reason(var) {
                Reason::Propagation(cid) => {
                    if let Some(id) = self.lrat_id_of(cid) {
                        hints.push(id);
                    }
                }
                Reason::Decision => {
                    if let Some(&id) = self.lrat_unit_id.get(idx)
                        && id != 0
                    {
                        hints.push(id);
                    }
                }
                Reason::Theory => {
                    // No clause backs a theory-asserted literal; nothing can
                    // be cited. `Solver::solve_with_theory` disables LRAT
                    // tracing before this path can ever run under it.
                }
            }
        }
        if final_id != 0 {
            hints.push(final_id);
        }
        hints
    }

    /// Compute the hint chain for the clause about to be learned (`given`,
    /// the *final* — post-minimization — literals conflict analysis
    /// produced) from `conflict` (the clause whose falsification triggered
    /// this learning step), via [`Self::lrat_build_hint_chain`]. `None` when
    /// LRAT tracing is off.
    ///
    /// **Must be called before any backtracking happens for this conflict.**
    /// The antecedent closure walks `self.trail` as it stands right now;
    /// backtracking unassigns exactly the literals a non-trivial backtrack
    /// level was computed to discard, which are frequently the same
    /// literals the hint chain needs to cite. [`Solver::solve`]'s main loop
    /// backtracks before it knows the learned clause's final watch
    /// literals, so it calls this first and [`Self::lrat_finish_learn`]
    /// afterward with the result.
    pub(super) fn lrat_hints_for_conflict(
        &self,
        conflict: ClauseId,
        given: &[Lit],
    ) -> Option<Vec<u64>> {
        self.lrat.as_ref()?;
        let seed_lits: SmallVec<[Lit; 8]> = self
            .clauses
            .get(conflict)
            .map(|c| c.lits.iter().copied().collect())
            .unwrap_or_default();
        let conflict_id = self.lrat_id_of(conflict).unwrap_or(0);
        Some(self.lrat_build_hint_chain(&seed_lits, given, conflict_id))
    }

    /// Register a just-learned clause with LRAT using an already-computed
    /// hint chain (from [`Self::lrat_hints_for_conflict`], captured before
    /// backtracking). No-op (`None`) when LRAT tracing is off.
    pub(super) fn lrat_finish_learn(&mut self, lits: &[Lit], hints: &[u64]) -> Option<u64> {
        let writer = self.lrat.as_mut()?;
        writer.add_clause(lits, hints).ok()
    }

    /// Record `cid`'s retraction with LRAT (no-op if tracing is off or `cid`
    /// was never registered — e.g. it was added while a gated-off mechanism
    /// had already been refused, so it never got an id in the first place).
    pub(super) fn lrat_delete(&mut self, cid: ClauseId) {
        let Some(id) = self.lrat_id_of(cid) else {
            return;
        };
        if let Some(writer) = self.lrat.as_mut() {
            let _ = writer.delete_clause(id);
        }
    }

    /// Finalize the LRAT proof of UNSAT: emit the empty clause, hint-chained
    /// via [`Self::lrat_build_hint_chain`] from `seed_lits` (the literals of
    /// whatever clause is fully falsified right now — see
    /// [`Self::lrat_hints_for_conflict`]'s `seed_lits` for the same
    /// convention) and `final_id` (that clause's own LRAT id). Idempotent —
    /// every `solve()` exit path that reaches UNSAT calls this, but only the
    /// first one to actually run does anything (an original clause
    /// registered as literally empty, via [`Self::lrat_register_original`],
    /// counts as already having finalized the proof: a length-0 original
    /// clause is by itself a complete proof of UNSAT, no derived line
    /// needed).
    ///
    /// Takes and closes the writer rather than leaving it in place: a
    /// finished LRAT proof's own conclusion line (the empty clause) is where
    /// a checker stops reading (see `oxiz_proof::lrat_check`'s per-line loop,
    /// which returns as soon as it sees one), so anything written after it
    /// can never affect verification — but a caller that keeps this same
    /// (incremental) solver around and adds more clauses after an `Unsat`
    /// verdict has no business appending more lines to an already-concluded
    /// proof either. Every other LRAT-writing method here goes through
    /// `self.lrat.as_mut()`/`self.lrat.as_ref()`, so once `self.lrat` is
    /// `None` they all silently become no-ops — the writer does not need a
    /// separate "already finalized" check of its own.
    pub(super) fn lrat_emit_empty_from(&mut self, seed_lits: &[Lit], final_id: u64) {
        if self.lrat_unsat_finalized || self.lrat.is_none() {
            return;
        }
        let hints = self.lrat_build_hint_chain(seed_lits, &[], final_id);
        if let Some(mut writer) = self.lrat.take() {
            let _ = writer.add_empty_clause(&hints);
            let _ = writer.flush();
        }
        self.lrat_unsat_finalized = true;
    }

    /// Mark the LRAT proof already finalized without emitting a derived
    /// empty-clause line — used when [`Self::lrat_register_original`] just
    /// registered a literally empty original clause, which is already
    /// sufficient on its own. Closes the writer for the same reason
    /// [`Self::lrat_emit_empty_from`] does.
    pub(super) fn lrat_mark_finalized_by_original_empty(&mut self) {
        if let Some(mut writer) = self.lrat.take() {
            let _ = writer.flush();
            self.lrat_unsat_finalized = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_lrat_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "oxiz_sat_lrat_trace_{tag}_{}_{}",
            std::process::id(),
            oxiz_time::SystemTime::now()
                .duration_since(oxiz_time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    /// Gatekeeper SK-4: `Solver::pop` must clear a rolled-back variable's
    /// [`Solver::lrat_unit_id`] entry, not leave it pointing at the clause
    /// that justified an assignment which no longer exists. Direct
    /// unit-level check on the field itself (crate-visible via `pub(super)`)
    /// rather than trying to force the stale id into an actual hint chain
    /// end-to-end, which needs the same variable index to be reused by an
    /// unrelated later decision — exact reuse is an implementation detail of
    /// variable/heap allocation this test should not have to pin down.
    #[test]
    fn test_pr26_gatekeeper_sk4_pop_clears_stale_lrat_unit_justification() {
        let mut solver = Solver::new();
        let path = unique_lrat_path("pop");
        solver.enable_lrat_proof(&path).expect("enable lrat");
        let v = solver.new_var();

        solver.push();
        assert!(solver.add_clause([Lit::pos(v)]));
        assert_ne!(
            solver.lrat_unit_id.get(v.index()).copied().unwrap_or(0),
            0,
            "the unit clause just added must have recorded a justification"
        );

        solver.pop();
        assert_eq!(
            solver.lrat_unit_id.get(v.index()).copied().unwrap_or(0),
            0,
            "pop() must clear the justification for every variable it unassigns"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Same guarantee as above, for [`Solver::restore_to_trail_size`] (the
    /// bit-vector theory's incremental-probe rollback) instead of `pop`.
    #[test]
    fn test_pr26_gatekeeper_sk4_restore_to_trail_size_clears_stale_lrat_unit_justification() {
        let mut solver = Solver::new();
        let path = unique_lrat_path("restore");
        solver.enable_lrat_proof(&path).expect("enable lrat");
        let v = solver.new_var();
        let checkpoint = solver.trail_size();

        assert!(solver.add_clause([Lit::pos(v)]));
        assert_ne!(
            solver.lrat_unit_id.get(v.index()).copied().unwrap_or(0),
            0,
            "the unit clause just added must have recorded a justification"
        );

        solver.restore_to_trail_size(checkpoint);
        assert_eq!(
            solver.lrat_unit_id.get(v.index()).copied().unwrap_or(0),
            0,
            "restore_to_trail_size() must clear the justification for every \
             variable it unassigns"
        );

        let _ = std::fs::remove_file(&path);
    }
}
