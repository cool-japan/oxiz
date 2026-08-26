//! Pre-search "lucky" phase (CaDiCaL `lucky.cpp` in spirit).
//!
//! A surprising share of real instances — and almost all of the trivially
//! satisfiable ones a portfolio or a preprocessing pipeline throws at a SAT
//! solver — are satisfied by an assignment that needs no search at all: every
//! variable true, every variable false, or the least model of a Horn formula.
//! CDCL will find those too, eventually, but "eventually" can mean minutes of
//! decisions and restarts on an instance whose answer is a single linear
//! sweep away (the `simon-r18-0`/`r21-1`/`r23-1` family in issue #35 is
//! exactly this shape: CaDiCaL answers in under 5 ms, an unaided CDCL loop
//! times out).
//!
//! This module runs a fixed, small set of structurally-motivated *guesses*
//! before search starts, in increasing order of cost, and reports the first
//! one that is verified to satisfy the formula. See [`LuckyScan`] for what
//! each guess is.
//!
//! # Invariants this phase holds to
//!
//! * **It only ever answers `Sat`, or declines.** Nothing here can conclude
//!   `Unsat`: a failed scan means "this particular guess is not a model",
//!   never "no model exists". The caller falls through to the search — and,
//!   under assumptions, to the ordinary assumption-conflict path that
//!   produces a genuine unsat core.
//! * **A reported model is verified, not assumed.** Every candidate is
//!   checked against the live original clauses by
//!   [`Solver::first_violated_clause`] before it is installed, so a bug in a
//!   scan can only cost time, never produce a wrong `Sat`.
//! * **A declined scan leaves zero side effects.** Candidates are built in
//!   scratch buffers; no clause is added, deleted or strengthened, no
//!   variable is eliminated, nothing is assigned on the trail, and the
//!   branching heuristics' state is untouched. That is what lets this run by
//!   default in every configuration — incremental scopes, assumptions, DRAT
//!   and LRAT tracing alike — where the rest of the pre-search toolkit has to
//!   gate itself off.
//! * **No proof obligation.** A `Sat` verdict needs no DRAT/LRAT line (proofs
//!   certify refutations), so the phase emits nothing even while tracing is
//!   active. It also never *removes* a clause, which is the operation that
//!   forces the other inprocessing passes to step aside under LRAT.
//!
//! # What the scans must respect
//!
//! Two things constrain a candidate beyond the clause database:
//!
//! * **Level-0 facts.** [`Solver::add_clause`] never stores a unit clause in
//!   the database — it assigns the literal at decision level 0 — so a scan
//!   over clauses alone would happily "satisfy" a formula while contradicting
//!   its units. Every candidate therefore starts from the level-0 trail
//!   values and treats them as frozen. (Assignments at levels *above* 0 are a
//!   previous `solve()` call's leftover model, not facts, and are ignored.)
//! * **Assumptions.** Under [`Solver::solve_with_assumptions`] the assumption
//!   literals are seeded into the same frozen base, so a candidate satisfies
//!   them by construction rather than being built and then filtered — the
//!   filtering version essentially never succeeds.
//!
//! # Not wired into `solve_with_theory`
//!
//! Deliberately: there, satisfying the Boolean abstraction is not enough —
//! the theory has to agree, and a `TheoryCallback` learns what the trail says
//! through `on_assignment`, which a scratch-buffer assignment never calls.

use super::*;

/// One pre-search guess. Tried in declaration order, cheapest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuckyScan {
    /// Every free variable true. Satisfies exactly the formulas whose every
    /// clause has a positive literal (or a satisfied frozen one).
    AllTrue,
    /// Every free variable false. The dual: every clause needs a negative
    /// literal.
    AllFalse,
    /// One forward pass over the clauses in database order: a clause already
    /// satisfied by the partial candidate is skipped, otherwise its first
    /// still-free literal is assigned to satisfy it. Fails as soon as a
    /// clause is falsified with no free literal left.
    ///
    /// Assignments are never revised, so a clause satisfied when it is
    /// visited stays satisfied — one pass is a decision procedure for
    /// "greedy in this order works", not an approximation of one.
    ForwardOrdered,
    /// [`LuckyScan::ForwardOrdered`] over the reversed clause order. Distinct
    /// in outcome because the greedy choice is order-dependent: a chain
    /// encoded front-to-back defeats one direction and falls straight out of
    /// the other.
    BackwardOrdered,
    /// The Horn least-model closure: start every free variable false, and
    /// whenever a clause ends up falsified, satisfy it by setting one of its
    /// *positive* literals true — never revising anything back. This computes
    /// the least model and therefore decides Horn satisfiability exactly (for
    /// a Horn formula, if this fails no model exists), while still succeeding
    /// on plenty of non-Horn formulas.
    HornLeastModel,
    /// The dual-Horn greatest-model closure: start every free variable true
    /// and repair falsified clauses through their *negative* literals.
    DualHornGreatestModel,
}

impl LuckyScan {
    /// Every scan, in the order [`Solver::try_lucky_phase`] tries them.
    ///
    /// The two constant scans are strictly subsumed by the two closures that
    /// follow them — a closure that never has to repair a clause *is* the
    /// corresponding constant assignment — and are kept first anyway because
    /// they are dramatically cheaper: one pass over the clauses with no
    /// occurrence lists, no counters and no allocation beyond the candidate
    /// itself. On the instances this phase exists for, they are also the ones
    /// that hit.
    pub const ALL: [Self; 6] = [
        Self::AllTrue,
        Self::AllFalse,
        Self::ForwardOrdered,
        Self::BackwardOrdered,
        Self::HornLeastModel,
        Self::DualHornGreatestModel,
    ];
}

impl Solver {
    /// Try each [`LuckyScan`] in turn; on the first verified model, install it
    /// and report which scan found it. `None` means "keep searching" and
    /// guarantees the solver state is exactly as it was found.
    ///
    /// `assumptions` are the *resolved* assumption literals
    /// ([`Solver::solve_with_assumptions`] rewrites them before deciding on
    /// them); pass an empty slice from the plain [`Solver::solve`] entry
    /// point.
    pub(super) fn try_lucky_phase(&mut self, assumptions: &[Lit]) -> Option<LuckyScan> {
        if !self.config.enable_lucky_phase || self.num_vars == 0 {
            return None;
        }

        // An already-fired interrupt or an already-exhausted conflict budget
        // means the caller has asked for `Unknown` *now*. The scans are
        // bounded and fast, so running them would not hang anything — but a
        // portfolio coordinator that has cancelled this solver is entitled to
        // the same prompt `Unknown` it got before this phase existed, and the
        // CDCL loop's own first act is this identical check.
        //
        // The budget half of it is not hypothetical: `stats.conflicts` is
        // cumulative across `solve()` calls, so a solver re-solved after
        // burning its `max_conflicts` allowance arrives here already over
        // budget and must keep answering `Unknown` rather than suddenly
        // producing a verdict from a route the budget does not meter.
        if self.should_stop_search() {
            return None;
        }

        // An assumption contradicting a level-0 fact is a genuine UNSAT that
        // owes the caller a core; decline so the ordinary assumption path
        // produces it.
        let base = self.lucky_base_assignment(assumptions)?;

        // The clause set every scan works over: live originals. Learned
        // clauses are excluded on purpose. They are *implied* by the
        // originals (1-UIP resolvents of clauses already in the database), so
        // any model of the originals satisfies them and checking them could
        // only ever reject a candidate that is in fact a model — while
        // including them would let a lemma that an incremental `pop` has
        // stranded (see `Solver::first_violated_clause`'s note on units)
        // veto a perfectly good answer. Collected once and reused by every
        // scan, including in reverse.
        let clause_ids: Vec<ClauseId> = self
            .clauses
            .iter_ids()
            .filter(|&id| self.clauses.get(id).is_some_and(|c| !c.learned))
            .collect();

        let mut candidate: Vec<LBool> = Vec::with_capacity(self.num_vars);
        for scan in LuckyScan::ALL {
            self.stats.lucky_attempts += 1;
            if !self.build_lucky_candidate(scan, &base, &clause_ids, &mut candidate) {
                continue;
            }
            // Redundant for the three scans that verify as they go, essential
            // for the constant ones, and cheap for all of them: one linear
            // sweep is the price of never reporting an unverified model.
            if self.first_violated_clause(&candidate, false, 1).is_some() {
                continue;
            }
            self.install_lucky_model(&candidate);
            self.stats.lucky_successes += 1;
            self.debug_verify_lucky_model(scan);
            return Some(scan);
        }
        None
    }

    /// The frozen part of every candidate: level-0 facts plus the
    /// assumptions. `Undef` marks a variable the scans may choose freely.
    ///
    /// `None` means an assumption contradicts a level-0 fact — see
    /// [`Solver::try_lucky_phase`] for why that is the caller's problem, not
    /// something to answer here.
    fn lucky_base_assignment(&self, assumptions: &[Lit]) -> Option<Vec<LBool>> {
        let mut base = vec![LBool::Undef; self.num_vars];
        for (index, slot) in base.iter_mut().enumerate() {
            let var = Var::new(index as u32);
            // Only decision level 0 is a *fact*. `solve()` does not backtrack
            // to the root on entry, so a second call can arrive here with the
            // previous model's decisions still on the trail at higher levels;
            // treating those as frozen would silently turn every scan into
            // "the previous model, patched".
            if self.trail.level(var) == 0 {
                *slot = self.trail.value(var);
            }
        }
        for &lit in assumptions {
            let index = lit.var().index();
            let wanted = LBool::from_bool(lit.is_pos());
            match base.get_mut(index) {
                Some(slot) if *slot == LBool::Undef || *slot == wanted => *slot = wanted,
                // Fixed to the opposite polarity at level 0: decline.
                Some(_) => return None,
                // Out of range cannot happen (`solve_with_assumptions`
                // allocates every assumption variable first), but a candidate
                // that does not cover the literal could not honor it either.
                None => return None,
            }
        }
        Some(base)
    }

    /// Build `scan`'s candidate into `out` (cleared first). `false` means the
    /// scan failed and `out`'s contents are meaningless.
    fn build_lucky_candidate(
        &self,
        scan: LuckyScan,
        base: &[LBool],
        clause_ids: &[ClauseId],
        out: &mut Vec<LBool>,
    ) -> bool {
        match scan {
            LuckyScan::AllTrue => {
                Self::fill_constant_candidate(base, true, out);
                true
            }
            LuckyScan::AllFalse => {
                Self::fill_constant_candidate(base, false, out);
                true
            }
            LuckyScan::ForwardOrdered => self.fill_ordered_candidate(base, clause_ids, false, out),
            LuckyScan::BackwardOrdered => self.fill_ordered_candidate(base, clause_ids, true, out),
            LuckyScan::HornLeastModel => self.fill_closure_candidate(base, clause_ids, false, out),
            LuckyScan::DualHornGreatestModel => {
                self.fill_closure_candidate(base, clause_ids, true, out)
            }
        }
    }

    /// `base` with every free variable set to `value`.
    fn fill_constant_candidate(base: &[LBool], value: bool, out: &mut Vec<LBool>) {
        let filler = LBool::from_bool(value);
        out.clear();
        out.extend(
            base.iter()
                .map(|&fixed| if fixed == LBool::Undef { filler } else { fixed }),
        );
    }

    /// Greedy single pass over the clauses (reversed when `backward`), as
    /// described on [`LuckyScan::ForwardOrdered`].
    fn fill_ordered_candidate(
        &self,
        base: &[LBool],
        clause_ids: &[ClauseId],
        backward: bool,
        out: &mut Vec<LBool>,
    ) -> bool {
        out.clear();
        out.extend_from_slice(base);

        let mut visit = |id: ClauseId| -> bool {
            let Some(clause) = self.clauses.get(id) else {
                return true;
            };
            let mut first_free: Option<Lit> = None;
            for &lit in &clause.lits {
                match out.get(lit.var().index()) {
                    Some(LBool::True) if lit.is_pos() => return true,
                    Some(LBool::False) if !lit.is_pos() => return true,
                    Some(LBool::Undef) if first_free.is_none() => first_free = Some(lit),
                    // Already assigned against this literal, or a variable
                    // outside the candidate's range (which nothing here can
                    // satisfy, so it is not a usable free literal either).
                    _ => {}
                }
            }
            match first_free {
                Some(lit) => {
                    if let Some(slot) = out.get_mut(lit.var().index()) {
                        *slot = LBool::from_bool(lit.is_pos());
                    }
                    true
                }
                // Falsified with nothing left to flip: this greedy order
                // fails on this formula.
                None => false,
            }
        };

        let ok = if backward {
            clause_ids.iter().rev().all(|&id| visit(id))
        } else {
            clause_ids.iter().all(|&id| visit(id))
        };
        if !ok {
            return false;
        }

        // Every clause was left satisfied and assignments are never revised,
        // so whatever is still free occurs in no clause this pass had to
        // touch and is genuinely unconstrained. `false` is therefore an
        // arbitrary — but deterministic — choice, not a meaningful one, and
        // the acceptance scan re-checks the finished candidate regardless.
        let filler = LBool::False;
        for slot in out.iter_mut() {
            if *slot == LBool::Undef {
                *slot = filler;
            }
        }
        true
    }

    /// Horn (`default_true == false`) / dual-Horn (`default_true == true`)
    /// closure.
    ///
    /// Both directions are the same monotone fixpoint with the polarities
    /// swapped, so they share one implementation. Terminology below, stated
    /// once for the `default_true == false` (Horn) reading and mirrored
    /// exactly for the dual:
    ///
    /// * A variable starts *default* (false) and may be *flipped* (to true)
    ///   at most once — never back. That monotonicity is what bounds the
    ///   whole pass at `O(vars + literals)` and what makes the fixpoint the
    ///   *least* model.
    /// * A **support** literal (negative here) satisfies its clause exactly
    ///   while its variable is still default. `support[c]` counts them, and
    ///   only ever decreases.
    /// * A **repair** literal (positive here) satisfies its clause once its
    ///   variable is flipped. When a clause's support runs out, it must be
    ///   repaired — or, if no repair literal is available, the scan fails.
    ///
    /// A variable frozen by [`Solver::lucky_base_assignment`] is never
    /// flipped, which is what keeps level-0 facts and assumptions intact; a
    /// frozen variable already sitting on the flipped value is simply a
    /// repair that came for free.
    fn fill_closure_candidate(
        &self,
        base: &[LBool],
        clause_ids: &[ClauseId],
        default_true: bool,
        out: &mut Vec<LBool>,
    ) -> bool {
        let default = LBool::from_bool(default_true);
        let flipped = LBool::from_bool(!default_true);
        out.clear();
        out.extend(base.iter().map(|&fixed| {
            if fixed == LBool::Undef {
                default
            } else {
                fixed
            }
        }));

        // A support literal is the one whose polarity agrees with the default
        // value: with everything false, `¬v` is the literal that is true.
        let is_support = |lit: Lit| lit.is_pos() == default_true;

        // Which clauses rely on a given variable staying default, as a flat
        // CSR-style index: `occurrence_entries[occurrence_starts[v] ..
        // occurrence_starts[v + 1]]` are the positions of the clauses `v`
        // currently supports. Two `Vec`s for the whole index rather than one
        // per variable — this phase runs on every solve, and a
        // `Vec<Vec<_>>` would allocate (and drop) a header per variable on
        // instances with millions of them, which is exactly the cost the
        // phase is supposed not to have.
        //
        // Only *flippable* variables get entries: a frozen one never changes,
        // so no clause can ever lose its support through it.
        let mut occurrence_starts = vec![0u32; out.len().saturating_add(1)];
        let mut support = vec![0u32; clause_ids.len()];
        let mut pending: Vec<u32> = Vec::new();

        // Pass 1: per-clause support counts, and per-variable degrees
        // accumulated one slot to the right, ready to be prefix-summed into
        // start offsets.
        let supported_by = |lit: Lit, out: &[LBool]| -> Option<usize> {
            let index = lit.var().index();
            (is_support(lit) && out.get(index) == Some(&default)).then_some(index)
        };
        for (position, &id) in clause_ids.iter().enumerate() {
            let Some(clause) = self.clauses.get(id) else {
                continue;
            };
            let mut count = 0u32;
            for &lit in &clause.lits {
                let Some(index) = supported_by(lit, out) else {
                    continue;
                };
                count += 1;
                if base.get(index) == Some(&LBool::Undef)
                    && let Some(degree) = occurrence_starts.get_mut(index + 1)
                {
                    *degree += 1;
                }
            }
            support[position] = count;
            if count == 0 {
                pending.push(position as u32);
            }
        }
        let mut total = 0u32;
        for start in &mut occurrence_starts {
            total += *start;
            *start = total;
        }

        // Pass 2: fill the entries, advancing each variable's cursor.
        let mut cursors = occurrence_starts.clone();
        let mut occurrence_entries = vec![0u32; total as usize];
        for (position, &id) in clause_ids.iter().enumerate() {
            let Some(clause) = self.clauses.get(id) else {
                continue;
            };
            for &lit in &clause.lits {
                let Some(index) = supported_by(lit, out) else {
                    continue;
                };
                if base.get(index) != Some(&LBool::Undef) {
                    continue;
                }
                let Some(cursor) = cursors.get_mut(index) else {
                    continue;
                };
                if let Some(slot) = occurrence_entries.get_mut(*cursor as usize) {
                    *slot = position as u32;
                    *cursor += 1;
                }
            }
        }

        // Worklist over clauses whose support has just run out. Each clause
        // enters at most once (`support` only ever reaches 0 once, since it
        // only decreases), and each variable is flipped at most once, so the
        // loop below touches every literal a constant number of times.
        while let Some(position) = pending.pop() {
            let Some(&id) = clause_ids.get(position as usize) else {
                continue;
            };
            let Some(clause) = self.clauses.get(id) else {
                continue;
            };
            let mut repair: Option<usize> = None;
            let mut satisfied = false;
            for &lit in &clause.lits {
                if is_support(lit) {
                    continue;
                }
                let index = lit.var().index();
                match out.get(index) {
                    Some(value) if *value == flipped => {
                        satisfied = true;
                        break;
                    }
                    // Free and still default: a usable repair. Frozen
                    // variables and out-of-range ones are not.
                    Some(_) if repair.is_none() && base.get(index) == Some(&LBool::Undef) => {
                        repair = Some(index);
                    }
                    _ => {}
                }
            }
            if satisfied {
                continue;
            }
            let Some(index) = repair else {
                // Falsified with no repair available: for a Horn formula this
                // is a proof that no model exists, but this phase never
                // reports UNSAT (see the module doc) — just decline.
                return false;
            };
            if let Some(slot) = out.get_mut(index) {
                *slot = flipped;
            }
            let (from, to) = (
                occurrence_starts.get(index).copied().unwrap_or(0) as usize,
                occurrence_starts.get(index + 1).copied().unwrap_or(0) as usize,
            );
            if let Some(occurrences) = occurrence_entries.get(from..to) {
                for &affected in occurrences {
                    if let Some(count) = support.get_mut(affected as usize) {
                        *count = count.saturating_sub(1);
                        if *count == 0 {
                            pending.push(affected);
                        }
                    }
                }
            }
        }
        true
    }

    /// Install a verified candidate as the model, exactly as the ordinary
    /// `Sat` exit does.
    ///
    /// [`Solver::save_model`] copies the trail and then reconstructs the
    /// variables preprocessing removed; here the copy comes from the
    /// candidate instead, and the very same reconstruction runs on top — so a
    /// caller reading [`Solver::model`] / [`Solver::model_value`] cannot tell
    /// which route produced the answer, including on a solver whose earlier
    /// `solve()` eliminated variables. Those variables occur in no live
    /// clause, so no scan constrains them and nothing but reconstruction
    /// should decide their value; the deleted clauses they do occur in are
    /// re-satisfied by that reconstruction, which is the same promise the
    /// trail-based path relies on.
    fn install_lucky_model(&mut self, candidate: &[LBool]) {
        self.model.clear();
        self.model.extend_from_slice(candidate);
        self.model.resize(self.num_vars, LBool::Undef);
        self.apply_model_reconstruction();
    }

    /// Release build: compiles away entirely.
    #[cfg(not(debug_assertions))]
    #[inline]
    fn debug_verify_lucky_model(&self, _scan: LuckyScan) {}

    /// Debug-only net over the *installed* model, after reconstruction.
    ///
    /// The sibling of [`Solver::debug_verify_model`] for this path, and
    /// restricted to original clauses for the same reason
    /// [`Solver::debug_verify_model_input`] is: a lucky model is verified
    /// against the originals only, learned clauses being implied by them
    /// rather than separately enforced here. Running it after
    /// [`Solver::install_lucky_model`] rather than on the raw candidate is
    /// the point — it is the reconstruction step, not the scan, that this
    /// adds coverage for.
    #[cfg(debug_assertions)]
    fn debug_verify_lucky_model(&self, scan: LuckyScan) {
        if let Some(id) = self.first_violated_clause(&self.model, false, 1) {
            let lits = self.clauses.get(id).map(|c| c.lits.clone());
            panic!(
                "lucky phase ({scan:?}) reported Sat with a model that violates ORIGINAL clause \
                 {id:?} ({lits:?}); a candidate that passed the acceptance scan was corrupted by \
                 model reconstruction"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(dimacs: i32) -> Lit {
        Lit::from_dimacs(dimacs)
    }

    /// Build a solver over `vars` variables with the given DIMACS clauses.
    fn solver_with(vars: u32, clauses: &[&[i32]]) -> Solver {
        let mut solver = Solver::new();
        for _ in 0..vars {
            solver.new_var();
        }
        for clause in clauses {
            let lits: Vec<Lit> = clause.iter().map(|&d| lit(d)).collect();
            assert!(solver.add_clause(lits), "clause {clause:?} rejected");
        }
        solver
    }

    /// Every clause has a positive literal: the first scan already wins.
    #[test]
    fn all_true_scan_solves_all_positive_formula() {
        let mut solver = solver_with(3, &[&[1, 2], &[2, 3], &[1, 3], &[1, -2, 3]]);
        assert_eq!(solver.try_lucky_phase(&[]), Some(LuckyScan::AllTrue));
        assert_eq!(solver.stats.lucky_attempts, 1);
        assert_eq!(solver.stats.lucky_successes, 1);
    }

    /// Every clause has a negative literal, and at least one has no positive
    /// one, so all-true fails and all-false is reached.
    #[test]
    fn all_false_scan_solves_all_negative_formula() {
        let mut solver = solver_with(3, &[&[-1, -2], &[-2, -3], &[-1, 2, -3]]);
        assert_eq!(solver.try_lucky_phase(&[]), Some(LuckyScan::AllFalse));
        assert_eq!(solver.stats.lucky_attempts, 2);
    }

    /// Neither constant assignment works — the first clause forces a positive
    /// choice, the last an all-negative one — but a greedy forward pass does:
    /// it satisfies `(1 ∨ 2)` by taking `1`, which leaves `2` free to be
    /// false for `(-2 ∨ -3)` and `(-2)`-containing clauses.
    #[test]
    fn forward_ordered_scan_solves_order_dependent_formula() {
        let mut solver = solver_with(3, &[&[1, 2], &[-2, -3], &[1, -2], &[-2, 3, -1]]);
        assert_eq!(solver.try_lucky_phase(&[]), Some(LuckyScan::ForwardOrdered));
        assert_eq!(solver.stats.lucky_attempts, 3);
    }

    /// The mirror image: the greedy choice made by the *first* clause in
    /// database order paints the forward pass into a corner, while starting
    /// from the last clause does not.
    #[test]
    fn backward_ordered_scan_solves_reverse_order_dependent_formula() {
        // Forward: (1∨2) takes 1 → (-1∨2) takes 2 → (-1∨-2) is falsified with
        // nothing free. Backward: (-1∨-2) takes ¬1 → (-1∨2) already true →
        // (1∨2) takes 2. Neither constant assignment works ((-1∨-2) kills
        // all-true, (1∨2) kills all-false), and the Horn closure is only
        // tried afterwards.
        let mut solver = solver_with(2, &[&[1, 2], &[-1, 2], &[-1, -2]]);
        assert_eq!(
            solver.try_lucky_phase(&[]),
            Some(LuckyScan::BackwardOrdered)
        );
        assert_eq!(solver.stats.lucky_attempts, 4);
    }

    /// Four clauses over `a, b, c` (offset by `base`) that defeat a *forward*
    /// greedy pass while remaining satisfiable and solvable by the Horn
    /// closure.
    ///
    /// Forward: `(¬a∨¬b)` takes its first free literal `¬a` (a := false),
    /// `(a∨b)` is then forced to `b := true`, `(a∨c)` to `c := true`, and
    /// `(¬c∨¬b)` is left fully falsified with nothing to flip. The Horn
    /// closure instead starts everything false — which already satisfies both
    /// negative clauses — and repairs only `(a∨b)` through its first positive
    /// literal, landing on `a` alone, which satisfies `(a∨c)` too.
    fn forward_trap(base: i32) -> Vec<Vec<i32>> {
        let (a, b, c) = (base + 1, base + 2, base + 3);
        vec![vec![-a, -b], vec![a, b], vec![a, c], vec![-c, -b]]
    }

    /// [`forward_trap`] with every literal negated: defeats the same greedy
    /// pass (which is polarity-blind) but is solvable by the *dual* closure
    /// rather than the Horn one.
    fn negated(clauses: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        clauses
            .into_iter()
            .map(|clause| clause.into_iter().map(|lit| -lit).collect())
            .collect()
    }

    fn as_slices(clauses: &[Vec<i32>]) -> Vec<&[i32]> {
        clauses.iter().map(Vec::as_slice).collect()
    }

    /// A [`forward_trap`] group followed by a second group in *reversed*
    /// order, so the forward pass trips on the first group and the backward
    /// pass trips on the second. Six variables, eight binary clauses,
    /// satisfiable, and solved by the Horn closure.
    fn both_greedy_directions_trapped() -> Vec<Vec<i32>> {
        let mut clauses = forward_trap(0);
        let mut mirrored = forward_trap(3);
        mirrored.reverse();
        clauses.extend(mirrored);
        clauses
    }

    /// Not a Horn formula — `(a∨b)` has two positive literals — but the Horn
    /// *closure* is exactly the right tool for it, and the only one of the
    /// six scans that works.
    ///
    /// A genuinely Horn formula would never reach this scan: after level-0
    /// propagation its every remaining clause still has an unassigned
    /// negative literal (otherwise it would have propagated), so `AllFalse`
    /// two scans earlier always wins. That is why this test's formula is
    /// built from greedy traps instead of an implication chain — and why
    /// [`LuckyScan::HornLeastModel`] earns its place by generalizing the
    /// constant scans, not by handling textbook Horn inputs.
    #[test]
    fn horn_closure_scan_solves_what_the_greedy_passes_cannot() {
        let clauses = both_greedy_directions_trapped();
        let mut solver = solver_with(6, &as_slices(&clauses));
        assert_eq!(solver.try_lucky_phase(&[]), Some(LuckyScan::HornLeastModel));
        assert_eq!(solver.stats.lucky_attempts, 5, "the first four must fail");
        assert_eq!(solver.stats.lucky_successes, 1);
        // The closure's fixpoint: one variable per group repaired to true.
        assert_eq!(solver.model_value(Var::new(0)), LBool::True);
        assert_eq!(solver.model_value(Var::new(1)), LBool::False);
        assert_eq!(solver.model_value(Var::new(2)), LBool::False);
        assert_eq!(solver.model_value(Var::new(3)), LBool::True);
    }

    /// The exact mirror image, which only the dual closure solves — the Horn
    /// closure is tried first and fails on it.
    #[test]
    fn dual_horn_closure_scan_solves_the_mirror_image() {
        let clauses = negated(both_greedy_directions_trapped());
        let mut solver = solver_with(6, &as_slices(&clauses));
        assert_eq!(
            solver.try_lucky_phase(&[]),
            Some(LuckyScan::DualHornGreatestModel)
        );
        assert_eq!(solver.stats.lucky_attempts, 6, "the last scan is the one");
        assert_eq!(solver.model_value(Var::new(0)), LBool::False);
        assert_eq!(solver.model_value(Var::new(1)), LBool::True);
        assert_eq!(solver.model_value(Var::new(2)), LBool::True);
        assert_eq!(solver.model_value(Var::new(3)), LBool::False);
    }

    /// A formula no scan solves: every scan is attempted, none succeeds, and
    /// the solver is left untouched for the search to take over.
    #[test]
    fn unlucky_formula_declines_after_every_scan() {
        // Both greedy traps *and* their mirror image, on disjoint variables:
        // the constant scans die on the first group, both greedy passes die
        // on one trap or the other, the Horn closure dies on the mirrored
        // half and the dual closure on the original half. Still perfectly
        // satisfiable — each group is independent and each has a model.
        let mut clauses = both_greedy_directions_trapped();
        clauses.extend(negated(
            both_greedy_directions_trapped()
                .into_iter()
                .map(|clause| {
                    clause
                        .into_iter()
                        .map(|lit| if lit > 0 { lit + 6 } else { lit - 6 })
                        .collect()
                })
                .collect(),
        ));
        let mut solver = solver_with(12, &as_slices(&clauses));
        let trail_before = solver.trail.assignments().len();
        assert_eq!(solver.try_lucky_phase(&[]), None);
        assert_eq!(
            solver.stats.lucky_attempts,
            LuckyScan::ALL.len() as u64,
            "every scan should have been tried"
        );
        assert_eq!(solver.stats.lucky_successes, 0);
        assert_eq!(solver.trail.assignments().len(), trail_before);
        // Still solvable the ordinary way, and the declined phase left no
        // residue behind.
        assert_eq!(solver.solve(), SolverResult::Sat);
    }

    /// The flag genuinely bypasses everything: no scan is even attempted.
    #[test]
    fn disabled_flag_skips_the_phase_entirely() {
        let mut solver = Solver::with_config(SolverConfig {
            enable_lucky_phase: false,
            ..SolverConfig::default()
        });
        for _ in 0..3 {
            solver.new_var();
        }
        assert!(solver.add_clause(vec![lit(1), lit(2)]));
        assert_eq!(solver.try_lucky_phase(&[]), None);
        assert_eq!(solver.stats.lucky_attempts, 0);
    }

    /// A level-0 fact the all-true scan would contradict is respected: the
    /// unit `(-1)` is not in the clause database at all, so only the frozen
    /// base can carry it.
    #[test]
    fn level_zero_facts_are_frozen_into_every_candidate() {
        let mut solver = solver_with(2, &[&[-1], &[1, 2]]);
        let scan = solver.try_lucky_phase(&[]);
        assert!(scan.is_some(), "formula is satisfiable by a lucky scan");
        assert_eq!(solver.model_value(Var::new(0)), LBool::False);
        assert_eq!(solver.model_value(Var::new(1)), LBool::True);
    }

    /// An assumption steers the candidate rather than being checked after the
    /// fact: `¬2` is seeded frozen, so the all-true scan cannot be taken.
    #[test]
    fn assumptions_are_seeded_and_respected() {
        let mut solver = solver_with(2, &[&[1, 2]]);
        let scan = solver.try_lucky_phase(&[lit(-2)]);
        assert!(scan.is_some());
        assert_eq!(solver.model_value(Var::new(1)), LBool::False);
        assert_eq!(solver.model_value(Var::new(0)), LBool::True);
    }

    /// An assumption contradicting a level-0 fact makes the phase decline
    /// outright, leaving the UNSAT-core path to the caller.
    #[test]
    fn assumption_contradicting_a_level_zero_fact_declines() {
        let mut solver = solver_with(2, &[&[1], &[1, 2]]);
        assert_eq!(solver.try_lucky_phase(&[lit(-1)]), None);
        assert_eq!(solver.stats.lucky_attempts, 0);
    }
}
