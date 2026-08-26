//! Preprocessing techniques for SAT solving
//!
//! This module implements various preprocessing and simplification techniques:
//! - Blocked Clause Elimination (BCE)
//! - Variable Elimination (VE)
//! - Subsumption Elimination
//! - Pure Literal Elimination
//! - Self-Subsuming Resolution

use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::{Lit, Var};
#[allow(unused_imports)]
use crate::prelude::*;

/// Occurrence list: maps literals to clauses containing them
#[derive(Debug, Clone)]
struct OccurrenceList {
    /// occurrences[lit] = list of clause IDs containing lit
    occurrences: Vec<Vec<ClauseId>>,
}

impl OccurrenceList {
    fn new(num_vars: usize) -> Self {
        Self {
            occurrences: vec![Vec::new(); num_vars * 2],
        }
    }

    /// Record `clause_id` under `lit`, growing the backing table if the
    /// literal names a variable beyond the count this list was sized for.
    ///
    /// `Preprocessor::new` is handed the caller's `num_vars`, but a clause
    /// database can legitimately mention a higher-numbered variable (a solver
    /// that created variables after the preprocessor was built, or a clause
    /// set loaded independently of any `num_vars` bookkeeping). Indexing
    /// straight into a fixed-size table would panic there; growing keeps the
    /// occurrence data complete instead, which matters because
    /// [`Preprocessor::subsumption_elimination`] now *drives* its candidate
    /// search off these lists — a silently missing entry would turn into a
    /// silently missed subsumption.
    fn add(&mut self, lit: Lit, clause_id: ClauseId) {
        let idx = lit.code() as usize;
        if idx >= self.occurrences.len() {
            self.occurrences.resize(idx + 1, Vec::new());
        }
        self.occurrences[idx].push(clause_id);
    }

    fn get(&self, lit: Lit) -> &[ClauseId] {
        self.occurrences
            .get(lit.code() as usize)
            .map_or(&[], |list| list.as_slice())
    }

    fn remove(&mut self, lit: Lit, clause_id: ClauseId) {
        let Some(list) = self.occurrences.get_mut(lit.code() as usize) else {
            return;
        };
        if let Some(pos) = list.iter().position(|&id| id == clause_id) {
            list.swap_remove(pos);
        }
    }

    fn clear(&mut self) {
        for list in &mut self.occurrences {
            list.clear();
        }
    }
}

/// Preprocessing context
#[derive(Debug)]
pub struct Preprocessor {
    /// Number of variables
    num_vars: usize,
    /// Occurrence lists
    occurrences: OccurrenceList,
    /// Eliminated variables
    eliminated: HashSet<Var>,
    /// Clauses to remove
    removed_clauses: HashSet<ClauseId>,
    /// Pure literals whose clauses were removed by [`Self::pure_literal_elimination`].
    ///
    /// Deleting a pure literal's clauses is only satisfiability-preserving if the
    /// literal is fixed to its satisfying polarity in the reconstructed model;
    /// otherwise the reported model can falsify a deleted (but still asserted)
    /// original clause. The caller must apply these to the model.
    eliminated_pure_literals: Vec<Lit>,
}

impl Preprocessor {
    /// Create a new preprocessor
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            occurrences: OccurrenceList::new(num_vars),
            eliminated: HashSet::new(),
            removed_clauses: HashSet::new(),
            eliminated_pure_literals: Vec::new(),
        }
    }

    /// Pure literals recorded during the last (and any prior) call to
    /// [`Self::pure_literal_elimination`]. Each must be fixed to its polarity
    /// (true) in the reconstructed model so the deleted clauses stay satisfied.
    #[must_use]
    pub fn eliminated_pure_literals(&self) -> &[Lit] {
        &self.eliminated_pure_literals
    }

    /// Build occurrence lists from clause database
    pub fn build_occurrences(&mut self, clauses: &ClauseDatabase) {
        self.occurrences.clear();
        for i in 0..clauses.len() {
            let clause_id = ClauseId::new(i as u32);
            if let Some(clause) = clauses.get(clause_id)
                && !clause.deleted
            {
                for &lit in &clause.lits {
                    self.occurrences.add(lit, clause_id);
                }
            }
        }
    }

    /// Check if a clause is a tautology (contains both l and ~l)
    fn is_tautology(lits: &[Lit]) -> bool {
        let mut seen = HashSet::new();
        for &lit in lits {
            if seen.contains(&lit.negate()) {
                return true;
            }
            seen.insert(lit);
        }
        false
    }

    /// Check if clause C is blocked on literal l
    ///
    /// A clause C is blocked on literal l ∈ C if for every clause D with ~l ∈ D,
    /// the resolvent of C and D on l is a tautology.
    fn is_blocked(&self, clause_lits: &[Lit], blocking_lit: Lit, clauses: &ClauseDatabase) -> bool {
        // Get all clauses containing ~blocking_lit
        let neg_lit = blocking_lit.negate();

        for &other_clause_id in self.occurrences.get(neg_lit) {
            if let Some(other_clause) = clauses.get(other_clause_id) {
                if other_clause.deleted {
                    continue;
                }

                // Compute resolvent: (C \ {l}) ∪ (D \ {~l})
                let mut resolvent = Vec::new();

                // Add literals from C except blocking_lit
                for &lit in clause_lits {
                    if lit != blocking_lit {
                        resolvent.push(lit);
                    }
                }

                // Add literals from D except ~blocking_lit
                for &lit in &other_clause.lits {
                    if lit != neg_lit {
                        resolvent.push(lit);
                    }
                }

                // Check if resolvent is a tautology
                if !Self::is_tautology(&resolvent) {
                    return false;
                }
            }
        }

        true
    }

    /// Blocked Clause Elimination
    ///
    /// Remove clauses that are blocked on some literal.
    /// Returns the number of clauses eliminated.
    pub fn blocked_clause_elimination(&mut self, clauses: &mut ClauseDatabase) -> usize {
        let mut eliminated = 0;
        self.build_occurrences(clauses);

        // Try to eliminate each clause
        let clause_ids: Vec<_> = (0..clauses.len())
            .map(|i| ClauseId::new(i as u32))
            .collect();

        for clause_id in clause_ids {
            if self.removed_clauses.contains(&clause_id) {
                continue;
            }

            if let Some(clause) = clauses.get(clause_id) {
                if clause.deleted || clause.learned {
                    continue;
                }

                let lits = clause.lits.clone();

                // Try each literal as blocking literal
                for &lit in &lits {
                    if self.is_blocked(&lits, lit, clauses) {
                        // Mark clause for removal
                        if let Some(clause) = clauses.get_mut(clause_id) {
                            clause.deleted = true;
                        }
                        self.removed_clauses.insert(clause_id);
                        eliminated += 1;

                        // Update occurrence lists
                        for &l in &lits {
                            self.occurrences.remove(l, clause_id);
                        }
                        break;
                    }
                }
            }
        }

        eliminated
    }

    /// Pure Literal Elimination
    ///
    /// A pure literal is one that appears only in positive or only in negative form.
    /// All clauses containing a pure literal can be satisfied and removed.
    ///
    /// `fixed` reports, per variable index, whether the variable already has a
    /// permanent (level-0) value that lives on the caller's trail rather than
    /// in `clauses`. Unit facts are typically consumed straight onto the trail
    /// and never stored as a retrievable clause, so `build_occurrences` below
    /// is blind to them: a variable whose only clause-level occurrences are
    /// positive can still be forced *false* by such a trail fact. Without this
    /// exclusion this pass would misjudge that variable "pure positive", drop
    /// its remaining clauses, and ask the caller to fix it `true` in the final
    /// model -- contradicting the trail value the rest of the search relied on
    /// and producing a model that violates whatever clause forced the fact.
    /// Passing an empty slice disables the check (treats every variable as
    /// unfixed), matching the pre-existing behavior for callers with no trail.
    ///
    /// Returns the number of clauses eliminated.
    pub fn pure_literal_elimination(
        &mut self,
        clauses: &mut ClauseDatabase,
        fixed: &[bool],
    ) -> usize {
        let mut eliminated = 0;
        self.build_occurrences(clauses);

        // Find pure literals
        let mut pure_literals = Vec::new();

        for v in 0..self.num_vars {
            if fixed.get(v).copied().unwrap_or(false) {
                continue;
            }
            let var = Var(v as u32);
            let pos_lit = Lit::pos(var);
            let neg_lit = Lit::neg(var);

            let pos_occurs = !self.occurrences.get(pos_lit).is_empty();
            let neg_occurs = !self.occurrences.get(neg_lit).is_empty();

            if pos_occurs && !neg_occurs {
                pure_literals.push(pos_lit);
            } else if neg_occurs && !pos_occurs {
                pure_literals.push(neg_lit);
            }
        }

        // Remove clauses containing pure literals
        for lit in pure_literals {
            let mut removed_any = false;
            for &clause_id in self.occurrences.get(lit).iter() {
                if !self.removed_clauses.contains(&clause_id)
                    && let Some(clause) = clauses.get_mut(clause_id)
                    && !clause.deleted
                    && !clause.learned
                {
                    clause.deleted = true;
                    self.removed_clauses.insert(clause_id);
                    eliminated += 1;
                    removed_any = true;
                }
            }
            // Record the pure literal so the caller can fix it to `true` in the
            // reconstructed model, keeping the deleted clauses satisfied.
            if removed_any {
                self.eliminated_pure_literals.push(lit);
            }
        }

        eliminated
    }

    /// Forward subsumption elimination (occurrence-accelerated,
    /// order-insensitive).
    ///
    /// A clause C subsumes clause D when C ⊆ D; D is then implied by C and can
    /// be deleted. Returns the number of clauses eliminated.
    ///
    /// Three properties this pass guarantees, the first two of which the
    /// earlier index-ordered all-pairs version did not:
    ///
    /// * **Order-insensitive.** Clauses are visited as subsumers in
    ///   *shortest-first* order, not database order, so a short clause stored
    ///   late still retires the long clauses stored before it. The previous
    ///   version only ever tested `earlier subsumes later`, which made the
    ///   result depend on insertion order: `(x0 ∨ x1)` added before `(x0)`
    ///   left both alive, the same two clauses in the other order left one.
    ///
    /// * **Duplicate-safe.** Two clauses with identical literal sets subsume
    ///   each other, so a naive "check both directions" fix deletes *both*
    ///   and loses the constraint entirely. Equal-length pairs are therefore
    ///   broken by clause id: only the higher-id twin is retired, so exactly
    ///   one representative always survives.
    ///
    /// * **Occurrence-driven.** For each subsumer only the clauses on the
    ///   occurrence list of its *least frequently occurring* literal are
    ///   examined — every clause D with C ⊆ D necessarily contains that
    ///   literal, so no subsumption is missed. The previous version rescanned
    ///   every later clause in the database for every clause, an
    ///   O(n² · |C| · |D|) sweep, while `build_occurrences` was computed and
    ///   then never read.
    ///
    /// Learned clauses are neither subsumers nor subsumption targets here
    /// (matching the previous behavior): the clause-database reduction
    /// machinery owns their lifetime, and deleting one out from under it
    /// would desynchronize `learned_clause_ids` bookkeeping in the caller.
    pub fn subsumption_elimination(&mut self, clauses: &mut ClauseDatabase) -> usize {
        let mut eliminated = 0;
        self.build_occurrences(clauses);

        // (length, id) pairs for every clause eligible to act as a subsumer or
        // to be subsumed. Sorting ascending puts the cheapest, most powerful
        // subsumers first and makes the pass independent of insertion order.
        let mut candidates: Vec<(usize, u32)> = Vec::new();
        for i in 0..clauses.len() {
            let clause_id = ClauseId::new(i as u32);
            if self.removed_clauses.contains(&clause_id) {
                continue;
            }
            if let Some(clause) = clauses.get(clause_id)
                && !clause.deleted
                && !clause.learned
            {
                candidates.push((clause.lits.len(), clause_id.0));
            }
        }
        candidates.sort_unstable();

        // `marked[lit]` flags the current subsumer's literals; `cover[lit]`
        // stamps which of them the candidate under test has already matched,
        // so a candidate carrying a repeated literal can never be miscounted
        // as covering two distinct subsumer literals.
        let mut marked: Vec<bool> = vec![false; self.num_vars * 2];
        let mut cover: Vec<u64> = vec![0; self.num_vars * 2];
        let mut stamp: u64 = 0;

        for &(sub_len, sub_raw) in &candidates {
            let sub_id = ClauseId::new(sub_raw);
            if self.removed_clauses.contains(&sub_id) {
                continue;
            }
            let sub_lits = match clauses.get(sub_id) {
                Some(clause) if !clause.deleted && !clause.learned => clause.lits.clone(),
                _ => continue,
            };
            if sub_lits.is_empty() {
                continue;
            }

            // Every clause containing all of `sub_lits` contains this literal,
            // so its occurrence list is a complete candidate set. Chosen
            // before any marking so the `None` arm (unreachable: `sub_lits` is
            // non-empty) cannot leave stale marks behind.
            let Some(pivot) = sub_lits
                .iter()
                .copied()
                .min_by_key(|&lit| self.occurrences.get(lit).len())
            else {
                continue;
            };
            let pivot_occurrences: Vec<ClauseId> = self.occurrences.get(pivot).to_vec();

            let needed_len = sub_lits
                .iter()
                .map(|lit| lit.code() as usize + 1)
                .max()
                .unwrap_or(0);
            if needed_len > marked.len() {
                marked.resize(needed_len, false);
                cover.resize(needed_len, 0);
            }

            // Mark the subsumer's distinct literals.
            let mut distinct = 0usize;
            for &lit in &sub_lits {
                let idx = lit.code() as usize;
                if !marked[idx] {
                    marked[idx] = true;
                    distinct += 1;
                }
            }

            for other_id in pivot_occurrences {
                if other_id == sub_id || self.removed_clauses.contains(&other_id) {
                    continue;
                }

                let covered = {
                    let Some(other) = clauses.get(other_id) else {
                        continue;
                    };
                    if other.deleted || other.learned {
                        continue;
                    }
                    let other_len = other.lits.len();
                    // A strictly shorter clause can never be a superset; an
                    // equally long one is only retired when it is the
                    // higher-id twin (see the duplicate-safety note above).
                    if other_len < sub_len || (other_len == sub_len && other_id.0 <= sub_raw) {
                        continue;
                    }

                    stamp += 1;
                    let mut hits = 0usize;
                    for &lit in &other.lits {
                        let idx = lit.code() as usize;
                        if idx < marked.len() && marked[idx] && cover[idx] != stamp {
                            cover[idx] = stamp;
                            hits += 1;
                        }
                    }
                    hits == distinct
                };

                if covered {
                    if let Some(other) = clauses.get_mut(other_id) {
                        other.deleted = true;
                    }
                    self.removed_clauses.insert(other_id);
                    eliminated += 1;
                }
            }

            for &lit in &sub_lits {
                marked[lit.code() as usize] = false;
            }
        }

        eliminated
    }

    /// Variable Elimination (Bounded Variable Elimination).
    ///
    /// Eliminate a variable by resolving all pairs of clauses containing v and ~v,
    /// but only if the number of resulting clauses is not too large.
    /// Returns the number of variables eliminated.
    ///
    /// # Alternative implementation — not the wired production pass
    ///
    /// The BVE that actually runs inside [`crate::Solver`] is
    /// `Solver::bounded_variable_elimination` (`solver/bve.rs`), reached from
    /// `Solver::solve` under [`crate::SolverConfig::enable_bve`].
    ///
    /// This one is deliberately **not** wired, and the reason is soundness,
    /// not merely duplication: eliminating a variable is only
    /// satisfiability-preserving if the variable's value can be *reconstructed*
    /// afterwards from the surviving assignment. This routine records the
    /// eliminated variable in `self.eliminated` and deletes its clauses, but
    /// keeps no record of *which* clauses defined it, so a caller has no way
    /// to recover a value for it — the reported model would leave the variable
    /// at whatever default the solver happens to hold and could falsify a
    /// deleted (but still asserted) original clause. `solver/bve.rs` keeps
    /// exactly that reconstruction data (`bve_def` / `bve_order`, replayed by
    /// `Solver::save_model`), which is why it is the wired one.
    ///
    /// Retained as a compact, dependency-free reference formulation of the
    /// resolution/growth-bound core, exercised only by this module's own
    /// tests.
    #[allow(dead_code)]
    pub fn variable_elimination(&mut self, clauses: &mut ClauseDatabase, limit: usize) -> usize {
        let mut eliminated = 0;
        self.build_occurrences(clauses);

        for v in 0..self.num_vars {
            let var = Var(v as u32);
            if self.eliminated.contains(&var) {
                continue;
            }

            let pos_lit = Lit::pos(var);
            let neg_lit = Lit::neg(var);

            let pos_clauses: Vec<_> = self.occurrences.get(pos_lit).to_vec();
            let neg_clauses: Vec<_> = self.occurrences.get(neg_lit).to_vec();

            // Bound check: only eliminate if cost is reasonable
            let resolvents = pos_clauses.len() * neg_clauses.len();
            let current = pos_clauses.len() + neg_clauses.len();

            if resolvents > limit || resolvents > current {
                continue;
            }

            // Generate all resolvents
            let mut new_clauses = Vec::new();

            for &pos_clause_id in &pos_clauses {
                for &neg_clause_id in &neg_clauses {
                    let pos_lits = if let Some(c) = clauses.get(pos_clause_id) {
                        &c.lits
                    } else {
                        continue;
                    };

                    let neg_lits = if let Some(c) = clauses.get(neg_clause_id) {
                        &c.lits
                    } else {
                        continue;
                    };

                    // Compute resolvent
                    let mut resolvent = Vec::new();

                    for &lit in pos_lits {
                        if lit != pos_lit {
                            resolvent.push(lit);
                        }
                    }

                    for &lit in neg_lits {
                        if lit != neg_lit && !resolvent.contains(&lit) {
                            resolvent.push(lit);
                        }
                    }

                    // Check if resolvent is tautology
                    if Self::is_tautology(&resolvent) {
                        continue;
                    }

                    new_clauses.push(resolvent);
                }
            }

            // Eliminate the variable
            self.eliminated.insert(var);
            eliminated += 1;

            // Remove old clauses
            for &clause_id in &pos_clauses {
                if let Some(clause) = clauses.get_mut(clause_id) {
                    clause.deleted = true;
                }
                self.removed_clauses.insert(clause_id);
            }

            for &clause_id in &neg_clauses {
                if let Some(clause) = clauses.get_mut(clause_id) {
                    clause.deleted = true;
                }
                self.removed_clauses.insert(clause_id);
            }

            // Add new clauses
            for resolvent in new_clauses {
                if !resolvent.is_empty() {
                    clauses.add_original(resolvent);
                }
            }
        }

        eliminated
    }

    /// Failed Literal Probing
    ///
    /// Try to assign each literal and propagate. If a literal leads to a conflict,
    /// we can infer its negation must be true (failed literal).
    /// Returns the number of failed literals found.
    pub fn failed_literal_probing(&mut self, clauses: &mut ClauseDatabase) -> usize {
        use crate::trail::Trail;
        use crate::watched::{WatchLists, Watcher};

        let mut found = 0;
        self.build_occurrences(clauses);

        // Create temporary trail and watch lists for probing
        let mut trail = Trail::new(self.num_vars);
        let mut watches = WatchLists::new(self.num_vars);

        // Build watch lists from current clauses
        for i in 0..clauses.len() {
            let clause_id = ClauseId::new(i as u32);
            if let Some(clause) = clauses.get(clause_id) {
                if clause.deleted || clause.lits.len() < 2 {
                    continue;
                }

                let lit0 = clause.lits[0];
                let lit1 = clause.lits[1];
                watches.add(lit0.negate(), Watcher::new(clause_id, lit1));
                watches.add(lit1.negate(), Watcher::new(clause_id, lit0));
            }
        }

        // Try to probe each literal
        let mut failed_literals = Vec::new();

        for v in 0..self.num_vars {
            let var = Var(v as u32);
            if trail.is_assigned(var) {
                continue;
            }

            for &polarity in &[false, true] {
                let probe_lit = if polarity {
                    Lit::pos(var)
                } else {
                    Lit::neg(var)
                };

                // Save trail state
                let saved_level = trail.decision_level();

                // Try to assign the literal
                trail.new_decision_level();
                trail.assign_decision(probe_lit);

                // Propagate and check for conflict
                let conflict = self.propagate_probe(&mut trail, &watches, clauses);

                // Backtrack
                trail.backtrack_to(saved_level);

                if conflict {
                    // Found a failed literal! Add its negation as a unit clause
                    failed_literals.push(probe_lit.negate());
                    found += 1;
                    break;
                }
            }
        }

        // Add all failed literals as unit clauses
        for lit in failed_literals {
            clauses.add_original([lit]);
        }

        found
    }

    /// Helper for propagating during probing
    fn propagate_probe(
        &self,
        trail: &mut crate::trail::Trail,
        watches: &crate::watched::WatchLists,
        clauses: &ClauseDatabase,
    ) -> bool {
        use crate::literal::LBool;

        while let Some(lit) = trail.next_to_propagate() {
            let watch_list = watches.get(lit);

            for &watcher in watch_list {
                let clause_id = watcher.clause;
                let blocker = watcher.blocker;

                // Check blocker literal
                if trail.lit_value(blocker) == LBool::True {
                    continue;
                }

                let clause = match clauses.get(clause_id) {
                    Some(c) if !c.deleted => c,
                    _ => continue,
                };

                // Find the two watched literals
                let mut first = clause.lits[0];
                let mut second = clause.lits[1];

                if first == lit.negate() {
                    core::mem::swap(&mut first, &mut second);
                }

                // Try to find a new watch
                let mut found_new_watch = false;
                for &other_lit in &clause.lits[2..] {
                    if trail.lit_value(other_lit) != LBool::False {
                        // Found a new watch - would need to update watches but we're read-only
                        found_new_watch = true;
                        break;
                    }
                }

                if !found_new_watch {
                    // Check if other watch is false (conflict)
                    if trail.lit_value(first) == LBool::False {
                        return true; // Conflict found
                    }

                    // Unit propagation
                    if !trail.is_assigned(first.var()) {
                        trail.assign_propagation(first, clause_id);
                    }
                }
            }
        }

        false // No conflict
    }

    /// Bounded Variable Addition (BVA)
    ///
    /// Introduce new variables to simplify formulas by factoring out common literals.
    /// For example, if we have clauses (a ∨ b ∨ c) and (a ∨ b ∨ d), we can replace them with:
    /// (a ∨ b ∨ x), (~x ∨ c), (~x ∨ d)
    /// where x is a fresh variable. This can reduce total clause size and improve solving.
    ///
    /// Returns the number of variables added.
    #[allow(dead_code)]
    pub fn bounded_variable_addition(
        &mut self,
        clauses: &mut ClauseDatabase,
        max_vars_to_add: usize,
    ) -> usize {
        let mut vars_added = 0;
        self.build_occurrences(clauses);

        // Collect all clause pairs with sufficient overlap
        let clause_ids: Vec<_> = (0..clauses.len())
            .map(|i| ClauseId::new(i as u32))
            .filter(|&id| {
                clauses
                    .get(id)
                    .is_some_and(|c| !c.deleted && !c.learned && c.lits.len() >= 3)
            })
            .collect();

        for i in 0..clause_ids.len() {
            if vars_added >= max_vars_to_add {
                break;
            }

            let clause1_id = clause_ids[i];
            let clause1_lits = match clauses.get(clause1_id) {
                Some(c) if !c.deleted => c.lits.clone(),
                _ => continue,
            };

            for &clause2_id in &clause_ids[(i + 1)..] {
                if vars_added >= max_vars_to_add {
                    break;
                }

                let clause2_lits = match clauses.get(clause2_id) {
                    Some(c) if !c.deleted => c.lits.clone(),
                    _ => continue,
                };

                // Find common literals
                let common: Vec<Lit> = clause1_lits
                    .iter()
                    .filter(|&lit| clause2_lits.contains(lit))
                    .copied()
                    .collect();

                // Only apply BVA if we have at least 2 common literals
                if common.len() < 2 {
                    continue;
                }

                // Check if it's beneficial (reduces total clause size)
                let unique1: Vec<Lit> = clause1_lits
                    .iter()
                    .filter(|lit| !common.contains(lit))
                    .copied()
                    .collect();

                let unique2: Vec<Lit> = clause2_lits
                    .iter()
                    .filter(|lit| !common.contains(lit))
                    .copied()
                    .collect();

                // Original size: |clause1| + |clause2|
                let original_size = clause1_lits.len() + clause2_lits.len();

                // New size: |common| + 1 + |unique1| + 1 + |unique2| + 1
                // = |common| + |unique1| + |unique2| + 3
                let new_size = common.len() + unique1.len() + unique2.len() + 3;

                // Only add variable if it reduces total size
                if new_size >= original_size {
                    continue;
                }

                // Create a new variable
                let new_var = Var::new((self.num_vars + vars_added) as u32);
                let new_lit = Lit::pos(new_var);

                // Create new clauses:
                // 1. (common literals) ∨ new_var
                // 2. ~new_var ∨ (unique literals from clause1)
                // 3. ~new_var ∨ (unique literals from clause2)

                let mut new_clause1 = common.clone();
                new_clause1.push(new_lit);

                let mut new_clause2 = vec![new_lit.negate()];
                new_clause2.extend(&unique1);

                let mut new_clause3 = vec![new_lit.negate()];
                new_clause3.extend(&unique2);

                // Remove old clauses
                if let Some(c) = clauses.get_mut(clause1_id) {
                    c.deleted = true;
                }
                if let Some(c) = clauses.get_mut(clause2_id) {
                    c.deleted = true;
                }
                self.removed_clauses.insert(clause1_id);
                self.removed_clauses.insert(clause2_id);

                // Add new clauses
                if !new_clause1.is_empty() {
                    clauses.add_original(new_clause1);
                }
                if !new_clause2.is_empty() {
                    clauses.add_original(new_clause2);
                }
                if !new_clause3.is_empty() {
                    clauses.add_original(new_clause3);
                }

                vars_added += 1;
                break; // Process next clause pair
            }
        }

        // Update num_vars to reflect new variables
        self.num_vars += vars_added;

        vars_added
    }

    /// Apply all preprocessing techniques
    ///
    /// `fixed` is forwarded to [`Self::pure_literal_elimination`]; see there
    /// for why trail-fixed variables must be excluded from the purity check.
    ///
    /// Returns (clauses_eliminated, vars_eliminated)
    pub fn preprocess(&mut self, clauses: &mut ClauseDatabase, fixed: &[bool]) -> (usize, usize) {
        let mut total_clauses = 0;
        let total_vars = 0;

        // Iteratively apply preprocessing until fixpoint
        loop {
            let mut changed = false;

            // Pure literal elimination
            let pure_elim = self.pure_literal_elimination(clauses, fixed);
            if pure_elim > 0 {
                total_clauses += pure_elim;
                changed = true;
            }

            // Subsumption elimination
            let subsumption = self.subsumption_elimination(clauses);
            if subsumption > 0 {
                total_clauses += subsumption;
                changed = true;
            }

            // Blocked clause elimination
            let bce = self.blocked_clause_elimination(clauses);
            if bce > 0 {
                total_clauses += bce;
                changed = true;
            }

            if !changed {
                break;
            }
        }

        (total_clauses, total_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tautology_detection() {
        let v0 = Var(0);
        let l0 = Lit::pos(v0);
        let l0_neg = Lit::neg(v0);

        // Tautology: x0 ∨ ~x0
        assert!(Preprocessor::is_tautology(&[l0, l0_neg]));

        // Not a tautology: x0 ∨ x0
        assert!(!Preprocessor::is_tautology(&[l0, l0]));
    }

    #[test]
    fn test_pure_literal() {
        let mut clauses = ClauseDatabase::new();
        let v0 = Var(0);
        let v1 = Var(1);

        // Add clauses: (x0 ∨ x1), (x0)
        // x0 and x1 are pure (only positive)
        clauses.add_original([Lit::pos(v0), Lit::pos(v1)]);
        clauses.add_original([Lit::pos(v0)]);

        let mut prep = Preprocessor::new(2);
        let eliminated = prep.pure_literal_elimination(&mut clauses, &[]);

        assert_eq!(eliminated, 2);
    }

    // Regression: a variable already fixed on the caller's trail (e.g. via a
    // unit clause consumed straight onto the trail, never stored in
    // `clauses`) must never be treated as "pure" from clause-level
    // occurrences alone -- see the doc comment on `pure_literal_elimination`.
    #[test]
    fn test_pr26_pure_literal_elimination_skips_trail_fixed_variable() {
        let mut clauses = ClauseDatabase::new();
        let v0 = Var(0);
        let v1 = Var(1);

        // (x0 ∨ x1) ∧ (¬x1 ∨ x0): from clause-occurrence bookkeeping alone x0
        // looks "pure positive" (both its occurrences here are positive) and
        // x1 is genuinely mixed (not pure), so absent the trail exclusion x0
        // would be eliminated. The caller reports x0 as already fixed (e.g.
        // forced false by a unit clause that was consumed onto the trail and
        // never became a stored clause), so it must be skipped entirely.
        clauses.add_original([Lit::pos(v0), Lit::pos(v1)]);
        clauses.add_original([Lit::neg(v1), Lit::pos(v0)]);

        let mut prep = Preprocessor::new(2);
        let fixed = [true, false];
        let eliminated = prep.pure_literal_elimination(&mut clauses, &fixed);

        assert_eq!(
            eliminated, 0,
            "trail-fixed variable must not be eliminated as pure"
        );
        assert!(prep.eliminated_pure_literals().is_empty());
    }

    #[test]
    fn test_subsumption() {
        let mut clauses = ClauseDatabase::new();
        let v0 = Var(0);
        let v1 = Var(1);

        // Add clauses: (x0), (x0 ∨ x1)
        // First clause subsumes second
        clauses.add_original([Lit::pos(v0)]);
        clauses.add_original([Lit::pos(v0), Lit::pos(v1)]);

        let mut prep = Preprocessor::new(2);
        let eliminated = prep.subsumption_elimination(&mut clauses);

        assert_eq!(eliminated, 1);
    }

    /// Collect the ids of every clause the database currently reports deleted.
    fn deleted_ids(clauses: &ClauseDatabase) -> Vec<u32> {
        (0..clauses.len())
            .map(|i| ClauseId::new(i as u32))
            .filter(|&id| clauses.get(id).is_some_and(|c| c.deleted))
            .map(|id| id.0)
            .collect()
    }

    /// Naive O(n^2) reference implementation of forward subsumption with
    /// exactly the semantics [`Preprocessor::subsumption_elimination`]
    /// promises: shortest-first subsumers, and equal-length pairs broken by
    /// clause id so a duplicate pair loses exactly one member. It differs
    /// only in *how* candidates are found — an all-pairs scan instead of the
    /// occurrence lists — which is precisely the property the equivalence
    /// test below pins.
    fn naive_subsumption(clauses: &mut ClauseDatabase) -> usize {
        let mut order: Vec<(usize, u32)> = (0..clauses.len())
            .map(|i| ClauseId::new(i as u32))
            .filter_map(|id| {
                clauses
                    .get(id)
                    .map(|c| (c.lits.len(), id.0, c.deleted, c.learned))
            })
            .filter(|&(_, _, deleted, learned)| !deleted && !learned)
            .map(|(len, id, _, _)| (len, id))
            .collect();
        order.sort_unstable();

        let mut eliminated = 0;
        for &(sub_len, sub_raw) in &order {
            let sub_id = ClauseId::new(sub_raw);
            let sub_lits = match clauses.get(sub_id) {
                Some(c) if !c.deleted && !c.learned => c.lits.clone(),
                _ => continue,
            };
            for &(other_len, other_raw) in &order {
                if other_raw == sub_raw {
                    continue;
                }
                if other_len < sub_len || (other_len == sub_len && other_raw <= sub_raw) {
                    continue;
                }
                let other_id = ClauseId::new(other_raw);
                let subsumed = match clauses.get(other_id) {
                    Some(c) if !c.deleted && !c.learned => {
                        sub_lits.iter().all(|lit| c.lits.contains(lit))
                    }
                    _ => false,
                };
                if subsumed {
                    if let Some(c) = clauses.get_mut(other_id) {
                        c.deleted = true;
                    }
                    eliminated += 1;
                }
            }
        }
        eliminated
    }

    // Defect (a) regression: a *later*, shorter clause must retire an
    // *earlier*, longer one. The previous index-ordered pass only ever tested
    // "earlier subsumes later", so this pair survived intact.
    #[test]
    fn test_issue36_subsumption_is_order_insensitive() {
        let (v0, v1) = (Var(0), Var(1));

        let mut forward = ClauseDatabase::new();
        forward.add_original([Lit::pos(v0)]);
        forward.add_original([Lit::pos(v0), Lit::pos(v1)]);
        let mut prep = Preprocessor::new(2);
        assert_eq!(prep.subsumption_elimination(&mut forward), 1);

        // Same two clauses, reversed insertion order: the short clause is now
        // stored *after* the long one it subsumes.
        let mut reversed = ClauseDatabase::new();
        let long_id = reversed.add_original([Lit::pos(v0), Lit::pos(v1)]);
        let short_id = reversed.add_original([Lit::pos(v0)]);
        let mut prep = Preprocessor::new(2);
        assert_eq!(
            prep.subsumption_elimination(&mut reversed),
            1,
            "a later, shorter clause must subsume an earlier, longer one"
        );
        assert!(
            reversed.get(long_id).is_some_and(|c| c.deleted),
            "the longer clause is the redundant one"
        );
        assert!(
            reversed.get(short_id).is_some_and(|c| !c.deleted),
            "the subsumer itself must survive"
        );
    }

    // Duplicate-safety: two clauses with identical literal sets subsume each
    // other. A "check both directions" fix deletes both and loses the
    // constraint entirely; exactly one representative must survive.
    #[test]
    fn test_issue36_subsumption_keeps_one_of_a_duplicate_pair() {
        let (v0, v1) = (Var(0), Var(1));
        let mut clauses = ClauseDatabase::new();
        let first = clauses.add_original([Lit::pos(v0), Lit::pos(v1)]);
        let second = clauses.add_original([Lit::pos(v0), Lit::pos(v1)]);

        let mut prep = Preprocessor::new(2);
        assert_eq!(prep.subsumption_elimination(&mut clauses), 1);

        let alive = [first, second]
            .into_iter()
            .filter(|&id| clauses.get(id).is_some_and(|c| !c.deleted))
            .count();
        assert_eq!(alive, 1, "exactly one twin of a duplicate pair survives");
    }

    // A permuted superset must still be recognized: `subsumes` here is a set
    // test, and clause literals are reordered in place by watch selection, so
    // nothing may depend on a sorted representation.
    #[test]
    fn test_issue36_subsumption_ignores_literal_order() {
        let (v0, v1, v2) = (Var(0), Var(1), Var(2));
        let mut clauses = ClauseDatabase::new();
        let long_id = clauses.add_original([Lit::pos(v2), Lit::neg(v1), Lit::pos(v0)]);
        clauses.add_original([Lit::pos(v0), Lit::pos(v2)]);

        let mut prep = Preprocessor::new(3);
        assert_eq!(prep.subsumption_elimination(&mut clauses), 1);
        assert!(clauses.get(long_id).is_some_and(|c| c.deleted));
    }

    // Defect (b) regression: the occurrence-accelerated pass must delete
    // exactly the same clause set as the naive all-pairs reference on
    // randomized instances.
    #[test]
    fn test_issue36_occurrence_accelerated_subsumption_matches_naive() {
        // Deterministic LCG (no rand dependency): numerical recipes constants.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize
        };

        let mut total_eliminated = 0usize;
        let mut rounds_with_work = 0usize;

        for round in 0..200 {
            let num_vars = 4 + round % 5;
            let num_clauses = 6 + round % 12;

            let mut generated: Vec<Vec<Lit>> = Vec::new();
            for _ in 0..num_clauses {
                let len = 1 + next() % 4;
                let mut lits: Vec<Lit> = Vec::new();
                for _ in 0..len {
                    let var = Var((next() % num_vars) as u32);
                    let lit = if next() % 2 == 0 {
                        Lit::pos(var)
                    } else {
                        Lit::neg(var)
                    };
                    // Skip duplicates and tautologies: the clause database's
                    // own normalization would reject them anyway, and neither
                    // implementation claims to handle them.
                    if !lits.contains(&lit) && !lits.contains(&lit.negate()) {
                        lits.push(lit);
                    }
                }
                if !lits.is_empty() {
                    generated.push(lits);
                }
            }

            let build = || {
                let mut db = ClauseDatabase::new();
                for lits in &generated {
                    db.add_original(lits.iter().copied());
                }
                db
            };

            let mut accelerated = build();
            let mut prep = Preprocessor::new(num_vars);
            let fast_count = prep.subsumption_elimination(&mut accelerated);

            let mut reference = build();
            let naive_count = naive_subsumption(&mut reference);

            assert_eq!(
                fast_count, naive_count,
                "round {round}: elimination count diverged from the naive reference"
            );
            assert_eq!(
                deleted_ids(&accelerated),
                deleted_ids(&reference),
                "round {round}: deleted clause set diverged from the naive reference"
            );

            total_eliminated += fast_count;
            if fast_count > 0 {
                rounds_with_work += 1;
            }
        }

        // Non-vacuity: two implementations that both delete nothing agree
        // trivially. The generator must actually produce subsumable pairs.
        assert!(
            total_eliminated > 0 && rounds_with_work >= 10,
            "the randomized instances produced almost no subsumptions \
             ({total_eliminated} over {rounds_with_work} rounds); the equivalence \
             above would be vacuous"
        );
    }
}
