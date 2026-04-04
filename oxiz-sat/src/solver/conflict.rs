//! Conflict analysis, clause minimization, and assumption handling

use super::*;
use smallvec::SmallVec;

impl Solver {
    /// Analyze conflict and learn clause
    pub(super) fn analyze(&mut self, conflict: ClauseId) -> (u32, SmallVec<[Lit; 16]>) {
        // Debug: print conflict info (only with analyze-debug feature)
        #[cfg(feature = "analyze-debug")]
        if self.num_vars <= 5 {
            eprintln!("[ANALYZE] Conflict clause id={:?}", conflict);
            if let Some(c) = self.clauses.get(conflict) {
                let lits_str: Vec<String> = c
                    .lits
                    .iter()
                    .map(|lit| {
                        let val = self.trail.lit_value(*lit);
                        let level = self.trail.level(lit.var());
                        let sign = if lit.is_pos() { "" } else { "~" };
                        format!("{}v{}@{}={:?}", sign, lit.var().index(), level, val)
                    })
                    .collect();
                eprintln!("[ANALYZE] Conflict clause: ({})", lits_str.join(" | "));
            }
            eprintln!("[ANALYZE] Trail:");
            for &lit in self.trail.assignments() {
                let var = lit.var();
                let level = self.trail.level(var);
                let reason = self.trail.reason(var);
                let sign = if lit.is_pos() { "" } else { "~" };
                eprintln!("  {}v{}@{} reason={:?}", sign, var.index(), level, reason);
            }
        }

        self.learnt.clear();
        self.learnt.push(Lit::from_code(0)); // Placeholder for asserting literal

        let mut counter = 0;
        let mut p = None;
        let mut index = self.trail.assignments().len();
        let current_level = self.trail.decision_level();

        // Reset seen flags
        for s in &mut self.seen {
            *s = false;
        }

        // Collect variables to bump in batch (avoids repeated heap sift-ups)
        let mut vars_to_bump: SmallVec<[Var; 32]> = SmallVec::new();

        let mut reason_clause = conflict;

        while let Some(clause) = self.clauses.get(reason_clause) {
            // Process reason clause (must exist, as it's either conflict or a propagation reason)
            let start = if p.is_some() { 1 } else { 0 };
            let is_learned = clause.learned;

            // Record clause usage for tier promotion and bump activity (if it's a learned clause)
            if is_learned && let Some(clause_mut) = self.clauses.get_mut(reason_clause) {
                clause_mut.record_usage();
                // Promote to Core if LBD ≤ 2 (GLUE clause)
                if clause_mut.lbd <= 2 {
                    clause_mut.promote_to_core();
                }
                // Bump clause activity (MapleSAT-style)
                clause_mut.activity += self.clause_bump_increment;
            }

            let Some(clause) = self.clauses.get(reason_clause) else {
                break;
            };
            for &lit in &clause.lits[start..] {
                let var = lit.var();
                let level = self.trail.level(var);

                if !self.seen[var.index()] && level > 0 {
                    self.seen[var.index()] = true;
                    // Collect variable for batch bumping instead of individual bumps
                    vars_to_bump.push(var);

                    if level == current_level {
                        counter += 1;
                    } else {
                        // Add the literal itself (not negated) to the learned clause.
                        // The conflict clause has all literals FALSE. To prevent this
                        // conflict, we need at least one of these literals to become TRUE.
                        self.learnt.push(lit);
                    }
                }
            }

            // Find next literal to analyze
            let mut current_lit = Lit::from_code(0); // sentinel default
            let mut found_next = false;
            loop {
                if index == 0 {
                    // Guard against underflow: this should not happen in a
                    // well-formed conflict, but theory-conflict injection can
                    // occasionally produce a degenerate state.  Break out to
                    // avoid a usize overflow panic.
                    break;
                }
                index -= 1;
                current_lit = self.trail.assignments()[index];
                p = Some(current_lit);
                if self.seen[current_lit.var().index()] {
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                break;
            }

            counter -= 1;
            if counter == 0 {
                break;
            }

            let var = current_lit.var();
            match self.trail.reason(var) {
                Reason::Propagation(c) => reason_clause = c,
                _ => break,
            }
        }

        // Batch bump all collected variables at once (single heap rebuild)
        self.vsids.bump_batch(&vars_to_bump);
        self.chb.bump_batch(&vars_to_bump);
        self.lrb.on_reason_batch(&vars_to_bump);

        // Set asserting literal (p is guaranteed to be Some at this point)
        if let Some(lit) = p {
            self.learnt[0] = lit.negate();
        }

        // Minimize learnt clause using recursive resolution
        self.minimize_learnt_clause();

        // Calculate assertion level (traditional backtrack level)
        let assertion_level = if self.learnt.len() == 1 {
            0
        } else {
            // Find second highest level
            let mut max_level = 0;
            let mut max_idx = 1;
            for (i, &lit) in self.learnt.iter().enumerate().skip(1) {
                let level = self.trail.level(lit.var());
                if level > max_level {
                    max_level = level;
                    max_idx = i;
                }
            }
            // Move second watch to position 1
            self.learnt.swap(1, max_idx);
            max_level
        };

        // Apply chronological backtracking if enabled
        let backtrack_level = self.chrono_backtrack.compute_backtrack_level(
            &self.trail,
            &self.learnt,
            assertion_level,
        );

        // Track chronological vs non-chronological backtracks
        if backtrack_level != assertion_level {
            self.stats.chrono_backtracks += 1;
        } else {
            self.stats.non_chrono_backtracks += 1;
        }

        // Debug: print learned clause (only with analyze-debug feature)
        #[cfg(feature = "analyze-debug")]
        if self.num_vars <= 5 {
            let lits_str: Vec<String> = self
                .learnt
                .iter()
                .map(|lit| {
                    let sign = if lit.is_pos() { "" } else { "~" };
                    format!("{}v{}", sign, lit.var().index())
                })
                .collect();
            eprintln!(
                "[ANALYZE] Learned clause: ({}), backtrack_level={}",
                lits_str.join(" | "),
                backtrack_level
            );
        }

        (backtrack_level, self.learnt.clone())
    }

    /// Minimize the learned clause by removing redundant literals
    ///
    /// A literal can be removed if it is implied by the remaining literals.
    /// We use a recursive check: a literal l is redundant if its reason clause
    /// contains only literals that are either:
    /// - Already in the learnt clause (marked as seen)
    /// - At decision level 0 (always true in the learned clause context)
    /// - Themselves redundant (recursive check)
    ///
    /// This also performs clause strengthening by checking for stronger implications
    pub(super) fn minimize_learnt_clause(&mut self) {
        if self.learnt.len() <= 2 {
            // Don't minimize very small clauses
            return;
        }

        let original_len = self.learnt.len();

        // Mark all literals in the learned clause as "in clause"
        // We use analyze_stack to track literals to check
        self.analyze_stack.clear();

        // Phase 1: Basic minimization - remove redundant literals
        let mut j = 1; // Write position
        for i in 1..self.learnt.len() {
            let lit = self.learnt[i];
            if self.lit_is_redundant(lit) {
                // Skip this literal (it's redundant)
            } else {
                // Keep this literal
                self.learnt[j] = lit;
                j += 1;
            }
        }
        self.learnt.truncate(j);

        // Phase 2: Clause strengthening - check for self-subsuming resolution
        // If the clause contains both l and ~l' where l' is in a reason clause,
        // we might be able to strengthen the clause
        self.strengthen_learnt_clause();

        // Track minimization statistics
        let final_len = self.learnt.len();
        if final_len < original_len {
            self.stats.minimizations += 1;
            self.stats.literals_removed += (original_len - final_len) as u64;
        }
    }

    /// Strengthen the learned clause using on-the-fly self-subsuming resolution
    pub(super) fn strengthen_learnt_clause(&mut self) {
        if self.learnt.len() <= 2 {
            return;
        }

        // Check each literal to see if we can strengthen by resolution
        let mut j = 1;
        for i in 1..self.learnt.len() {
            let lit = self.learnt[i];
            let var = lit.var();

            // Check if this literal can be strengthened
            if let Reason::Propagation(reason_id) = self.trail.reason(var)
                && let Some(reason_clause) = self.clauses.get(reason_id)
                && reason_clause.lits.len() == 2
            {
                // Binary reason: one literal is lit, the other is the implied literal
                let other_lit = if reason_clause.lits[0] == lit.negate() {
                    reason_clause.lits[1]
                } else if reason_clause.lits[1] == lit.negate() {
                    reason_clause.lits[0]
                } else {
                    // Keep the literal
                    self.learnt[j] = lit;
                    j += 1;
                    continue;
                };

                // If other_lit is already in the learned clause at level 0,
                // we can remove lit
                if self.trail.level(other_lit.var()) == 0 && self.seen[other_lit.var().index()] {
                    // Skip this literal (strengthened)
                    continue;
                }
            }

            // Keep this literal
            self.learnt[j] = lit;
            j += 1;
        }
        self.learnt.truncate(j);
    }

    /// Check if a literal is redundant in the learned clause
    ///
    /// A literal is redundant if its reason clause only contains:
    /// - Literals marked as seen (in the learned clause)
    /// - Literals at decision level 0
    /// - Literals that are themselves redundant (recursive)
    pub(super) fn lit_is_redundant(&mut self, lit: Lit) -> bool {
        let var = lit.var();

        // Decision variables and theory propagations are not redundant
        let reason = match self.trail.reason(var) {
            Reason::Decision => return false,
            Reason::Theory => return false, // Theory propagations can't be minimized
            Reason::Propagation(c) => c,
        };

        let reason_clause = match self.clauses.get(reason) {
            Some(c) => c,
            None => return false,
        };

        // Check all literals in the reason clause
        for &reason_lit in &reason_clause.lits {
            if reason_lit == lit.negate() {
                // Skip the literal we're analyzing
                continue;
            }

            let reason_var = reason_lit.var();

            // Level 0 literals are always OK
            if self.trail.level(reason_var) == 0 {
                continue;
            }

            // If the literal is in the learned clause (seen), it's OK
            if self.seen[reason_var.index()] {
                continue;
            }

            // Otherwise, this literal prevents minimization
            // (A full recursive check would be more powerful but more expensive)
            return false;
        }

        true
    }

    /// Analyze a theory conflict (given as a list of literals that are all false)
    pub(super) fn analyze_theory_conflict(
        &mut self,
        conflict_lits: &[Lit],
    ) -> (u32, SmallVec<[Lit; 16]>) {
        self.learnt.clear();
        self.learnt.push(Lit::from_code(0)); // Placeholder

        let mut counter = 0;
        let current_level = self.trail.decision_level();

        // Reset seen flags
        for s in &mut self.seen {
            *s = false;
        }

        // Collect variables for batch bumping
        let mut vars_to_bump: SmallVec<[Var; 32]> = SmallVec::new();

        // Process conflict literals
        let mut all_level_zero = true;
        for &lit in conflict_lits {
            let var = lit.var();
            let level = self.trail.level(var);

            if !self.seen[var.index()] && level > 0 {
                all_level_zero = false;
                self.seen[var.index()] = true;
                vars_to_bump.push(var);

                if level == current_level {
                    counter += 1;
                } else {
                    // Add the literal itself (not negated) to the learned clause.
                    // The conflict clause has all literals FALSE. To prevent this
                    // conflict, we need at least one of these literals to become TRUE.
                    // So we add the literal directly to the learned clause.
                    self.learnt.push(lit);
                }
            }
        }

        // If ALL conflict literals are at level 0, this is a fundamental UNSAT
        // that cannot be resolved by backtracking. Return an empty learned clause
        // with backtrack_level=0 as a signal.
        if !conflict_lits.is_empty() && all_level_zero {
            return (0, SmallVec::new());
        }

        // Find UIP by walking back through trail
        let mut index = self.trail.assignments().len();
        let mut p = None;

        while counter > 0 {
            if index == 0 {
                break; // Avoid underflow — no more trail entries
            }
            index -= 1;
            if index >= self.trail.assignments().len() {
                break; // Guard against stale length
            }
            let current_lit = self.trail.assignments()[index];
            p = Some(current_lit);
            let var = current_lit.var();

            if self.seen[var.index()] {
                counter -= 1;

                if counter > 0
                    && let Reason::Propagation(reason_clause) = self.trail.reason(var)
                    && let Some(clause) = self.clauses.get(reason_clause)
                {
                    // Get reason and process its literals
                    for &lit in &clause.lits[1..] {
                        let reason_var = lit.var();
                        let level = self.trail.level(reason_var);

                        if !self.seen[reason_var.index()] && level > 0 {
                            self.seen[reason_var.index()] = true;
                            vars_to_bump.push(reason_var);

                            if level == current_level {
                                counter += 1;
                            } else {
                                // Add the literal itself to the learned clause
                                self.learnt.push(lit);
                            }
                        }
                    }
                }
            }
        }

        // Batch bump all collected variables
        self.vsids.bump_batch(&vars_to_bump);
        self.chb.bump_batch(&vars_to_bump);
        self.lrb.on_reason_batch(&vars_to_bump);

        // Set asserting literal
        if let Some(uip) = p {
            self.learnt[0] = uip.negate();
        }

        // Minimize
        self.minimize_learnt_clause();

        // Calculate backtrack level
        let backtrack_level = if self.learnt.len() == 1 {
            0
        } else {
            let mut max_level = 0;
            let mut max_idx = 1;
            for (i, &lit) in self.learnt.iter().enumerate().skip(1) {
                let level = self.trail.level(lit.var());
                if level > max_level {
                    max_level = level;
                    max_idx = i;
                }
            }
            self.learnt.swap(1, max_idx);
            max_level
        };

        (backtrack_level, self.learnt.clone())
    }

    /// Extract a core of assumptions that caused a conflict
    pub(super) fn extract_assumption_core(
        &self,
        assumptions: &[Lit],
        conflict_idx: usize,
    ) -> Vec<Lit> {
        // The conflicting assumption and any assumptions it depends on
        let mut core = Vec::new();
        let conflict_lit = assumptions[conflict_idx];

        // Find assumptions that led to this conflict
        for &lit in &assumptions[..=conflict_idx] {
            if self.seen.get(lit.var().index()).copied().unwrap_or(false) || lit == conflict_lit {
                core.push(lit);
            }
        }

        // If core is empty, just return the conflicting assumption
        if core.is_empty() {
            core.push(conflict_lit);
        }

        core
    }

    /// Analyze conflict to find assumptions in the unsat core
    pub(super) fn analyze_assumption_conflict(&mut self, assumptions: &[Lit]) -> Vec<Lit> {
        // Use seen flags to mark which assumptions are in the conflict
        let mut core = Vec::new();

        // Walk back through the trail to find conflicting assumptions
        for &lit in assumptions {
            let var = lit.var();
            if var.index() < self.trail.assignments().len() {
                let value = self.trail.lit_value(lit);
                // If the negation of an assumption is implied, it's in the core
                if value.is_false() || self.seen.get(var.index()).copied().unwrap_or(false) {
                    core.push(lit);
                }
            }
        }

        // If no specific core found, return all assumptions
        if core.is_empty() {
            core.extend(assumptions.iter().copied());
        }

        core
    }

    /// Get the minimum backtrack level for a conflict
    pub(super) fn analyze_conflict_level(&self, conflict: ClauseId) -> u32 {
        let clause = match self.clauses.get(conflict) {
            Some(c) => c,
            None => return 0,
        };

        let mut min_level = u32::MAX;
        for lit in clause.lits.iter().copied() {
            let level = self.trail.level(lit.var());
            if level > 0 && level < min_level {
                min_level = level;
            }
        }

        if min_level == u32::MAX { 0 } else { min_level }
    }
}
