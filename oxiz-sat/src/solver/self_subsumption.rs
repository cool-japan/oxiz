//! Self-subsuming resolution (clause strengthening) for the inprocessing path.
//!
//! If a clause `C` contains `¬l` and `C \ {¬l} ⊆ D \ {l}` for a clause `D`
//! containing `l`, then resolving `C` and `D` on that variable yields exactly
//! `D \ {l}` — a clause that subsumes `D`. `D` may therefore be *strengthened*
//! in place by dropping `l`. The result is strictly stronger than plain
//! subsumption: it shrinks clauses that no clause in the formula subsumes.
//!
//! Reference (technique, not implementation): N. Eén, A. Biere, *Effective
//! Preprocessing in SAT Through Variable and Clause Elimination*, SAT 2005;
//! the "self-subsuming resolution" rule.
//!
//! ## Where this sits
//!
//! The check itself has lived in this crate for a long time in two places —
//! [`crate::clause::Clause::self_subsuming_resolvent`] and
//! [`crate::dynamic_subsumption::DynamicSubsumption`] — but neither was ever
//! reached from a solve: `DynamicSubsumption` is exported and never
//! constructed. This module is the wired production path. It reuses
//! `Clause::self_subsuming_resolvent` for the predicate so there is exactly
//! one definition of the rule in the crate, and drives it from occurrence
//! lists so the candidate set stays linear in the occurrences of one literal
//! rather than quadratic in the database.
//!
//! ## Constraints this pass respects
//!
//! * **Proof tracing.** Every strengthening goes through
//!   [`Solver::remove_literal_and_rewatch`], which emits the DRAT
//!   `add(shortened)` + `delete(original)` pair and rebuilds the clause's two
//!   watches. Mutating `Clause::lits` directly and rebuilding the whole
//!   propagation index afterwards would be *silently* wrong for DRAT:
//!   `Solver::inprocess`'s pre-pass literal snapshot only detects clauses that
//!   became `deleted`, so an in-place shrink would never reach the proof. LRAT
//!   is out of scope here for the same reason the rest of inprocessing is —
//!   `inprocess` returns early whenever LRAT tracing is active.
//!
//! * **Reason clauses.** A clause that is currently the antecedent of a
//!   level-0 trail assignment is never strengthened. Dropping the propagated
//!   literal itself from such a clause would leave `Solver::analyze` resolving
//!   against a clause that no longer mentions the literal it is resolving out.
//!   The whole clause is skipped rather than only that one literal: cheaper to
//!   check, and level-0 reason clauses are a small set.
//!
//! * **No new binaries.** `remove_literal_and_rewatch` requires more than two
//!   literals and only maintains the watch lists; a ternary shrunk to a binary
//!   would additionally need an entry in the solver's binary implication
//!   graph. This pass therefore registers that edge itself when a
//!   strengthening produces a binary clause, keeping the graph consistent with
//!   the live clause set.

use super::*;

/// Clauses longer than this are neither strengthened nor used as strengtheners.
///
/// Self-subsuming resolution is quadratic in clause length in the worst case
/// and the payoff falls off fast: long clauses are rarely the ones that gate
/// propagation. Mirrors the default cap
/// `crate::dynamic_subsumption::SubsumptionConfig::max_clause_size` so the two
/// formulations of the rule agree on what they consider worth examining.
const MAX_CLAUSE_SIZE: usize = 20;

/// Upper bound on literals removed in a single pass, so one inprocessing stop
/// cannot dominate the solve on a database with very many strengthenable
/// clauses. The next inprocessing stop picks up where this one left off.
const MAX_STRENGTHENINGS_PER_PASS: usize = 256;

/// Upper bound on candidate strengtheners examined for one literal of one
/// clause. Occurrence lists for a frequently-used literal can be enormous;
/// beyond this many candidates the expected yield no longer pays for the scan.
const MAX_CANDIDATES_PER_LITERAL: usize = 64;

impl Solver {
    /// Run one self-subsuming-resolution sweep over the live clause database.
    ///
    /// Returns the number of literals removed. Callable only at decision
    /// level 0 (the caller, [`Solver::inprocess`], already guarantees this).
    pub(super) fn self_subsuming_resolution(&mut self) -> usize {
        if !self.config.enable_self_subsumption
            || self.trail.decision_level() != 0
            || self.lrat.is_some()
        {
            return 0;
        }

        // Antecedents of the current level-0 trail: off limits (see module doc).
        let mut reason_clauses: HashSet<ClauseId> = HashSet::default();
        for &lit in self.trail.assignments() {
            if let Reason::Propagation(id) = self.trail.reason(lit.var()) {
                reason_clauses.insert(id);
            }
        }

        let mut occ = crate::occurrence::OccurrenceList::new();
        occ.resize(self.num_vars);
        let mut targets: Vec<ClauseId> = Vec::new();
        for id in self.clauses.iter_ids() {
            let Some(clause) = self.clauses.get(id) else {
                continue;
            };
            if clause.deleted || clause.lits.len() > MAX_CLAUSE_SIZE {
                continue;
            }
            for &lit in &clause.lits {
                occ.add(lit, id);
            }
            if clause.lits.len() > 2 && !reason_clauses.contains(&id) {
                targets.push(id);
            }
        }

        // Collect first, apply second: `remove_literal_and_rewatch` mutates
        // both the clause and the watch lists, which would invalidate the
        // occurrence lists mid-scan. Every collected action is re-verified
        // against the *live* clauses immediately before it is applied, so a
        // strengthener that was itself strengthened (or deleted) in the
        // meantime can never authorize a removal that no longer holds.
        let mut actions: Vec<(ClauseId, ClauseId, Lit)> = Vec::new();
        'targets: for &target in &targets {
            let Some(target_clause) = self.clauses.get(target) else {
                continue;
            };
            let target_lits = target_clause.lits.clone();

            for &lit in &target_lits {
                let mut examined = 0usize;
                for &cand in occ.get(lit.negate()) {
                    if cand == target {
                        continue;
                    }
                    examined += 1;
                    if examined > MAX_CANDIDATES_PER_LITERAL {
                        break;
                    }
                    if self.strengthens(cand, target) == Some(lit) {
                        actions.push((target, cand, lit));
                        // One strengthening per clause per pass keeps the
                        // collected actions independent of each other on the
                        // target side; further literals of this clause are
                        // reconsidered at the next inprocessing stop.
                        if actions.len() >= MAX_STRENGTHENINGS_PER_PASS {
                            break 'targets;
                        }
                        continue 'targets;
                    }
                }
            }
        }

        let mut removed = 0usize;
        for (target, strengthener, lit) in actions {
            if reason_clauses.contains(&target) {
                continue;
            }
            // Re-verify against the live database (see above).
            if self.strengthens(strengthener, target) != Some(lit) {
                continue;
            }
            let Some(idx) = self
                .clauses
                .get(target)
                .and_then(|c| c.lits.iter().position(|&l| l == lit))
            else {
                continue;
            };
            self.remove_literal_and_rewatch(target, idx);
            removed += 1;

            // A strengthening that lands on exactly two literals produces a
            // binary clause; `remove_literal_and_rewatch` maintains only the
            // watch lists, so register the implication edges here.
            if let Some(clause) = self.clauses.get(target)
                && !clause.deleted
                && clause.lits.len() == 2
            {
                let a = clause.lits[0];
                let b = clause.lits[1];
                self.binary_graph.add(a.negate(), b, target);
                self.binary_graph.add(b.negate(), a, target);
            }
        }

        self.stats.self_subsumed_literals += removed as u64;
        removed
    }

    /// Does live clause `strengthener` self-subsume live clause `target`?
    ///
    /// Returns the literal that may be dropped from `target`, or `None`.
    /// Re-reads both clauses from the database on every call so the answer
    /// always reflects the current contents.
    fn strengthens(&self, strengthener: ClauseId, target: ClauseId) -> Option<Lit> {
        if strengthener == target {
            return None;
        }
        let source = self.clauses.get(strengthener)?;
        let dest = self.clauses.get(target)?;
        if source.deleted || dest.deleted || dest.lits.len() <= 2 {
            return None;
        }
        if source.lits.len() > MAX_CLAUSE_SIZE || dest.lits.len() > MAX_CLAUSE_SIZE {
            return None;
        }
        source.self_subsuming_resolvent(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(var: u32, positive: bool) -> Lit {
        let v = Var::new(var);
        if positive { Lit::pos(v) } else { Lit::neg(v) }
    }

    /// `(x0 ∨ x1 ∨ x2)` resolved against `(x0 ∨ x1 ∨ ¬x2)` on `x2` yields
    /// `(x0 ∨ x1)`, which subsumes both: the pass must shrink one of them.
    #[test]
    fn test_issue36_self_subsumption_strengthens_a_clause() {
        let mut solver = Solver::new();
        solver.ensure_vars(4);
        let long_a = [lit(0, true), lit(1, true), lit(2, true)];
        let long_b = [lit(0, true), lit(1, true), lit(2, false)];
        let _ = solver.add_clause(long_a);
        let _ = solver.add_clause(long_b);
        // Keep the variables alive so nothing else simplifies them away.
        let _ = solver.add_clause([lit(3, true), lit(0, false)]);

        assert!(solver.config.enable_self_subsumption);
        let removed = solver.self_subsuming_resolution();
        assert_eq!(removed, 1, "exactly one of the ternary pair shrinks");

        let shortened = solver
            .clauses
            .iter_ids()
            .filter_map(|id| solver.clauses.get(id))
            .any(|c| !c.deleted && c.lits.len() == 2 && c.lits.contains(&lit(0, true)));
        assert!(shortened, "a two-literal (x0 ∨ x1) clause must now exist");
        assert_eq!(solver.stats.self_subsumed_literals, 1);
    }

    /// A binary produced by strengthening must be reachable through the binary
    /// implication graph, not only through the watch lists.
    #[test]
    fn test_issue36_strengthened_binary_registers_implication_edge() {
        let mut solver = Solver::new();
        solver.ensure_vars(4);
        let _ = solver.add_clause([lit(0, true), lit(1, true), lit(2, true)]);
        let _ = solver.add_clause([lit(0, true), lit(1, true), lit(2, false)]);
        let _ = solver.add_clause([lit(3, true), lit(0, false)]);

        assert_eq!(solver.self_subsuming_resolution(), 1);
        assert!(
            solver
                .binary_graph
                .get(lit(0, false))
                .iter()
                .any(|&(implied, _)| implied == lit(1, true))
                || solver
                    .binary_graph
                    .get(lit(1, false))
                    .iter()
                    .any(|&(implied, _)| implied == lit(0, true)),
            "the new binary clause must appear in the implication graph"
        );
    }

    /// The flag must actually gate the pass.
    #[test]
    fn test_issue36_self_subsumption_respects_config_flag() {
        let mut solver = Solver::with_config(SolverConfig {
            enable_self_subsumption: false,
            ..SolverConfig::default()
        });
        solver.ensure_vars(3);
        let _ = solver.add_clause([lit(0, true), lit(1, true), lit(2, true)]);
        let _ = solver.add_clause([lit(0, true), lit(1, true), lit(2, false)]);
        assert_eq!(solver.self_subsuming_resolution(), 0);
    }

    /// Antecedents of level-0 trail assignments are left alone.
    #[test]
    fn test_issue36_self_subsumption_skips_reason_clauses() {
        let mut solver = Solver::new();
        solver.ensure_vars(4);
        // Force x0 and x1 false at level 0, making (x0 ∨ x1 ∨ x2) the reason
        // for x2 and (x0 ∨ x1 ∨ ¬x2) the reason for ¬x2 (the second one
        // conflicts, but the first is a genuine level-0 antecedent).
        let _ = solver.add_clause([lit(0, false)]);
        let _ = solver.add_clause([lit(1, false)]);
        let _ = solver.add_clause([lit(0, true), lit(1, true), lit(2, true)]);
        let _ = solver.add_clause([lit(0, true), lit(1, true), lit(2, false), lit(3, true)]);
        let _ = solver.propagate();

        let before: Vec<usize> = solver
            .clauses
            .iter_ids()
            .filter_map(|id| solver.clauses.get(id))
            .map(|c| c.lits.len())
            .collect();
        solver.self_subsuming_resolution();
        let after: Vec<usize> = solver
            .clauses
            .iter_ids()
            .filter_map(|id| solver.clauses.get(id))
            .map(|c| c.lits.len())
            .collect();
        // Whatever the pass decides, it must never have shortened a clause
        // that is currently a level-0 antecedent.
        for (idx, (b, a)) in before.iter().zip(after.iter()).enumerate() {
            let id = ClauseId::new(idx as u32);
            let is_reason = solver
                .trail
                .assignments()
                .iter()
                .any(|&l| solver.trail.reason(l.var()) == Reason::Propagation(id));
            if is_reason {
                assert_eq!(b, a, "reason clause {idx} must not be strengthened");
            }
        }
    }
}
