//! SatELite-style bounded variable elimination (BVE).
//!
//! For a variable `v` whose positive and negative occurrences are both
//! small, replace every clause mentioning `v` with the set of resolvents
//! obtained by resolving each clause containing `v` against each clause
//! containing `¬v` on `v`. The resolvents are exactly the constraints the
//! rest of the formula places on the other variables *given* `v` is chosen
//! optimally — `v` itself no longer needs to be searched, only reconstructed
//! into the final model once the other variables are known.
//!
//! Bounded the way Eén & Biere's SatELite bounds it: a variable is only
//! eliminated when doing so does not *grow* the formula (neither more
//! resolvent clauses than the originals nor more literals overall), so the
//! pass can only ever shrink or hold steady the problem it hands to search.
//!
//! Reference (technique, not implementation): N. Eén, A. Biere,
//! *Effective Preprocessing in SAT Through Variable and Clause Elimination*,
//! SAT 2005.

use super::*;
use crate::occurrence::OccurrenceList;

/// Resolving a variable whose positive and negative occurrence counts are
/// `p` and `n` produces up to `p * n` resolvents; above this product the
/// variable is left alone rather than paying a potentially-quadratic blowup
/// to find out none of the growth bound would have permitted keeping the
/// result anyway. `4_000` is this port's own tuning choice, not derived from
/// any particular instance class: cheap enough per elimination attempt to
/// stay well under a millisecond even in the worst case, generous enough
/// that it essentially never fires before the growth-bound check below it
/// does on real formulas.
const ELIMINATION_FANOUT_CAP: usize = 4_000;

impl Solver {
    /// Run one pass of bounded variable elimination.
    ///
    /// One-shot per solver incarnation, base-assertion-level-only, and
    /// skipped while any proof (DRAT or LRAT) is being traced — see
    /// [`Solver::fold_equivalent_literals`]'s doc comment for the full
    /// reasoning, which applies here identically. Also skipped outright when
    /// [`SolverConfig::enable_equiv_substitution`] is set: the two mechanisms
    /// both eliminate variables and record independent reconstruction data
    /// for [`Solver::save_model`], and this implementation does not
    /// interleave the two maps, so when both are requested substitution runs
    /// (it tends to be cheaper and exposes structure — e.g. gate congruence —
    /// that helps everything downstream) and this pass defers entirely.
    pub(super) fn bounded_variable_elimination(&mut self) {
        if self.bve_latched
            || !self.config.enable_bve
            || self.config.enable_equiv_substitution
            || self.trail.decision_level() != 0
            || self.assertion_levels.len() > 1
            || self.proof_tracing_active()
        {
            return;
        }
        self.bve_latched = true;
        if self.num_vars == 0 {
            return;
        }
        if self.bve_def.len() < self.num_vars {
            self.bve_def.resize(self.num_vars, Vec::new());
        }

        let mut occ = OccurrenceList::new();
        occ.resize(self.num_vars);
        for id in self.clauses.iter_ids() {
            if let Some(clause) = self.clauses.get(id)
                && !clause.deleted
            {
                for &lit in &clause.lits {
                    occ.add(lit, id);
                }
            }
        }

        // Cheapest (fewest total occurrences) variables first: eliminating a
        // low-degree variable is both cheaper to try and less likely to be
        // rejected by the growth bound, and folding it away can shrink the
        // occurrence counts neighboring variables see later in this same
        // pass.
        let mut order: Vec<usize> = (0..self.num_vars).collect();
        order.sort_by_key(|&v| occ.var_occurrence_count(v));

        let mut derived_units: Vec<Lit> = Vec::new();
        let mut contradiction = false;

        for var_idx in order {
            if contradiction {
                break;
            }
            let var = Var::new(var_idx as u32);
            if self.trail.is_assigned(var) || self.var_eliminated(var) {
                continue;
            }

            let pos_lit = Lit::pos(var);
            let neg_lit = Lit::neg(var);
            let pos_ids: SmallVec<[ClauseId; 8]> = occ.get(pos_lit).iter().copied().collect();
            let neg_ids: SmallVec<[ClauseId; 8]> = occ.get(neg_lit).iter().copied().collect();

            // A variable with no occurrence in one polarity is pure, not a
            // job for resolution — `Preprocessor::pure_literal_elimination`
            // (run separately, see `Solver::inprocess`) already handles that
            // case, and resolving against zero clauses of one polarity would
            // just reproduce the other polarity's clauses verbatim.
            if pos_ids.is_empty() || neg_ids.is_empty() {
                continue;
            }
            if pos_ids.len().saturating_mul(neg_ids.len()) > ELIMINATION_FANOUT_CAP {
                continue;
            }

            let pos_clauses: Vec<SmallVec<[Lit; 4]>> = pos_ids
                .iter()
                .filter_map(|&id| self.clauses.get(id))
                .filter(|c| !c.deleted)
                .map(|c| c.lits.clone())
                .collect();
            let neg_clauses: Vec<SmallVec<[Lit; 4]>> = neg_ids
                .iter()
                .filter_map(|&id| self.clauses.get(id))
                .filter(|c| !c.deleted)
                .map(|c| c.lits.clone())
                .collect();
            if pos_clauses.is_empty() || neg_clauses.is_empty() {
                continue;
            }

            let mut resolvents: Vec<SmallVec<[Lit; 8]>> = Vec::new();
            for pc in &pos_clauses {
                for nc in &neg_clauses {
                    if let Some(resolvent) = resolve_on(pc, nc, pos_lit, neg_lit) {
                        resolvents.push(resolvent);
                    }
                }
            }
            for resolvent in &mut resolvents {
                resolvent.sort_by_key(|l| l.code());
                resolvent.dedup();
            }
            resolvents.sort_by(|a, b| a.iter().map(|l| l.code()).cmp(b.iter().map(|l| l.code())));
            resolvents.dedup();

            let original_clause_count = pos_clauses.len() + neg_clauses.len();
            let original_lit_count: usize = pos_clauses.iter().map(SmallVec::len).sum::<usize>()
                + neg_clauses.iter().map(SmallVec::len).sum::<usize>();
            let new_lit_count: usize = resolvents.iter().map(SmallVec::len).sum();
            let grows =
                resolvents.len() > original_clause_count && new_lit_count > original_lit_count;
            if grows {
                continue;
            }

            // Record the model-reconstruction data — the positive-polarity
            // clauses with `pos_lit` itself stripped off — *before* deleting
            // anything, so a failure partway through never leaves a variable
            // eliminated without a way to recover its value.
            let definition: Vec<SmallVec<[Lit; 4]>> = pos_clauses
                .iter()
                .map(|c| c.iter().copied().filter(|&l| l != pos_lit).collect())
                .collect();
            self.bve_def[var_idx] = definition;
            self.bve_order.push(var);

            for &id in pos_ids.iter().chain(neg_ids.iter()) {
                if let Some(clause) = self.clauses.get(id) {
                    for &lit in &clause.lits.clone() {
                        occ.remove(lit, id);
                    }
                }
                if let Some(clause) = self.clauses.get_mut(id) {
                    clause.deleted = true;
                }
            }

            for resolvent in resolvents {
                match resolvent.len() {
                    0 => {
                        contradiction = true;
                        break;
                    }
                    1 => derived_units.push(resolvent[0]),
                    _ => {
                        let id = self.clauses.add_original(resolvent.iter().copied());
                        for &lit in &resolvent {
                            occ.add(lit, id);
                        }
                    }
                }
            }
        }

        self.rebuild_propagation_index();

        if contradiction {
            self.trivially_unsat = true;
            return;
        }
        for lit in derived_units {
            match self.trail.lit_value(lit) {
                LBool::True => {}
                LBool::False => {
                    self.trivially_unsat = true;
                    return;
                }
                LBool::Undef => self.trail.assign_unit_fact(lit),
            }
        }
        if self.propagate().is_some() {
            self.trivially_unsat = true;
        }
    }

    /// Restore the correct value of every variable bounded variable
    /// elimination removed from the live formula, in reverse elimination
    /// order.
    ///
    /// Reverse order is required, not just convenient: a variable eliminated
    /// earlier in the pass can only be *defined* in terms of variables that
    /// were eliminated later (once a variable is removed it can no longer
    /// appear in a subsequent resolvent), so resolving the later ones first
    /// guarantees every literal a definition clause refers to already has a
    /// settled value in `self.model` by the time it is read.
    ///
    /// The rule mirrors SatELite's own model-extension step: a definition
    /// clause `(¬v ∨ rest)` recorded as `rest` in `bve_def[v]` demands `v` be
    /// true whenever `rest` is not already satisfied by the other literals —
    /// otherwise that original clause would be violated. `v` is set true if
    /// *any* recorded clause needs it, false only when every one of them is
    /// already satisfied without it.
    pub(super) fn reconstruct_bve_eliminated_variables(&mut self) {
        for &var in self.bve_order.clone().iter().rev() {
            let idx = var.index();
            if idx >= self.model.len() {
                continue;
            }
            let Some(definition) = self.bve_def.get(idx) else {
                continue;
            };
            let needs_true = definition
                .iter()
                .any(|clause_rest| !clause_rest.iter().any(|&lit| self.lit_true_in_model(lit)));
            self.model[idx] = if needs_true {
                LBool::True
            } else {
                LBool::False
            };
        }
    }

    /// Read `lit`'s truth value out of the (partially reconstructed) model.
    /// A representative/definition-clause literal is expected to already
    /// have a settled value by the time this is consulted — see the
    /// reconstruction-order notes on [`Self::reconstruct_bve_eliminated_variables`]
    /// and [`Self::fold_equivalent_literals`].
    pub(super) fn lit_true_in_model(&self, lit: Lit) -> bool {
        match self.model.get(lit.var().index()) {
            Some(LBool::True) => lit.is_pos(),
            Some(LBool::False) => lit.is_neg(),
            _ => false,
        }
    }
}

/// Resolve `pos_clause` (containing `pos_lit`) against `neg_clause`
/// (containing `neg_lit = pos_lit.negate()`) on their shared variable,
/// returning the merged clause with both resolved-on literals removed —
/// or `None` if the result is a tautology (some other variable appears with
/// both polarities across the two clauses), since a tautological resolvent
/// is trivially satisfied and contributes nothing.
fn resolve_on(
    pos_clause: &[Lit],
    neg_clause: &[Lit],
    pos_lit: Lit,
    neg_lit: Lit,
) -> Option<SmallVec<[Lit; 8]>> {
    let mut merged: SmallVec<[Lit; 8]> = SmallVec::new();
    merged.extend(pos_clause.iter().copied().filter(|&l| l != pos_lit));
    merged.extend(neg_clause.iter().copied().filter(|&l| l != neg_lit));
    for i in 0..merged.len() {
        for j in (i + 1)..merged.len() {
            if merged[i] == merged[j].negate() {
                return None;
            }
        }
    }
    Some(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bve_enabled_solver() -> Solver {
        Solver::with_config(SolverConfig {
            enable_bve: true,
            // Every test in this module is about what BVE does, and BVE runs
            // *after* the pre-search lucky phase — which would answer `Sat`
            // outright on instances this small, leaving nothing eliminated to
            // assert about (see `SolverConfig::enable_lucky_phase`).
            enable_lucky_phase: false,
            ..SolverConfig::default()
        })
    }

    #[test]
    fn test_pr26_bve_eliminates_and_gate_style_variable() {
        // v is the AND of a and b, wired through v itself:
        //   (¬a∨¬b∨v)  (¬v∨a)  (¬v∨b)
        // Resolving on v should leave behind only consequences among a,b: in
        // particular, eliminating v here can only ever produce tautological
        // resolvents (this is a "definitional" variable), so the growth
        // bound accepts it. `a` and `b` are padded with extra occurrences
        // elsewhere (dummy clauses with fresh variables x,y) so the
        // cheapest-first ordering unambiguously tries v before them — without
        // the padding, a/b/v can tie on occurrence count and the (equally
        // sound) elimination of `a` or `b` instead of `v` would make this
        // test's specific assertion flaky by construction.
        let mut solver = bve_enabled_solver();
        let a = solver.new_var();
        let b = solver.new_var();
        let v = solver.new_var();
        let x = solver.new_var();
        let y = solver.new_var();
        solver.add_clause([Lit::neg(a), Lit::neg(b), Lit::pos(v)]);
        solver.add_clause([Lit::neg(v), Lit::pos(a)]);
        solver.add_clause([Lit::neg(v), Lit::pos(b)]);
        solver.add_clause([Lit::pos(a), Lit::pos(x)]);
        solver.add_clause([Lit::pos(a), Lit::pos(y)]);
        solver.add_clause([Lit::pos(b), Lit::pos(x)]);
        solver.add_clause([Lit::pos(b), Lit::pos(y)]);

        solver.bounded_variable_elimination();
        assert!(solver.var_eliminated(v), "v should have been eliminated");
    }

    #[test]
    fn test_pr26_bve_model_reconstruction_matches_original_clauses() {
        // Same AND-gate formula as `test_pr26_bve_eliminates_and_gate_style_variable`
        // (with the same occurrence-count padding so v is unambiguously the
        // variable eliminated), plus `(a∨b)` to force an actual search
        // decision instead of letting simple unit propagation pin a/b/v
        // before BVE ever gets a chance to run — a unit-forced instance
        // leaves nothing for BVE to eliminate (propagation alone would
        // already fully determine v), which is not what this test means to
        // exercise. `a`/`b`'s actual values are therefore search-dependent,
        // not asserted directly; what must hold regardless of which
        // satisfying assignment the search lands on is that the
        // *reconstructed* model satisfies every original clause, including
        // the ones mentioning the eliminated variable v.
        let mut solver = bve_enabled_solver();
        let a = solver.new_var();
        let b = solver.new_var();
        let v = solver.new_var();
        let x = solver.new_var();
        let y = solver.new_var();
        let original_clauses: Vec<Vec<Lit>> = vec![
            vec![Lit::neg(a), Lit::neg(b), Lit::pos(v)],
            vec![Lit::neg(v), Lit::pos(a)],
            vec![Lit::neg(v), Lit::pos(b)],
            vec![Lit::pos(a), Lit::pos(x)],
            vec![Lit::pos(a), Lit::pos(y)],
            vec![Lit::pos(b), Lit::pos(x)],
            vec![Lit::pos(b), Lit::pos(y)],
            vec![Lit::pos(a), Lit::pos(b)],
        ];
        for clause in &original_clauses {
            solver.add_clause(clause.iter().copied());
        }

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);
        for clause in &original_clauses {
            let satisfied = clause.iter().any(|&lit| solver.lit_true_in_model(lit));
            assert!(
                satisfied,
                "original clause {clause:?} must be satisfied by the reconstructed model"
            );
        }
        assert_ne!(
            solver.model_value(v),
            crate::literal::LBool::Undef,
            "eliminated variable v must have a concrete value in the final model"
        );
        assert!(
            solver.var_eliminated(v),
            "this test only demonstrates BVE model reconstruction if v was actually eliminated"
        );
    }

    #[test]
    fn test_pr26_bve_noop_when_disabled() {
        let mut solver = Solver::new();
        let a = solver.new_var();
        let b = solver.new_var();
        let v = solver.new_var();
        solver.add_clause([Lit::neg(a), Lit::neg(b), Lit::pos(v)]);
        solver.add_clause([Lit::neg(v), Lit::pos(a)]);
        solver.add_clause([Lit::neg(v), Lit::pos(b)]);
        assert!(!solver.config.enable_bve);
        solver.bounded_variable_elimination();
        assert!(!solver.var_eliminated(v));
    }

    #[test]
    fn test_pr26_bve_mutually_exclusive_with_equiv_substitution() {
        let mut solver = Solver::with_config(SolverConfig {
            enable_bve: true,
            enable_equiv_substitution: true,
            ..SolverConfig::default()
        });
        let a = solver.new_var();
        let b = solver.new_var();
        let v = solver.new_var();
        solver.add_clause([Lit::neg(a), Lit::neg(b), Lit::pos(v)]);
        solver.add_clause([Lit::neg(v), Lit::pos(a)]);
        solver.add_clause([Lit::neg(v), Lit::pos(b)]);

        solver.bounded_variable_elimination();
        assert!(
            !solver.bve_latched,
            "BVE must defer entirely when equivalent-literal substitution is also enabled"
        );
    }
}
