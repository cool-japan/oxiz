//! Propagation methods for the NLSAT solver.
//!
//! Implements boolean constraint propagation (BCP) using the two-watched
//! literal scheme, as well as theory propagation for polynomial constraints.

use super::{AtomId, NlsatSolver};
use crate::assignment::Justification;
use crate::clause::{ClauseId, Watch};
use crate::types::{Atom, AtomKind, Lbool, Literal};
use num_traits::{Signed, Zero};
use oxiz_math::polynomial::Polynomial;
use rustc_hash::FxHashMap;
use std::cmp::Ordering as CmpOrdering;

/// The state of a propagation result.
#[derive(Debug, Clone)]
pub enum PropagationResult {
    /// Propagation succeeded with no conflict.
    Ok,
    /// A conflict was detected.
    Conflict(ClauseId),
    /// Theory conflict (polynomial constraint violation).
    TheoryConflict(Vec<Literal>),
}

impl NlsatSolver {
    /// Perform boolean constraint propagation.
    ///
    /// Two-watched literal scheme:
    /// - Each clause [L1, L2, ...] watches L1 and L2
    /// - Watch for Li is stored in watch list indexed by ~Li
    /// - When ~Li becomes TRUE (i.e., Li becomes FALSE), we're notified
    ///
    pub(super) fn propagate(&mut self) -> PropagationResult {
        while let Some(lit) = self.propagation_queue.pop() {
            self.stats.propagations += 1;

            // `lit` was just assigned TRUE.
            // We need to find clauses where `lit.negate()` is being watched (i.e., was possibly true).
            // These clauses registered their watch in the list indexed by `lit` (since watch is on ~(lit.negate()) = lit).
            //
            // Wait, let me think again:
            // - Clause [L1, L2] puts watch on L1, registered at index ~L1
            // - Clause [L1, L2] puts watch on L2, registered at index ~L2
            // - When literal X becomes TRUE:
            //   - If X = ~L1 (so L1 becomes FALSE), we look at watches indexed by X = ~L1
            //   - These are clauses where L1 was being watched and now L1 is false
            //
            // So `lit` (assigned TRUE) means we look at watches indexed by `lit`.
            // The watched literal that became FALSE is `lit.negate()`.
            let false_lit = lit.negate();
            let watches = self.clauses.watches(lit).to_vec();

            for watch in watches {
                let clause_id = watch.clause_id;

                // Check blocker first (optimization)
                if self.assignment.lit_value(watch.blocker).is_true() {
                    continue;
                }

                // Get the clause literals
                let (lits_len, lit0) = {
                    let clause = match self.clauses.get(clause_id) {
                        Some(c) => c,
                        None => continue,
                    };
                    let lits = clause.literals();
                    (lits.len(), lits[0])
                };

                if lits_len < 2 {
                    // Unit clause - should have been handled during add_clause
                    let first_val = self.assignment.lit_value(lit0);
                    if first_val.is_false() {
                        self.propagation_queue.clear();
                        return PropagationResult::Conflict(clause_id);
                    }
                    continue;
                }

                // We already checked lits_len >= 2, so lit1 must exist
                // (just needed to verify it's not None, no need to use the value)

                // The false literal (false_lit) should be one of lit0 or lit1.
                // We want to ensure the false literal is at position 1 so position 0 is the "other" watch.
                // If lit0 is the false one, swap positions.
                if lit0 == false_lit
                    && let Some(clause) = self.clauses.get_mut(clause_id)
                {
                    clause.swap(0, 1);
                }

                // After potential swap, re-read the literals
                let Some(clause) = self.clauses.get(clause_id) else {
                    continue;
                };
                let lits = clause.literals();
                let (first_lit, second_lit) = (lits[0], lits[1]);

                // Now second_lit should be the one that became false
                // Check if the first literal (the other watched one) is true - clause is satisfied
                let first_val = self.assignment.lit_value(first_lit);
                if first_val.is_true() {
                    // Update blocker
                    self.update_watch_blocker(lit, clause_id, first_lit);
                    continue;
                }

                // Look for a new watch among literals 2..n
                let found_new = self.find_new_watch(clause_id, lit, second_lit);

                if found_new {
                    continue;
                }

                // No new watch found - all other literals are false
                // Check if conflict or unit propagation
                if first_val.is_false() {
                    // All literals are false - conflict!
                    self.propagation_queue.clear();
                    return PropagationResult::Conflict(clause_id);
                }

                // first_val is undef - unit propagation
                self.assignment
                    .assign(first_lit, Justification::Propagation(clause_id));
                self.propagation_queue.push(first_lit);
            }
        }

        PropagationResult::Ok
    }

    /// Update a watch blocker.
    pub(super) fn update_watch_blocker(
        &mut self,
        lit: Literal,
        clause_id: ClauseId,
        new_blocker: Literal,
    ) {
        let watches = self.clauses.watches_mut(lit);
        for w in watches.iter_mut() {
            if w.clause_id == clause_id {
                w.blocker = new_blocker;
                break;
            }
        }
    }

    /// Find a new watch for a clause.
    /// `old_watch_index_lit` is the literal index where the old watch was registered (i.e., `lit` from propagate).
    /// `old_watched_lit` is the actual watched literal that became false.
    pub(super) fn find_new_watch(
        &mut self,
        clause_id: ClauseId,
        old_watch_index_lit: Literal,
        _old_watched_lit: Literal,
    ) -> bool {
        let clause = match self.clauses.get(clause_id) {
            Some(c) => c,
            None => return false,
        };

        let lits_len = clause.len();
        if lits_len <= 2 {
            return false;
        }

        // Find a literal (at position 2 or later) that's not false
        for i in 2..lits_len {
            let Some(lit_i) = clause.get(i) else {
                continue;
            };
            let val = self.assignment.lit_value(lit_i);
            if !val.is_false() {
                // Found a new watch candidate
                let Some(first_lit) = clause.get(0) else {
                    continue;
                };

                // Swap position 1 with position i
                if let Some(clause_mut) = self.clauses.get_mut(clause_id) {
                    clause_mut.swap(1, i);
                }

                // Get the new watched literal (now at position 1)
                let Some(clause_ref) = self.clauses.get(clause_id) else {
                    return false;
                };
                let Some(new_watch_lit) = clause_ref.get(1) else {
                    return false;
                };

                // Remove from old watch list
                self.clauses
                    .watches_mut(old_watch_index_lit)
                    .retain(|w| w.clause_id != clause_id);

                // Add to new watch list (indexed by ~new_watch_lit)
                self.clauses
                    .watches_mut(new_watch_lit.negate())
                    .push(Watch::new(clause_id, first_lit));

                return true;
            }
        }

        false
    }

    /// Perform theory propagation (evaluate polynomial constraints).
    pub(super) fn theory_propagate(&mut self) -> Option<Vec<Literal>> {
        // Track literals for phase saving (to avoid borrow checker issues)
        let mut lits_to_save = Vec::new();

        // Evaluate all atoms under the current arithmetic assignment
        for (id, atom) in self.atoms.iter().enumerate() {
            let atom_id = id as AtomId;

            match atom {
                Atom::Ineq(ineq) => {
                    let bool_var = ineq.bool_var;
                    let current_val = self.assignment.bool_value(bool_var);

                    // Only check if the arithmetic variables are assigned
                    if !self.can_evaluate_atom(atom_id) {
                        continue;
                    }

                    // Evaluate the polynomial
                    let eval_result = self.evaluate_atom(atom_id);

                    match (current_val, eval_result) {
                        (Lbool::True, Lbool::False) | (Lbool::False, Lbool::True) => {
                            // Conflict between boolean assignment and theory evaluation
                            let lit = if current_val.is_true() {
                                Literal::positive(bool_var)
                            } else {
                                Literal::negative(bool_var)
                            };

                            // Return conflict clause explaining why this is unsatisfiable
                            let explanation = self.explain_theory_conflict(atom_id, lit);
                            return Some(explanation);
                        }
                        (Lbool::Undef, result) if !result.is_undef() => {
                            // Theory propagation
                            let lit = if result.is_true() {
                                Literal::positive(bool_var)
                            } else {
                                Literal::negative(bool_var)
                            };
                            self.assignment.assign(lit, Justification::Theory);
                            lits_to_save.push(lit);
                            self.stats.theory_propagations += 1;
                        }
                        _ => {}
                    }
                }
                Atom::Root(root) => {
                    let bool_var = root.bool_var;
                    let current_val = self.assignment.bool_value(bool_var);

                    // Only check if the arithmetic variables are assigned
                    if !self.can_evaluate_atom(atom_id) {
                        continue;
                    }

                    // Evaluate the root atom
                    let eval_result = self.evaluate_root_atom(root);

                    match (current_val, eval_result) {
                        (Lbool::True, Lbool::False) | (Lbool::False, Lbool::True) => {
                            // Conflict between boolean assignment and theory evaluation
                            let lit = if current_val.is_true() {
                                Literal::positive(bool_var)
                            } else {
                                Literal::negative(bool_var)
                            };

                            // Return conflict clause explaining why this is unsatisfiable
                            let explanation = self.explain_theory_conflict(atom_id, lit);
                            return Some(explanation);
                        }
                        (Lbool::Undef, result) if !result.is_undef() => {
                            // Theory propagation
                            let lit = if result.is_true() {
                                Literal::positive(bool_var)
                            } else {
                                Literal::negative(bool_var)
                            };
                            self.assignment.assign(lit, Justification::Theory);
                            lits_to_save.push(lit);
                            self.stats.theory_propagations += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Save phases for all propagated literals
        for lit in lits_to_save {
            self.save_phase(lit);
        }

        None
    }

    /// Check if we can evaluate an atom (all required variables assigned).
    pub(super) fn can_evaluate_atom(&self, atom_id: AtomId) -> bool {
        use oxiz_math::polynomial::NULL_VAR;
        match self.get_atom(atom_id) {
            Some(Atom::Ineq(ineq)) => {
                // Check if all variables in any polynomial factor are assigned
                for factor in &ineq.factors {
                    for var in factor.poly.vars() {
                        if !self.assignment.is_arith_assigned(var) {
                            return false;
                        }
                    }
                }
                true
            }
            Some(Atom::Root(root)) => {
                // For root atoms, we need all variables up to max_var assigned
                let max = root.max_var();
                if max == NULL_VAR {
                    return true;
                }
                for var in 0..=max {
                    if !self.assignment.is_arith_assigned(var) {
                        return false;
                    }
                }
                true
            }
            None => false,
        }
    }

    /// Evaluate an atom under the current assignment.
    pub(super) fn evaluate_atom(&self, atom_id: AtomId) -> Lbool {
        match self.get_atom(atom_id) {
            Some(Atom::Ineq(ineq)) => {
                // Compute signs for each factor
                let mut signs = Vec::with_capacity(ineq.factors.len());

                for factor in &ineq.factors {
                    // Build the evaluation map
                    let mut eval_map = FxHashMap::default();
                    for var in factor.poly.vars() {
                        if let Some(val) = self.assignment.arith_value(var) {
                            eval_map.insert(var, val.clone());
                        } else {
                            return Lbool::Undef;
                        }
                    }

                    // Evaluate polynomial
                    let value = factor.poly.eval(&eval_map);
                    let sign = if value.is_zero() {
                        0
                    } else if value.is_positive() {
                        1
                    } else {
                        -1
                    };
                    signs.push(sign);
                }

                // Use evaluate_sign method
                match ineq.evaluate_sign(&signs) {
                    Some(true) => Lbool::True,
                    Some(false) => Lbool::False,
                    None => Lbool::Undef,
                }
            }
            Some(Atom::Root(root)) => self.evaluate_root_atom(root),
            None => Lbool::Undef,
        }
    }

    /// Evaluate a root atom under the current assignment.
    ///
    /// For a root atom like `x op root[i](p)`, where op is =, <, >, <=, or >=:
    /// 1. Substitute all assigned variables (except x) into p
    /// 2. Isolate the roots of the resulting univariate polynomial
    /// 3. Get the i-th root (1-indexed, sorted in ascending order)
    /// 4. Compare x's value with the root value
    pub(super) fn evaluate_root_atom(&self, root: &crate::types::RootAtom) -> Lbool {
        use crate::cad::SturmSequence;

        // Get the value of the variable x
        let x_val = match self.assignment.arith_value(root.var) {
            Some(v) => v,
            None => return Lbool::Undef,
        };

        // Substitute all assigned variables (except root.var) into the polynomial
        let mut sub_poly = root.poly.clone();
        for var in root.poly.vars() {
            if var != root.var {
                if let Some(val) = self.assignment.arith_value(var) {
                    sub_poly = sub_poly.substitute(var, &Polynomial::constant(val.clone()));
                } else {
                    return Lbool::Undef;
                }
            }
        }

        // Now sub_poly should be univariate in root.var
        if !sub_poly.is_univariate() && !sub_poly.is_constant() {
            return Lbool::Undef;
        }

        // If the polynomial is constant (all roots are gone), we can't satisfy the root atom
        if sub_poly.is_constant() {
            // No roots exist
            return Lbool::False;
        }

        // Use Sturm sequence to isolate roots
        let sturm = SturmSequence::new(&sub_poly, root.var);
        let root_intervals = sturm.isolate_roots();

        // Check if we have enough roots
        if (root.root_index as usize) > root_intervals.len() || root.root_index == 0 {
            // Root index out of bounds (1-indexed)
            return Lbool::False;
        }

        // Get the i-th root interval (root_index is 1-based)
        let (root_lo, root_hi) = &root_intervals[(root.root_index - 1) as usize];

        // Compare x_val with the root
        // The root is in the interval [root_lo, root_hi]
        // For precise comparison, we refine the interval if x_val is within it
        let cmp_lo = x_val.cmp(root_lo);
        let cmp_hi = x_val.cmp(root_hi);

        // Determine the comparison result based on the atom kind
        let result = match root.kind {
            AtomKind::RootEq => {
                // x = root[i](p)
                // This is true only if the interval is a point and x_val equals it
                if root_lo == root_hi && x_val == root_lo {
                    true
                } else if cmp_lo == CmpOrdering::Less || cmp_hi == CmpOrdering::Greater {
                    // x is definitely not equal to the root
                    false
                } else {
                    // x_val is within the isolating interval - we can't determine for sure
                    // In a real implementation, we would refine the interval further
                    return Lbool::Undef;
                }
            }
            AtomKind::RootLt => {
                // x < root[i](p)
                if cmp_hi == CmpOrdering::Less {
                    // x < root_hi, so definitely x < root
                    true
                } else if cmp_lo != CmpOrdering::Less {
                    // x >= root_lo, so definitely x >= root (not less)
                    false
                } else {
                    // x is in [root_lo, root_hi) - unclear
                    return Lbool::Undef;
                }
            }
            AtomKind::RootGt => {
                // x > root[i](p)
                if cmp_lo == CmpOrdering::Greater {
                    // x > root_lo, so definitely x > root
                    true
                } else if cmp_hi != CmpOrdering::Greater {
                    // x <= root_hi, so definitely x <= root (not greater)
                    false
                } else {
                    // x is in (root_lo, root_hi] - unclear
                    return Lbool::Undef;
                }
            }
            AtomKind::RootLe => {
                // x <= root[i](p)
                if cmp_hi != CmpOrdering::Greater {
                    // x <= root_hi, so definitely x <= root
                    true
                } else if cmp_lo == CmpOrdering::Greater {
                    // x > root_lo, so definitely x > root (not <=)
                    false
                } else {
                    return Lbool::Undef;
                }
            }
            AtomKind::RootGe => {
                // x >= root[i](p)
                if cmp_lo != CmpOrdering::Less {
                    // x >= root_lo, so definitely x >= root
                    true
                } else if cmp_hi == CmpOrdering::Less {
                    // x < root_hi, so definitely x < root (not >=)
                    false
                } else {
                    return Lbool::Undef;
                }
            }
            _ => return Lbool::Undef,
        };

        Lbool::from_bool(result)
    }

    /// Explain a theory conflict.
    pub(super) fn explain_theory_conflict(
        &mut self,
        atom_id: AtomId,
        conflicting_lit: Literal,
    ) -> Vec<Literal> {
        let mut explanation = Vec::new();

        // The conflicting literal is part of the explanation
        explanation.push(conflicting_lit.negate());

        // Collect variables involved in conflict first
        let mut conflict_vars = Vec::new();

        // Add arithmetic variable assignments that led to this conflict
        // For each assigned arithmetic variable, find the atoms that constrained it
        if let Some(Atom::Ineq(ineq)) = self.get_atom(atom_id) {
            for factor in &ineq.factors {
                for var in factor.poly.vars() {
                    conflict_vars.push(var);

                    // Find atoms that assigned this variable
                    for (other_id, other_atom) in self.atoms.iter().enumerate() {
                        if other_id == atom_id as usize {
                            continue;
                        }

                        if let Atom::Ineq(other_ineq) = other_atom {
                            let has_var = other_ineq
                                .factors
                                .iter()
                                .any(|f| f.poly.vars().contains(&var));
                            if has_var {
                                let bool_var = other_ineq.bool_var;
                                let val = self.assignment.bool_value(bool_var);
                                if !val.is_undef() {
                                    let lit = if val.is_true() {
                                        Literal::negative(bool_var)
                                    } else {
                                        Literal::positive(bool_var)
                                    };
                                    if !explanation.contains(&lit)
                                        && !explanation.contains(&lit.negate())
                                    {
                                        explanation.push(lit);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Bump activity for variables involved in conflict
        for var in conflict_vars {
            self.bump_arith_activity(var);
        }

        explanation
    }
}
