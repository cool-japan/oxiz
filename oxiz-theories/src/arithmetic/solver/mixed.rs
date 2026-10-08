//! Mixed-integer arithmetic in a solver that runs over the reals.
//!
//! `ArithSolver` has two modes and no third: `LIA`, where every variable is
//! an integer, and `LRA`, where every variable is a real.  The mode follows
//! the logic (`Solver::set_logic`), and `(set-logic ALL)` — or no logic at
//! all — keeps the default, `LRA`.  An `Int` term in such a solver was solved
//! over the reals: `(declare-const x Int) (assert (= (* 2 x) 1))` answered
//! `sat` on every build (0.3.3, `c4b04b7`, `c702310`; z3: `unsat`), with the
//! model builder printing the numerator of the relaxation's `1/2`.  The
//! model gate did not object — it reads the relaxation too.
//!
//! The solver now knows which of its terms are integers
//! ([`ArithSolver::mark_int_term`], called by the encoder wherever it
//! registers an `Int`-sorted term), and [`ArithSolver::check_integrality`] —
//! run by the theory manager once per final check, after theory combination
//! has reached its fixpoint, never on the checks inside the search — decides
//! a relaxation that gives one of them a non-integral value by the same
//! branch-and-bound `LIA` mode runs, over those terms only: `x ≤ ⌊v⌋` or `x ≥ ⌈v⌉` is valid
//! for an integer, so every branch keeps every integral solution and a leaf
//! at which every marked term is integral is a model of the mixed problem.
//! A strict bound over a row of `Int` terms with integral coefficients is an
//! integer bound (`x < k ⇒ x ≤ ⌈k⌉ − 1`, as `LIA` mode rewrites it); over any
//! other row it keeps its infinitesimal, so a value `k − δ` on an `Int` term
//! counts as non-integral and branches (`find_fractional_int_var`).  An
//! equality whose every term is an integer with integral coefficients gets
//! the `LIA` mode's divisibility check and joins the Diophantine consistency
//! check.  The leaf's snapshot holds every variable — the reals with the
//! leaf's own `δ` instantiated — so the model the solver reports is one
//! assignment, not integers from the leaf beside reals from the relaxation.
//! A solver with no `Int` term interned decides exactly as before.
//!
//! # Deferred integrality (a caller that searches in rounds)
//!
//! A caller whose search runs in rounds that each read the candidate model —
//! the solver's MBQI loop, whose counterexample search reads the arithmetic
//! values of every term — may ask for the integrality to be left to it
//! ([`ArithSolver::set_defer_integrality`]).  The solver then runs exactly as
//! it does with no `Int` term marked: no strict bound is tightened, no
//! equality gets the divisibility check, [`ArithSolver::value`] reports the
//! relaxation's value (the reals' `δ` instantiated, never rounded) and
//! [`ArithSolver::check_integrality`] is the plain check.  Where the caller's
//! search would conclude and [`ArithSolver::first_non_integral_int_term`]
//! names a marked term the relaxation left fractional, the caller takes the
//! deferral back and searches again with the integrality decided here.  The
//! marks changed which candidate a round read (`#P2b-79`'s trajectory cost,
//! re-fix pass 16): `(forall ((i Int)) (distinct i j))` went from `unsat` to
//! `unknown`, the tightened strict bounds and the branch-and-bound leaf
//! sending the chase for `i` down a new value each round.

use super::*;

impl ArithSolver {
    /// Record that `term` has sort `Int`.  In `LIA` mode every variable is
    /// already an integer and this records nothing.
    pub fn mark_int_term(&mut self, term: TermId) {
        if !self.is_integer {
            self.int_terms.insert(term);
        }
    }

    /// Leave the integrality of the marked `Int` terms to the caller (see the
    /// module docs, "Deferred integrality"), or take it back.  Survives
    /// `reset()`, as the marks do.
    pub fn set_defer_integrality(&mut self, defer: bool) {
        self.defer_integrality = defer;
    }

    /// Whether the integrality of the marked `Int` terms is deferred to the
    /// caller ([`ArithSolver::set_defer_integrality`]).
    #[must_use]
    pub fn integrality_deferred(&self) -> bool {
        self.defer_integrality
    }

    /// Whether `term` is an `Int` term this solver itself keeps integral: a
    /// marked term while the integrality is not deferred.
    pub(super) fn rounds_as_integer(&self, term: TermId) -> bool {
        !self.defer_integrality && self.int_terms.contains(&term)
    }

    /// The first marked `Int` term (in variable order) whose value in the
    /// current relaxation is not an integer, with the two bounds of the split
    /// `term ≤ down ∨ term ≥ up` that keeps every integral solution; `None`
    /// when every marked term is integral.  A value `r ± δ` at an integral
    /// `r` (a strict bound the relaxation sits at) is no integer either:
    /// `r − δ` splits to `≤ r − 1` / `≥ r`, `r + δ` to `≤ r` / `≥ r + 1`.
    #[must_use]
    pub fn first_non_integral_int_term(&self) -> Option<(TermId, Rational64, Rational64)> {
        if self.is_integer || self.int_terms.is_empty() {
            return None;
        }
        let mut marked: Vec<(VarId, TermId)> = self
            .int_terms
            .iter()
            .filter_map(|&term| self.term_to_var.get(&term).map(|&var| (var, term)))
            .collect();
        marked.sort_unstable();
        for (var, term) in marked {
            let value = self.simplex.delta_value(var);
            let real = value.real;
            if !real.is_integer() {
                return Some((term, real.floor(), real.ceil()));
            }
            if value.delta.is_negative() {
                return Some((term, real - Rational64::one(), real));
            }
            if value.delta.is_positive() {
                return Some((term, real, real + Rational64::one()));
            }
        }
        None
    }

    /// Whether the current constraints have a solution that gives every
    /// `Int` term an integral value: the relaxation is re-checked, and a
    /// feasible one is decided by branch-and-bound over the `Int` terms,
    /// whose leaf becomes the assignment [`ArithSolver::value`] reports (see
    /// the module docs).  In `LIA` mode `check()` already branches, and this
    /// is `check()`.
    ///
    /// Branch-and-bound is exponential in the worst case, so it runs where
    /// the theory manager asks for it — once per final check, after theory
    /// combination — and never on the checks inside a search, whose
    /// relaxations are only ever partial.
    pub fn check_integrality(&mut self) -> Result<TheoryResult> {
        match self.check()? {
            TheoryResult::Sat if !self.is_integer && !self.defer_integrality => {
                self.mixed_integer_check()
            }
            other => Ok(other),
        }
    }

    /// The branch-and-bound over the `Int` terms of a feasible `LRA`-mode
    /// relaxation: `Sat` (with the leaf snapshot) when there are none.
    fn mixed_integer_check(&mut self) -> Result<TheoryResult> {
        let int_vars = self.interned_int_vars();
        if int_vars.is_empty() {
            return Ok(TheoryResult::Sat);
        }
        // Precondition of `bnb_recurse`: the current assignment is the
        // relaxation's (`check_integrality` just checked it).
        if self.int_equalities_infeasible() {
            return Ok(TheoryResult::Unsat(self.full_unsat_core()));
        }
        let mut nodes: usize = 0;
        self.bnb_recurse(&int_vars, 0, &mut nodes)
    }

    /// The branch-and-bound leaf's assignment of every variable: the integers
    /// as they stand, every other variable with the leaf's `δ` instantiated.
    pub(super) fn snapshot_mixed_model(&mut self) {
        let delta = self.simplex.delta_instantiation();
        let vars: Vec<VarId> = self
            .var_to_term
            .iter()
            .filter_map(|term| self.term_to_var.get(term).copied())
            .collect();
        for var in vars {
            let value = self.simplex.delta_value(var);
            let concrete = if value.delta.is_zero() {
                value.real
            } else {
                value.real + value.delta * delta
            };
            self.lia_model.insert(var, concrete);
        }
    }

    /// Whether `lhs` is an integer in every solution of a mixed (`LRA`-mode)
    /// solver: a non-empty sum of `Int` terms with integral coefficients.
    /// A strict bound on such a row is an integer bound (`assert_lt`,
    /// `assert_gt`), so it carries no infinitesimal for branch-and-bound to
    /// chase.
    pub(super) fn is_integral_row(&self, lhs: &[(TermId, Rational64)]) -> bool {
        !self.is_integer
            && !self.defer_integrality
            && !lhs.is_empty()
            && lhs
                .iter()
                .all(|(term, coeff)| coeff.is_integer() && self.int_terms.contains(term))
    }

    /// For an `LRA`-mode equality whose every term is an `Int` term with an
    /// integral coefficient: `true` (with contradictory bounds installed
    /// under `reason`) when no integer satisfies it on its own, after
    /// recording it for the Diophantine consistency check; `false`, and the
    /// equality is asserted as usual, otherwise.
    pub(super) fn mixed_integer_row_infeasible(
        &mut self,
        lhs: &[(TermId, Rational64)],
        expr: &LinExpr,
        reason: TermId,
    ) -> bool {
        if self.defer_integrality
            || expr.terms.is_empty()
            || !lhs.iter().all(|(term, _)| self.int_terms.contains(term))
            || !expr.terms.iter().all(|(_, coeff)| coeff.is_integer())
        {
            return false;
        }
        let infeasible = if expr.constant.is_integer() {
            let rhs = -*expr.constant.numer();
            let terms: Vec<(VarId, i64)> = expr
                .terms
                .iter()
                .map(|(var, coeff)| (*var, *coeff.numer()))
                .collect();
            let divisor = terms
                .iter()
                .fold(0i64, |acc, &(_, coeff)| gcd_i64(acc, coeff.abs()));
            self.int_equalities.push(IntEquation { terms, rhs });
            divisor > 0 && rhs % divisor != 0
        } else {
            // Integers with integral coefficients sum to an integer.
            true
        };
        if infeasible {
            let reason_id = self.add_reason(reason);
            if let Some(&(var, _)) = expr.terms.first() {
                self.simplex
                    .set_lower(var, Rational64::from_integer(1), reason_id);
                self.simplex
                    .set_upper(var, Rational64::from_integer(0), reason_id);
            }
        }
        infeasible
    }
}
