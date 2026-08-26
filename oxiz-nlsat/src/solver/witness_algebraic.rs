//! Exact real-algebraic witnesses for the NLSAT search.
//!
//! ## The gap this closes
//!
//! `NlsatSolver::pick_arith_value` hands each arithmetic variable a *rational*
//! sample point. For `x² = 2` there is no such point, so the search used to
//! classify the variable as `ArithDecision::IrrationalOnly` and the whole
//! problem came back `unknown` — the true feasible set is `{−√2, √2}`, both of
//! which the assignment simply could not hold. This module lets the search
//! commit to one of them exactly.
//!
//! ## The carrier, and the three types deliberately *not* used
//!
//! The value carrier is [`CadPoint`] — a rational, or a defining polynomial
//! plus an isolating interval and root index, which is exactly Z3's `root-obj`
//! shape. It is already the carrier of `crate::cad`'s sample points, and
//! `CadPoint::try_sign_of` (see `crate::cad_algebraic`) answers *sign of a
//! polynomial at this point* exactly, by a gcd zero-test followed by
//! Sturm-guided interval refinement.
//!
//! Sign-at-a-point is the *only* operation a witness check needs, which is why
//! no field arithmetic on algebraic numbers appears here, and why the three
//! `AlgebraicNumber` types in `oxiz-math` were **rejected** for this role
//! rather than adopted:
//!
//! * they are built on a different `Polynomial` type than the one the NLSAT
//!   atoms carry, so every check would need a lossy conversion;
//! * one of their constructors *panics* unless `num_roots == 1`, which is a
//!   precondition a solver cannot guarantee about a specialized atom;
//! * adopting one would make a third parallel real-algebraic implementation in
//!   this workspace, with `crate::cad` and `crate::cad_algebraic` already
//!   holding the exact sign machinery the solver actually calls.
//!
//! This note exists so nobody later "upgrades" [`CadPoint`] to one of them.
//!
//! ## Why at most one algebraic value at a time
//!
//! `Assignment::eval_poly_sign` can decide the sign of a polynomial over
//! *rational* values, or over rationals plus **one** algebraic value (the
//! rationals substitute away, leaving a univariate polynomial the point can be
//! evaluated against exactly). Two algebraic values leave a genuinely
//! multivariate real-algebraic problem, which this crate cannot solve — and an
//! atom whose sign cannot be decided is an atom that cannot be *checked*.
//!
//! That is not a performance concern, it is a soundness one. With `x = √2` and
//! `y = √3` installed, `x·y = 1` evaluates to "don't know", every atom is
//! Boolean-assigned, `is_complete()` is true, and the solver answers `sat` to a
//! plainly unsatisfiable problem. So [`NlsatSolver::sample_algebraic_witness`]
//! refuses to produce a second algebraic witness, and
//! [`NlsatSolver::algebraic_model_is_verified`] re-checks every assigned atom
//! exactly before any `Sat` that involves an algebraic value is returned.
//!
//! ## Root atoms
//!
//! `x op root[i](p)` atoms have no exact evaluation path at an algebraic point
//! (`NlsatSolver::evaluate_root_atom` reads a rational value and reports
//! `Undef` otherwise), so a solver instance that holds *any* root atom declines
//! algebraic witnesses altogether. Root atoms are only ever created through the
//! public `NlsatSolver::new_root_atom`; nothing inside the search builds one,
//! so this costs the dispatch paths nothing.

use super::NlsatSolver;
use crate::cad::CadPoint;
use crate::cad_algebraic::{isolate_root_samples, open_sample};
use crate::types::{Atom, AtomKind, IneqAtom, Literal};
use oxiz_math::polynomial::{Polynomial, Var};

/// What a search for an algebraic witness for one variable found.
///
/// Three-valued on purpose. An `Option<CadPoint>` cannot distinguish "every
/// candidate cell was *proved* not to satisfy the constraints" from "some
/// check could not be completed", and the caller turns the first into a theory
/// lemma (`ProvedEmpty`) — so conflating them would fabricate `unsat`.
pub(super) enum AlgebraicWitness {
    /// An exact point satisfying every constraint on the variable that is
    /// decidable under the current assignment.
    Found(CadPoint),
    /// Every cell of the sign-invariant decomposition induced by the
    /// variable's *pure* (single-variable) constraints was refuted, with every
    /// sign proved rather than approximated. The attached literals are the
    /// negations of those constraints' current literals: a valid theory lemma.
    NoRealPoint(Vec<Literal>),
    /// Neither established. The caller must answer `Unknown`, never `Unsat`.
    Inconclusive,
}

/// Result of checking one candidate point against the collected atoms.
enum PointVerdict {
    /// Every checked atom holds at the point, each by a proved sign.
    Satisfied,
    /// Some checked atom is refuted at the point, by a proved sign.
    Refuted,
    /// At least one sign could not be proved (refinement budget exhausted, or
    /// an atom shape with no exact evaluation).
    Undecided,
}

/// One Boolean-assigned inequality atom, specialized to the variable being
/// sampled by substituting every other variable's rational value.
struct CheckedAtom<'a> {
    /// The atom itself (for `kind` and `evaluate_sign`'s factor parity rules).
    ineq: &'a IneqAtom,
    /// Specialized factor polynomials, parallel to `ineq.factors`.
    factors: Vec<Polynomial>,
    /// Whether the atom is currently assigned `true`.
    is_true: bool,
    /// The literal to negate when this atom joins a lemma.
    lit: Literal,
    /// Whether the atom mentions no variable other than the sampled one — only
    /// then is its refutation independent of earlier (greedy) choices.
    pure: bool,
}

impl NlsatSolver {
    /// Search for an exact witness for `var` among the cells of the
    /// sign-invariant decomposition its currently-assigned constraints induce.
    ///
    /// The candidate list is one representative per cell — every root of every
    /// specialized constraint polynomial, and one rational strictly inside each
    /// gap between consecutive roots. That makes the search a *decision
    /// procedure* over those constraints: a cell is sign-invariant, so if no
    /// representative satisfies them, no real number does.
    pub(super) fn sample_algebraic_witness(&self, var: Var) -> AlgebraicWitness {
        // One algebraic value at a time — see the module docs. This is the
        // guard that keeps every atom over fully-assigned variables inside
        // `Assignment::eval_poly_sign`'s decidable range.
        if self.assignment.has_algebraic_value() {
            return AlgebraicWitness::Inconclusive;
        }
        // Root atoms have no exact algebraic evaluation path at all.
        if self.atoms.iter().any(|atom| atom.is_root()) {
            return AlgebraicWitness::Inconclusive;
        }

        let Some(checked) = self.collect_checked_atoms(var) else {
            return AlgebraicWitness::Inconclusive;
        };

        // The polynomials whose roots delimit the cells.
        let mut polys: Vec<Polynomial> = Vec::new();
        for atom in &checked {
            for factor in &atom.factors {
                if factor.degree(var) > 0 {
                    polys.push(factor.clone());
                }
            }
        }
        if polys.is_empty() {
            // Nothing decidable constrains `var`; the rational sampler owns
            // that case and there is no algebraic witness to find here.
            return AlgebraicWitness::Inconclusive;
        }

        let roots = isolate_root_samples(&polys, var);

        // Candidates, left to right: the open cell before the first root, then
        // each root and the open cell that follows it.
        let mut candidates: Vec<CadPoint> = Vec::new();
        candidates.push(CadPoint::rational(open_sample(None, roots.first())));
        for (i, root) in roots.iter().enumerate() {
            candidates.push(root.to_point());
            candidates.push(CadPoint::rational(open_sample(
                Some(root),
                roots.get(i + 1),
            )));
        }

        let mut any_undecided = false;
        for candidate in candidates {
            match self.point_verdict(var, &candidate, &checked) {
                PointVerdict::Satisfied => return AlgebraicWitness::Found(candidate),
                PointVerdict::Refuted => {}
                PointVerdict::Undecided => any_undecided = true,
            }
        }

        // Every cell refuted. That is a proof only if every refutation was
        // exact *and* every refuting constraint mentions `var` alone: an atom
        // coupling `var` with an already-assigned variable is refuted only
        // under that variable's greedy value, which is not a global lemma.
        if any_undecided || !checked.iter().all(|atom| atom.pure) {
            return AlgebraicWitness::Inconclusive;
        }
        let lemma: Vec<Literal> = checked.iter().map(|atom| atom.lit.negate()).collect();
        if lemma.is_empty() {
            return AlgebraicWitness::Inconclusive;
        }
        AlgebraicWitness::NoRealPoint(lemma)
    }

    /// Whether `point` satisfies every Boolean-assigned atom that mentions
    /// `var` and is decidable under the current assignment.
    ///
    /// `None` means at least one such atom could not be decided exactly, which
    /// callers must never read as "satisfied".
    pub(super) fn point_satisfies_assigned_atoms(
        &self,
        var: Var,
        point: &CadPoint,
    ) -> Option<bool> {
        if self.atoms.iter().any(|atom| atom.is_root()) {
            return None;
        }
        let checked = self.collect_checked_atoms(var)?;
        match self.point_verdict(var, point, &checked) {
            PointVerdict::Satisfied => Some(true),
            PointVerdict::Refuted => Some(false),
            PointVerdict::Undecided => None,
        }
    }

    /// Collect and specialize the Boolean-assigned inequality atoms that
    /// mention `var`.
    ///
    /// An atom whose *other* variables are not all assigned to rationals is
    /// omitted: it places no constraint on `var` yet (the same reading
    /// `NlsatSolver::compute_arith_regions` takes when its `arith_value(v)?`
    /// yields `None`), and it will be checked once the coupled variable is
    /// assigned. `None` means an atom shape appeared that this cannot handle
    /// at all, so nothing about `var` should be concluded.
    fn collect_checked_atoms(&self, var: Var) -> Option<Vec<CheckedAtom<'_>>> {
        let mut checked = Vec::new();
        for atom in &self.atoms {
            let Atom::Ineq(ineq) = atom else {
                return None; // Root atom: refused wholesale by the callers.
            };
            let value = self.assignment.bool_value(ineq.bool_var);
            if value.is_undef() {
                continue;
            }
            if !ineq
                .factors
                .iter()
                .any(|factor| factor.poly.vars().contains(&var))
            {
                continue; // Does not mention `var`.
            }
            // `IneqAtom::evaluate_sign` panics on a root kind; refuse instead.
            if !matches!(ineq.kind, AtomKind::Eq | AtomKind::Lt | AtomKind::Gt) {
                return None;
            }

            let mut factors = Vec::with_capacity(ineq.factors.len());
            let mut pure = true;
            let mut decidable = true;
            for factor in &ineq.factors {
                let mut specialized = factor.poly.clone();
                for other in factor.poly.vars() {
                    if other == var {
                        continue;
                    }
                    pure = false;
                    let Some(value) = self.assignment.arith_value(other) else {
                        decidable = false;
                        break;
                    };
                    specialized =
                        specialized.substitute(other, &Polynomial::constant(value.clone()));
                }
                if !decidable {
                    break;
                }
                factors.push(specialized);
            }
            if !decidable {
                continue; // Coupled with a variable that has no value yet.
            }

            let is_true = value.is_true();
            checked.push(CheckedAtom {
                ineq,
                factors,
                is_true,
                lit: if is_true {
                    Literal::positive(ineq.bool_var)
                } else {
                    Literal::negative(ineq.bool_var)
                },
                pure,
            });
        }
        Some(checked)
    }

    /// Evaluate every collected atom at `point` (as the value of `var`).
    fn point_verdict(
        &self,
        var: Var,
        point: &CadPoint,
        checked: &[CheckedAtom<'_>],
    ) -> PointVerdict {
        let mut undecided = false;
        for atom in checked {
            let mut signs = Vec::with_capacity(atom.factors.len());
            let mut decided = true;
            for factor in &atom.factors {
                match point.try_sign_of(factor, var) {
                    Some(sign) => signs.push(sign),
                    None => {
                        decided = false;
                        break;
                    }
                }
            }
            if !decided {
                undecided = true;
                continue;
            }
            match atom.ineq.evaluate_sign(&signs) {
                // A refutation is decisive: no later atom can rescue the point.
                Some(holds) if holds != atom.is_true => return PointVerdict::Refuted,
                Some(_) => {}
                None => undecided = true,
            }
        }
        if undecided {
            PointVerdict::Undecided
        } else {
            PointVerdict::Satisfied
        }
    }

    /// Install an exact (possibly algebraic) witness for `var` and note it on
    /// the witness ledger.
    ///
    /// The counterpart of `NlsatSolver::commit_arith_witness`, with one
    /// difference: the ledger entry offers **no** replacement point. The
    /// natural retry for an algebraic witness is a *different root*, not a
    /// rational drawn from the same region — the rationals immediately around
    /// `√2` all violate `x² = 2`, so offering them would burn the retry
    /// allowance on points that are already known to fail. Withdrawal
    /// therefore walks straight past this entry to an earlier, genuinely
    /// chosen one (see `solver/resample.rs`); retrying alternative algebraic
    /// roots is a later phase.
    pub(super) fn commit_arith_witness_point(&mut self, var: Var, point: CadPoint) {
        // The whole feature rests on this: a point is installed only after it
        // has been *proved* to satisfy every constraint on `var` that is
        // decidable now. Re-assert it independently of the search that
        // produced it, so a future change to the candidate walk cannot quietly
        // start installing unchecked points.
        debug_assert_eq!(
            self.point_satisfies_assigned_atoms(var, &point),
            Some(true),
            "an algebraic witness must be proved to satisfy the assigned atoms before it is committed"
        );
        self.assignment.set_arith_point(var, point);
        self.eval_cache.clear();
        self.arith_witnesses.record_algebraic(var);
    }

    /// Final exactness sweep before reporting `Sat` with an algebraic value in
    /// the assignment.
    ///
    /// The rational search path is left untouched: with no algebraic value
    /// installed this is an immediate `true`. Otherwise every Boolean-assigned
    /// atom must be *proved* to agree with its assigned polarity. Anything
    /// undecidable — a refinement budget that ran out, a root atom, an atom
    /// over two algebraic values — fails the sweep and the caller answers
    /// `Unknown`.
    ///
    /// This is the only net under an algebraic `Sat`:
    /// `oxiz_theories::nlsat::dispatch_nra_constraints` trusts this solver's
    /// verdict directly, without the model re-check its integer sibling does.
    pub(super) fn algebraic_model_is_verified(&self) -> bool {
        if !self.assignment.has_algebraic_value() {
            return true;
        }
        for atom in &self.atoms {
            let Atom::Ineq(ineq) = atom else {
                return false; // Root atom: no exact path at an algebraic point.
            };
            let value = self.assignment.bool_value(ineq.bool_var);
            if value.is_undef() {
                continue;
            }
            if !matches!(ineq.kind, AtomKind::Eq | AtomKind::Lt | AtomKind::Gt) {
                return false;
            }
            let mut signs = Vec::with_capacity(ineq.factors.len());
            for factor in &ineq.factors {
                match self.assignment.eval_poly_sign(&factor.poly) {
                    Some(sign) => signs.push(sign),
                    None => return false,
                }
            }
            match ineq.evaluate_sign(&signs) {
                Some(holds) if holds == value.is_true() => {}
                _ => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SolverResult;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    /// `x` as a polynomial over variable index 0.
    fn x_squared_minus_two() -> Polynomial {
        Polynomial::univariate(0, &[rat(-2), rat(0), rat(1)])
    }

    /// Assert `x² = 2` (as a unit clause) on a fresh solver and return it.
    fn solver_with_sqrt2() -> (NlsatSolver, Var) {
        let mut solver = NlsatSolver::new();
        let x = solver.new_arith_var();
        let atom = solver.new_ineq_atom(x_squared_minus_two(), AtomKind::Eq);
        let lit = solver.atom_literal(atom, true);
        solver.add_clause(vec![lit]);
        (solver, x)
    }

    /// `x² = 2` alone is satisfiable, and the witness the search commits to is
    /// an exact algebraic point that really is a root.
    #[test]
    fn sqrt2_alone_is_sat_with_an_algebraic_witness() {
        let (mut solver, x) = solver_with_sqrt2();
        assert_eq!(solver.solve(), SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("a Sat verdict must carry a model");
        let point = model
            .arith_point(x)
            .expect("x must hold an exact point in the model");
        assert!(
            !point.is_rational(),
            "no rational squares to 2, so the witness must be algebraic: {point:?}"
        );
        assert_eq!(
            point.try_sign_of(&x_squared_minus_two(), x),
            Some(0),
            "the witness must be an exact root of x^2 - 2"
        );
        assert!(
            model.arith_value(x).is_none(),
            "an algebraic witness has no rational value to report"
        );
    }

    /// `x² = 2 ∧ x > 0` must pick the *right* root: √2, the second (1-based)
    /// real root of `x² − 2`, whose isolating bracket contains 1.414.
    #[test]
    fn sqrt2_positive_branch() {
        let (mut solver, x) = solver_with_sqrt2();
        // x > 0
        let gt = solver.new_ineq_atom(Polynomial::univariate(0, &[rat(0), rat(1)]), AtomKind::Gt);
        let lit = solver.atom_literal(gt, true);
        solver.add_clause(vec![lit]);

        assert_eq!(solver.solve(), SolverResult::Sat);
        let model = solver.get_model().expect("Sat must carry a model");
        let point = model.arith_point(x).expect("x must be pinned");
        let CadPoint::Algebraic { lo, hi, index, .. } = point else {
            panic!("expected an algebraic witness for +√2, got {point:?}");
        };
        assert_eq!(*index, 2, "+√2 is the second real root of x^2 - 2");
        let approx = BigRational::new(BigInt::from(1414), BigInt::from(1000));
        assert!(
            *lo <= approx && approx <= *hi,
            "the isolating bracket [{lo}, {hi}] must contain 1.414"
        );
        assert_eq!(point.try_sign_of(&x_squared_minus_two(), x), Some(0));
    }

    /// The mirrored branch must pick the *first* root, −√2.
    #[test]
    fn sqrt2_negative_branch() {
        let (mut solver, x) = solver_with_sqrt2();
        // x < 0
        let lt = solver.new_ineq_atom(Polynomial::univariate(0, &[rat(0), rat(1)]), AtomKind::Lt);
        let lit = solver.atom_literal(lt, true);
        solver.add_clause(vec![lit]);

        assert_eq!(solver.solve(), SolverResult::Sat);
        let model = solver.get_model().expect("Sat must carry a model");
        let point = model.arith_point(x).expect("x must be pinned");
        let CadPoint::Algebraic { lo, hi, index, .. } = point else {
            panic!("expected an algebraic witness for -√2, got {point:?}");
        };
        assert_eq!(*index, 1, "-√2 is the first real root of x^2 - 2");
        let approx = -BigRational::new(BigInt::from(1414), BigInt::from(1000));
        assert!(
            *lo <= approx && approx <= *hi,
            "the isolating bracket [{lo}, {hi}] must contain -1.414"
        );
        assert!(*hi <= rat(0), "the bracket must not reach past 0");
        assert_eq!(point.try_sign_of(&x_squared_minus_two(), x), Some(0));
    }

    /// `x² = 2 ∧ x = 1` has no real solution at all. The algebraic sampler must
    /// never call it `Sat`.
    #[test]
    fn sqrt2_conflicting_with_a_rational_equality_is_never_sat() {
        let (mut solver, _x) = solver_with_sqrt2();
        // x - 1 = 0
        let eq = solver.new_ineq_atom(Polynomial::univariate(0, &[rat(-1), rat(1)]), AtomKind::Eq);
        let lit = solver.atom_literal(eq, true);
        solver.add_clause(vec![lit]);

        assert_ne!(
            solver.solve(),
            SolverResult::Sat,
            "x^2 = 2 and x = 1 cannot both hold"
        );
    }

    /// The open cell `(−√2, 0)` is non-empty, and the candidate walk must have
    /// a representative *inside* it. This is the end-to-end guard on the
    /// touching-bracket bug fixed in `crate::cad_algebraic`: `−√2`'s isolating
    /// bracket can end exactly at the rational root `0` contributed by `x < 0`,
    /// and an "open sample" taken at that shared endpoint is the root itself,
    /// leaving the cell unrepresented and its solutions unfindable.
    #[test]
    fn an_open_cell_between_an_irrational_and_a_rational_root_is_reachable() {
        let mut solver = NlsatSolver::new();
        // x^2 - 2 < 0
        let lt = solver.new_ineq_atom(x_squared_minus_two(), AtomKind::Lt);
        let l1 = solver.atom_literal(lt, true);
        solver.add_clause(vec![l1]);
        // x < 0
        let neg = solver.new_ineq_atom(Polynomial::from_var(0), AtomKind::Lt);
        let l2 = solver.atom_literal(neg, true);
        solver.add_clause(vec![l2]);

        assert_eq!(
            solver.solve(),
            SolverResult::Sat,
            "every x in (-√2, 0) satisfies both constraints"
        );
    }

    /// A pure but genuinely infeasible constraint set must come back as a
    /// *proof* (`NoRealPoint`), which the caller turns into a theory lemma —
    /// not as `Inconclusive`.
    #[test]
    fn pure_infeasible_set_is_proved_empty_not_inconclusive() {
        let mut solver = NlsatSolver::new();
        let x = solver.new_arith_var();
        // x^2 - 2 = 0
        let eq = solver.new_ineq_atom(x_squared_minus_two(), AtomKind::Eq);
        let lit = solver.atom_literal(eq, true);
        solver.add_clause(vec![lit]);
        // x - 1 = 0
        let eq2 = solver.new_ineq_atom(Polynomial::univariate(0, &[rat(-1), rat(1)]), AtomKind::Eq);
        let lit2 = solver.atom_literal(eq2, true);
        solver.add_clause(vec![lit2]);
        // Assign both atoms true without running the search.
        solver
            .assignment
            .assign(lit, crate::assignment::Justification::Unit);
        solver
            .assignment
            .assign(lit2, crate::assignment::Justification::Unit);

        match solver.sample_algebraic_witness(x) {
            AlgebraicWitness::NoRealPoint(lemma) => {
                assert_eq!(lemma.len(), 2, "both atoms belong to the lemma: {lemma:?}");
            }
            AlgebraicWitness::Found(point) => {
                panic!("no real x satisfies both, yet {point:?} was accepted")
            }
            AlgebraicWitness::Inconclusive => {
                panic!("both constraints are pure and exactly decidable — this is a proof")
            }
        }
    }

    /// The point checker must *refute* a point that violates an assigned atom,
    /// rather than shrugging with "undecided".
    #[test]
    fn point_check_rejects_a_violating_point() {
        let mut solver = NlsatSolver::new();
        let x = solver.new_arith_var();
        let eq = solver.new_ineq_atom(x_squared_minus_two(), AtomKind::Eq);
        let lit = solver.atom_literal(eq, true);
        solver.add_clause(vec![lit]);
        solver
            .assignment
            .assign(lit, crate::assignment::Justification::Unit);

        // 1 is not a root of x^2 - 2.
        assert_eq!(
            solver.point_satisfies_assigned_atoms(x, &CadPoint::rational(rat(1))),
            Some(false)
        );
        // Its actual roots are.
        let roots = isolate_root_samples(&[x_squared_minus_two()], x);
        for root in &roots {
            assert_eq!(
                solver.point_satisfies_assigned_atoms(x, &root.to_point()),
                Some(true),
                "±√2 satisfy x^2 = 2"
            );
        }
    }

    /// Regression: `x² = 2 ∧ x·y = 1 ∧ y = 1`. The coupled atom is invisible
    /// while `y` is unassigned, so `x` may take √2; once `y = 1` lands, the
    /// product is `√2 ≠ 1`. Answering `Sat` here would be a wrong verdict, and
    /// the exact-evaluation net exists precisely to stop it.
    #[test]
    fn coupled_rational_variable_cannot_produce_a_wrong_sat() {
        let mut solver = NlsatSolver::new();
        let x = solver.new_arith_var();
        let y = solver.new_arith_var();

        let eq = solver.new_ineq_atom(x_squared_minus_two(), AtomKind::Eq);
        let l1 = solver.atom_literal(eq, true);
        solver.add_clause(vec![l1]);

        // x*y - 1 = 0
        let xy = Polynomial::sub(
            &Polynomial::mul(&Polynomial::from_var(x), &Polynomial::from_var(y)),
            &Polynomial::constant(rat(1)),
        );
        let eq2 = solver.new_ineq_atom(xy, AtomKind::Eq);
        let l2 = solver.atom_literal(eq2, true);
        solver.add_clause(vec![l2]);

        // y - 1 = 0
        let eq3 = solver.new_ineq_atom(
            Polynomial::sub(&Polynomial::from_var(y), &Polynomial::constant(rat(1))),
            AtomKind::Eq,
        );
        let l3 = solver.atom_literal(eq3, true);
        solver.add_clause(vec![l3]);

        assert_ne!(
            solver.solve(),
            SolverResult::Sat,
            "x^2 = 2, x*y = 1 and y = 1 force x = 1, contradicting x^2 = 2"
        );
    }

    /// Regression: `x² = 2 ∧ y² = 3 ∧ x·y = 1`. Two algebraic values would
    /// leave `x·y − 1` undecidable, so the second one must be refused.
    #[test]
    fn two_algebraic_variables_cannot_produce_a_wrong_sat() {
        let mut solver = NlsatSolver::new();
        let x = solver.new_arith_var();
        let y = solver.new_arith_var();

        let eq = solver.new_ineq_atom(x_squared_minus_two(), AtomKind::Eq);
        let l1 = solver.atom_literal(eq, true);
        solver.add_clause(vec![l1]);

        // y^2 - 3 = 0
        let y2 = Polynomial::sub(
            &Polynomial::mul(&Polynomial::from_var(y), &Polynomial::from_var(y)),
            &Polynomial::constant(rat(3)),
        );
        let eq2 = solver.new_ineq_atom(y2, AtomKind::Eq);
        let l2 = solver.atom_literal(eq2, true);
        solver.add_clause(vec![l2]);

        // x*y - 1 = 0
        let xy = Polynomial::sub(
            &Polynomial::mul(&Polynomial::from_var(x), &Polynomial::from_var(y)),
            &Polynomial::constant(rat(1)),
        );
        let eq3 = solver.new_ineq_atom(xy, AtomKind::Eq);
        let l3 = solver.atom_literal(eq3, true);
        solver.add_clause(vec![l3]);

        assert_ne!(
            solver.solve(),
            SolverResult::Sat,
            "√2 · √3 = √6 ≠ 1, so this must never be reported satisfiable"
        );
    }
}
