//! An LP relaxation of the formula's unconditional facts, used to bound terms
//! that no single atom bounds on its own.
//!
//! ## Why the single-atom reading is not enough
//!
//! [`super::int_case_split`] turns a short integer domain into an explicit
//! `(or (= t v0) … (= t vk))` disjunction so CDCL has an atom to branch a
//! shared term's *value* on. It derives that domain by reading one atom at a
//! time, and it deliberately refuses any atom mentioning more than one term:
//! transferring a bound from one term to another needs to know what the other
//! terms are, and inside the search that knowledge belongs to a branch rather
//! than to the formula.
//!
//! The QF_UFLIA `Wisa` family is entirely built out of multi-term facts. A
//! function argument such as `(- (- fmt1 2) fmt0)` is confined to five values
//! by `fmt0 = 0`, `fmt1 > fmt0 + 1` and `fmt1 < fmt0 + fmt_length - 1` with
//! `fmt_length = 8` — but *no single one* of those atoms bounds it, and after
//! `purify_numeric_uf_args` hoists the argument into a proxy variable the fact
//! that names the proxy (`v = fmt1 - 2 - fmt0`) mentions three terms. So the
//! single-atom reading learns nothing, no disjunction is emitted, congruence
//! closure never relates `s_count(v)` to `s_count(4)`, and an unsatisfiable
//! instance comes back `sat`.
//!
//! ## What this module derives, and why it is sound
//!
//! Take **every** parsed arithmetic atom that is assigned `true` at decision
//! level 0 — the same facts the single-atom reading already trusts, with the
//! single-term restriction lifted — and assert all of them together into a
//! *scratch* simplex tableau, one column per distinct term. Integrality is
//! dropped: the tableau is a plain LP over the rationals. Then minimise and
//! maximise the target term's column over that region.
//!
//! The three properties that make the result usable are:
//!
//! 1. **Level-0 truth.** Each atom holds in every model of the input (nothing
//!    was decided, so no backtrack can retract it). The conjunction of all of
//!    them therefore also holds in every model.
//! 2. **Relaxation.** Every model of the input maps to a point of the LP
//!    region: dropping integrality and treating distinct `TermId`s as
//!    independent columns can only *enlarge* the feasible set, never shrink it.
//!    (Two terms that happen to denote the same value get two columns; that
//!    loses tightness and nothing else.)
//! 3. **Widening in one direction only.** Consequently `lp_min <= t <= lp_max`
//!    in every model, so `t ∈ [ceil(lp_min), floor(lp_max)]` in every model and
//!    the disjunction over that range is implied by the input. Every
//!    approximation this module makes — the dropped integrality, the strict
//!    bounds read at their open endpoint, the aliased columns — widens the
//!    interval. A wider interval means more disjuncts, never a lost model.
//!
//! The direction matters: a bound that came out too *tight* would delete
//! models and manufacture a wrong `unsat`. None of the approximations here can
//! do that, because each of them adds feasible points rather than removing
//! them.
//!
//! Anything short of a proved optimum yields no bound at all. `Unbounded` says
//! the region really is open on that side; `Unknown` means the pivot budget ran
//! out, and the truncated search's incumbent is feasible but not optimal, so
//! reading it as an extremum could tighten the interval past what the facts
//! entail; `Infeasible` says the level-0 facts contradict each other, which is
//! the *solver's* business to report, not this module's.
//!
//! ## Cost
//!
//! One tableau is built per refinement round and shared by every term the round
//! wants to bound; each term costs two `optimize_linexpr` calls. The build is
//! declined outright for a formula with more atoms or terms than the caps below
//! — past that size the round is not affordable anyway, and the refinement it
//! feeds is itself gated on the first solve having been fast.

use num_rational::Rational64;
use num_traits::ToPrimitive;
use oxiz_core::ast::TermId;
use oxiz_theories::arithmetic::{LinExpr, Simplex, SimplexOptStatus, VarId};
use rustc_hash::FxHashMap;

use super::types::{ArithConstraintType, ParsedArithConstraint};

/// Most level-0 atoms one scratch tableau will hold.
///
/// OxiZ tuning decision. The instances this rescues are small, fast and wrong;
/// a formula with more unconditional arithmetic facts than this is a large one
/// where a second full search is the last thing worth spending time on.
const MAX_LP_ATOMS: usize = 4_096;

/// Most distinct terms (columns) one scratch tableau will hold.
///
/// OxiZ tuning decision, paired with [`MAX_LP_ATOMS`]: the simplex cost is
/// driven by rows times columns, so both need a ceiling for the build to have a
/// bounded price.
const MAX_LP_COLUMNS: usize = 4_096;

/// A scratch LP over the formula's level-0 arithmetic facts.
pub(super) struct RootLevelLp {
    /// The relaxation. Owned outright — the live theory solvers are never
    /// touched, so nothing here can perturb the search that is about to be
    /// replayed.
    simplex: Simplex,
    /// Which column stands for which term.
    column_of: FxHashMap<TermId, VarId>,
}

impl RootLevelLp {
    /// Build the relaxation from `atoms`, each paired with whether its SAT
    /// variable carries a `Constraint::Eq` (a numeric equality is parsed with
    /// [`ArithConstraintType::Le`] as a placeholder, so the flag is the only
    /// thing that distinguishes `=` from `<=`).
    ///
    /// Returns `None` when there is nothing to relax or the problem is past the
    /// size caps.
    pub(super) fn build(atoms: &[(&ParsedArithConstraint, bool)]) -> Option<Self> {
        if atoms.is_empty() || atoms.len() > MAX_LP_ATOMS {
            return None;
        }

        // Pass 1: allocate every column before a single row is added.
        // `Simplex::add_le` allocates its slack from the same id space as
        // `new_var`, so interleaving the two would hand a term a column id that
        // a slack had already taken.
        let mut column_of: FxHashMap<TermId, VarId> = FxHashMap::default();
        let mut simplex = Simplex::new();
        for (parsed, _) in atoms {
            for &(term, _) in parsed.terms.iter() {
                if column_of.len() >= MAX_LP_COLUMNS && !column_of.contains_key(&term) {
                    return None;
                }
                column_of.entry(term).or_insert_with(|| simplex.new_var());
            }
        }
        if column_of.is_empty() {
            return None;
        }

        // Pass 2: one row per atom. `Σ cᵢ·tᵢ ⋈ k` is asserted in the
        // `expr ⋈ 0` form the tableau wants, i.e. with the constant moved
        // across.
        //
        // The `reason` slot is what the simplex would cite in a conflict
        // explanation. This tableau is never asked for one — only for optima —
        // so the row index is used purely to keep the tags distinct.
        for (row, (parsed, pins_exactly)) in atoms.iter().enumerate() {
            let mut expr = LinExpr::new();
            for &(term, coefficient) in parsed.terms.iter() {
                let Some(&column) = column_of.get(&term) else {
                    continue;
                };
                expr.add_term(column, coefficient);
            }
            expr.add_constant(-parsed.constant);
            let reason = u32::try_from(row).unwrap_or(u32::MAX);
            if *pins_exactly {
                simplex.add_eq(expr, reason);
                continue;
            }
            match parsed.constraint_type {
                ArithConstraintType::Le => simplex.add_le(expr, reason),
                ArithConstraintType::Lt => simplex.add_strict_lt(expr, reason),
                ArithConstraintType::Ge => simplex.add_ge(expr, reason),
                ArithConstraintType::Gt => simplex.add_strict_gt(expr, reason),
            }
        }

        Some(Self { simplex, column_of })
    }

    /// The interval the relaxation puts `term` in, as integer endpoints, with
    /// `None` on a side the region leaves open (or that the optimiser could not
    /// prove).
    ///
    /// A term with no column at all appears in no level-0 fact, so nothing is
    /// known about it and both sides come back open.
    pub(super) fn integer_bounds(&mut self, term: TermId) -> (Option<i64>, Option<i64>) {
        let Some(&column) = self.column_of.get(&term) else {
            return (None, None);
        };

        // min t: the smallest value the region allows, rounded *up* to the
        // first integer that could actually be taken.
        let mut minimise = LinExpr::new();
        minimise.add_term(column, Rational64::from_integer(1));
        let low = self
            .optimum(&minimise)
            .and_then(|value| value.ceil().to_integer().to_i64());

        // max t = -min(-t), rounded *down* for the same reason.
        let mut maximise = LinExpr::new();
        maximise.add_term(column, Rational64::from_integer(-1));
        let high = self
            .optimum(&maximise)
            .and_then(|value| (-value).floor().to_integer().to_i64());

        (low, high)
    }

    /// The proved minimum of `objective`, or `None` for every other outcome.
    ///
    /// `Unknown` in particular must not be read as a bound: the optimiser
    /// reports it when the pivot budget runs out, and the incumbent it leaves
    /// behind is merely feasible. Treating that as the minimum would produce an
    /// interval tighter than the facts entail — the one direction of error that
    /// can lose a model.
    fn optimum(&mut self, objective: &LinExpr) -> Option<Rational64> {
        match self.simplex.optimize_linexpr(objective) {
            SimplexOptStatus::Optimal(value) => Some(value),
            SimplexOptStatus::Unbounded
            | SimplexOptStatus::Unknown
            | SimplexOptStatus::Infeasible => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::ast::TermManager;
    use smallvec::smallvec;

    /// `Σ terms ⋈ constant`, in the shape the encoder records.
    fn atom(
        terms: Vec<(TermId, i64)>,
        constant: i64,
        constraint_type: ArithConstraintType,
        reason_term: TermId,
    ) -> ParsedArithConstraint {
        ParsedArithConstraint {
            terms: terms
                .into_iter()
                .map(|(t, c)| (t, Rational64::from_integer(c)))
                .collect(),
            constant: Rational64::from_integer(constant),
            constraint_type,
            reason_term,
        }
    }

    /// Three variables and a spare term id to tag rows with.
    fn fixture() -> (TermManager, TermId, TermId, TermId) {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let z = tm.mk_var("z", int_sort);
        (tm, x, y, z)
    }

    /// The shape the single-atom reading cannot see: `y` is pinned by an
    /// equality, and `x` is bounded only *relative* to `y`. Every fact here
    /// mentions two terms, so `int_case_split::domain_from_atom` declines all
    /// of them; the relaxation reads them together and gets `x ∈ [3, 7]`.
    #[test]
    fn a_bound_relative_to_a_pinned_term_is_recovered() {
        let (mut tm, x, y, _z) = fixture();
        let tag = tm.mk_var("tag", tm.sorts.bool_sort);
        // y - 5 = 0
        let pin = atom(vec![(y, 1)], 5, ArithConstraintType::Le, tag);
        // x - y >= -2   (x >= 3)
        let lower = atom(vec![(x, 1), (y, -1)], -2, ArithConstraintType::Ge, tag);
        // x - y <= 2    (x <= 7)
        let upper = atom(vec![(x, 1), (y, -1)], 2, ArithConstraintType::Le, tag);
        let mut lp = RootLevelLp::build(&[(&pin, true), (&lower, false), (&upper, false)])
            .expect("three atoms over two terms is within the caps");
        assert_eq!(lp.integer_bounds(x), (Some(3), Some(7)));
        assert_eq!(lp.integer_bounds(y), (Some(5), Some(5)));
    }

    /// Rounding must go outward on both ends: a region of `[3/2, 9/2]` admits
    /// the integers `2..=4`, and reporting `[2, 4]` (rather than `[1, 5]` or
    /// `[2, 5]`) is what keeps the disjunction both implied and short.
    #[test]
    fn fractional_optima_round_inward_to_the_integers_they_admit() {
        let (mut tm, x, _y, _z) = fixture();
        let tag = tm.mk_var("tag", tm.sorts.bool_sort);
        // 2x >= 3  ->  x >= 1.5
        let lower = atom(vec![(x, 2)], 3, ArithConstraintType::Ge, tag);
        // 2x <= 9  ->  x <= 4.5
        let upper = atom(vec![(x, 2)], 9, ArithConstraintType::Le, tag);
        let mut lp =
            RootLevelLp::build(&[(&lower, false), (&upper, false)]).expect("within the caps");
        assert_eq!(lp.integer_bounds(x), (Some(2), Some(4)));
    }

    /// A side the facts leave open must come back open, not as whatever the
    /// incumbent assignment happened to be.
    #[test]
    fn an_unbounded_side_reports_no_bound() {
        let (mut tm, x, _y, _z) = fixture();
        let tag = tm.mk_var("tag", tm.sorts.bool_sort);
        let lower = atom(vec![(x, 1)], 0, ArithConstraintType::Ge, tag);
        let mut lp = RootLevelLp::build(&[(&lower, false)]).expect("within the caps");
        assert_eq!(lp.integer_bounds(x), (Some(0), None));
    }

    /// A term that appears in no level-0 fact has no column, and asking about
    /// it must not silently report the tableau's constant term as its value.
    #[test]
    fn a_term_with_no_column_is_unconstrained() {
        let (mut tm, x, _y, z) = fixture();
        let tag = tm.mk_var("tag", tm.sorts.bool_sort);
        let bounded = atom(vec![(x, 1)], 4, ArithConstraintType::Le, tag);
        let mut lp = RootLevelLp::build(&[(&bounded, false)]).expect("within the caps");
        assert_eq!(lp.integer_bounds(z), (None, None));
    }

    /// A strict bound is read at its open endpoint, which is one unit looser
    /// than integrality would allow (`x > 1` really means `x >= 2`). That is
    /// the safe direction: an extra disjunct costs a branch, a missing one
    /// would cost a model.
    #[test]
    fn strict_bounds_widen_rather_than_tighten() {
        let (mut tm, x, _y, _z) = fixture();
        let tag = tm.mk_var("tag", tm.sorts.bool_sort);
        let lower = atom(vec![(x, 1)], 1, ArithConstraintType::Gt, tag);
        let upper = atom(vec![(x, 1)], 4, ArithConstraintType::Lt, tag);
        let mut lp =
            RootLevelLp::build(&[(&lower, false), (&upper, false)]).expect("within the caps");
        let (low, high) = lp.integer_bounds(x);
        assert!(
            low.is_some_and(|l| l <= 2),
            "a lower bound may be loose, never tighter than 2"
        );
        assert!(
            high.is_some_and(|h| h >= 3),
            "an upper bound may be loose, never tighter than 3"
        );
    }

    /// Nothing to relax means no tableau, rather than an empty one whose optima
    /// would all read as zero.
    #[test]
    fn an_empty_fact_set_builds_nothing() {
        assert!(RootLevelLp::build(&[]).is_none());
    }

    /// Columns must all be allocated before any row is added.
    ///
    /// `Simplex::new_slack` takes its id from the same counter as `new_var`, so
    /// interleaving "allocate a column" with "add a row" would hand a later
    /// term the id a slack had already claimed — silently aliasing a term onto
    /// a constraint's slack variable and reading that slack's range as the
    /// term's. The failure is invisible from outside (no panic, just wrong
    /// optima), so it is pinned here: many terms are introduced by *later*
    /// atoms than the first, and each must still get its own honest bounds.
    #[test]
    fn columns_are_not_aliased_onto_constraint_slacks() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let tag = tm.mk_var("tag", tm.sorts.bool_sort);
        // Eight terms, each pinned to a distinct value by its own equality, and
        // each first mentioned by a different (increasingly late) atom.
        let terms: Vec<TermId> = (0..8)
            .map(|i| tm.mk_var(&format!("t{i}"), int_sort))
            .collect();
        let atoms: Vec<ParsedArithConstraint> = terms
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                atom(
                    vec![(t, 1)],
                    i64::try_from(i).unwrap_or(0) * 10,
                    ArithConstraintType::Le,
                    tag,
                )
            })
            .collect();
        let refs: Vec<(&ParsedArithConstraint, bool)> = atoms.iter().map(|a| (a, true)).collect();
        let mut lp = RootLevelLp::build(&refs).expect("eight atoms is within the caps");
        for (i, &t) in terms.iter().enumerate() {
            let expected = i64::try_from(i).unwrap_or(0) * 10;
            assert_eq!(
                lp.integer_bounds(t),
                (Some(expected), Some(expected)),
                "term {i} must read its own equality, not some slack's range"
            );
        }
    }

    /// A fact with no terms at all (a parse that folded everything into the
    /// constant) contributes no column, and a tableau with no columns is not
    /// built.
    #[test]
    fn a_constant_only_fact_builds_nothing() {
        let (mut tm, _x, _y, _z) = fixture();
        let tag = tm.mk_var("tag", tm.sorts.bool_sort);
        let constant_only = ParsedArithConstraint {
            terms: smallvec![],
            constant: Rational64::from_integer(3),
            constraint_type: ArithConstraintType::Le,
            reason_term: tag,
        };
        assert!(RootLevelLp::build(&[(&constant_only, false)]).is_none());
    }
}
