//! Turning a small integer domain into Boolean branches the CDCL core can see.
//!
//! ## The shape of the problem
//!
//! Two theories that share a term have to agree about it, and Nelson-Oppen
//! equality sharing (`theory_manager::nelson_oppen_combine`) is how they do:
//! whenever arithmetic *entails* `t = v`, that equality is handed to EUF. The
//! word "entails" is the catch. Over the integers a term can be confined to a
//! handful of values without any one of them being entailed — `(>= d 0)`
//! together with `(<= d 1)` leaves `d` in `{0, 1}` and entails neither `d = 0`
//! nor `d = 1` — because linear integer arithmetic is not convex.
//!
//! That costs nothing until the same term is an uninterpreted-function
//! argument, at which point *which* of the values it takes decides the answer:
//! `f(0)` and `f(1)` are unrelated as far as congruence closure is concerned.
//! Arithmetic has nothing to propagate, EUF has nothing to refute, and the
//! CDCL(T) core has no atom whose two polarities correspond to the two cases —
//! so it never explores them, and an unsatisfiable formula comes back `sat`.
//!
//! ## What this module does about it
//!
//! It supplies the missing atoms. Once a candidate `sat` exists, every numeric
//! term that (a) is shared with EUF and (b) is confined to a short range by
//! facts the formula asserts unconditionally gets an explicit disjunction
//! `(or (= t lo) … (= t hi))` asserted before the search is run again. Each
//! disjunct is an equality atom both theories see, so the moment CDCL commits
//! to one, congruence closure has something concrete to work with. It is the
//! same effect a hand-written `(assert (or (= d 0) (= d 1)))` would have, and
//! it is the textbook remedy for non-convex theory combination (Barrett et
//! al., *Decision Procedures*, ch. 10; both Z3 and cvc5 do a version of it).
//!
//! ## Why asserting it cannot change the answer
//!
//! The disjunction restates a range the formula's own unconditional facts
//! already impose, so every model of the input satisfies it and no model is
//! lost. "Unconditional" is doing real work in that sentence, and it is why
//! the derivation reads only atoms that are (i) assigned `true` and (ii) at
//! decision level 0: together those mean the formula forced the atom on its
//! own, so it survives every backtrack and holds in every model.
//!
//! Two derivations run over exactly those atoms, and both are used.
//!
//! The **single-atom reading** below ([`Solver::root_level_int_domains`])
//! additionally requires each atom to constrain a *single* term. That is not
//! frugality: reading a bound off one term and transferring it to another
//! through a multi-term fact needs to know what the other terms are, and
//! inside a search that knowledge belongs to the branch being explored rather
//! than to the formula. Baking a branch-local bound into a lemma that outlives
//! the branch is exactly how a satisfiable instance acquires a spurious
//! `unsat`.
//!
//! The **LP relaxation** ([`super::int_range_lp`]) lifts the single-term
//! restriction the only way that stays sound: it asserts *all* the level-0
//! facts simultaneously into a scratch tableau and reads the target term's
//! proved optima off that. Nothing is transferred from a branch, because the
//! tableau contains no branch — only facts that hold in every model — and
//! every approximation it makes (dropped integrality, aliased columns, strict
//! bounds read at their open endpoint) widens the resulting interval rather
//! than narrowing it. See that module for the argument in full. Whichever
//! derivation bounds a side more tightly wins, since both are sound.
//!
//! One further guard is belt-and-braces rather than load-bearing: if the
//! candidate model the core just produced puts a term outside the range
//! derived for it, the derivation and the model disagree and something is
//! wrong with one of them, so the term is skipped instead of split.
//!
//! ## Paying for it
//!
//! A refinement round discards the search and starts over, which is only worth
//! doing when starting over is cheap. The instances a missing case split
//! actually rescues are the ones that answered quickly and wrongly, so the
//! whole refinement is gated on the first solve having finished inside
//! [`REFINEMENT_TIME_CEILING_MS`] and is allowed
//! [`MAX_REFINEMENT_ROUNDS`] round.

use num_traits::ToPrimitive;
use oxiz_core::ast::TermManager;
use oxiz_sat::Lit;
use rustc_hash::FxHashMap;
use std::ops::RangeInclusive;

use super::Solver;
use super::int_range_lp::RootLevelLp;
use super::trail::TrailOp;
use super::types::{ArithConstraintType, Constraint, ParsedArithConstraint};
use oxiz_core::ast::TermId;

/// Wall-clock ceiling on the *first* solve, past which no refinement round is
/// attempted at all.
///
/// OxiZ tuning decision. A round throws the search away and redoes it, so the
/// ceiling buys a bounded slowdown (roughly a doubling) on instances that get
/// no benefit, in exchange for the chance to correct a wrong `sat`.
///
/// It was 5s, chosen when the only refinement was the single-atom reading,
/// whose targets are found in milliseconds — "a fast, confident, wrong `sat`"
/// really did describe every instance it could help. The LP relaxation
/// ([`super::int_range_lp`]) changed that. The QF_UFLIA `Wisa` instances it
/// exists for are wrong `sat`s that take a *minute* to reach the first
/// candidate model, so at 5s the refinement was built, correct, and never
/// reached: `xs-08-20-3-2-4-5` answers `sat` at 258s with the gate at 5s, and
/// `unsat` at 254s with the gate raised — the refinement costs almost nothing
/// there, it was simply forbidden from running.
///
/// 120s is set against that measurement: high enough that a minute-scale
/// candidate model still gets its one refinement round, low enough that a
/// genuinely long search is not doubled. A wrong verdict is a much worse
/// outcome than a slow one, and the round budget
/// ([`MAX_REFINEMENT_ROUNDS`]) caps the downside at one extra search.
pub(super) const REFINEMENT_TIME_CEILING_MS: u64 = 120_000;

/// How many discard-and-re-solve rounds one `check` may spend.
///
/// OxiZ tuning decision: one. Every eligible term is split within that single
/// round (see [`MAX_TERMS_PER_ROUND`]), so a second round would have nothing
/// new to split and would only be re-running a search that already had all
/// the atoms it was going to get.
const MAX_REFINEMENT_ROUNDS: u32 = 1;

/// How many terms one round may split.
///
/// OxiZ tuning decision, set deliberately high. The term whose enumeration
/// resolves the formula is not reliably the one with the narrowest domain, so
/// a tight cap risks spending the round on terms that were never the problem
/// and leaving the one that was untouched.
const MAX_TERMS_PER_ROUND: usize = 48;

/// How many terms one round may ask the LP relaxation about.
///
/// OxiZ tuning decision. Each query is two full `optimize_linexpr` calls over
/// the whole level-0 tableau, and the round can only spend
/// [`MAX_TERMS_PER_ROUND`] splits however many answers come back — so querying
/// past that budget buys nothing and can cost a great deal on a formula with
/// thousands of shared terms. Set to the split budget itself.
const MAX_LP_QUERIES_PER_ROUND: usize = MAX_TERMS_PER_ROUND;

/// The widest domain still worth enumerating, measured as `high - low`.
///
/// OxiZ tuning decision. Every extra unit of width is another disjunct and
/// another branch for CDCL to explore, and a term left this loose by the
/// formula's unconditional facts is usually loose because the instance is a
/// large one where the extra search is least affordable. The non-convex
/// shapes that occur in practice — a function argument pinned to a few
/// alternatives — sit well inside this.
const MAX_ENUMERABLE_SPAN: i64 = 12;

/// Which way [`integer_quotient`] rounds an inexact division.
#[derive(Clone, Copy)]
enum Rounding {
    /// Toward negative infinity.
    Down,
    /// Toward positive infinity.
    Up,
}

/// `numerator / divisor`, rounded as `rounding` says.
///
/// Rust's `/` truncates toward zero, which is the wrong direction on one side
/// of zero and would silently widen a derived bound into one the formula does
/// not entail. `div_euclid` rounds toward negative infinity for a *positive*
/// divisor only, so a negative divisor is handled by negating both operands
/// first — the one case where the answer is not representable (`i64::MIN` has
/// no negation) reports `None`, which callers treat as "no bound learned".
fn integer_quotient(numerator: i64, divisor: i64, rounding: Rounding) -> Option<i64> {
    let (numerator, divisor) = match rounding {
        Rounding::Down => (numerator, divisor),
        // ceil(a / b) = -floor(-a / b): flip the numerator, round down, flip
        // back. The final negation is checked too, since `-i64::MIN` is the
        // one quotient that does not fit.
        Rounding::Up => (numerator.checked_neg()?, divisor),
    };
    let quotient = if divisor < 0 {
        numerator.checked_neg()?.div_euclid(divisor.checked_neg()?)
    } else {
        numerator.div_euclid(divisor)
    };
    match rounding {
        Rounding::Down => Some(quotient),
        Rounding::Up => quotient.checked_neg(),
    }
}

/// What is known about an integer term's value: an inclusive range with either
/// side possibly still open.
///
/// `low > high` is a legitimate state — it says the facts contradict each
/// other on this term — and is reported as having nothing to enumerate rather
/// than as a contradiction, since this module's job is to add branches, not to
/// decide anything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct IntDomain {
    /// Greatest lower bound learned so far.
    low: Option<i64>,
    /// Least upper bound learned so far.
    high: Option<i64>,
}

impl IntDomain {
    /// The domain that rules nothing out.
    fn open() -> Self {
        Self::default()
    }

    /// `value <= term`, with no upper bound.
    fn from_below(value: Option<i64>) -> Self {
        Self {
            low: value,
            high: None,
        }
    }

    /// `term <= value`, with no lower bound.
    fn from_above(value: Option<i64>) -> Self {
        Self {
            low: None,
            high: value,
        }
    }

    /// Whether this domain says anything at all.
    fn is_open(self) -> bool {
        self.low.is_none() && self.high.is_none()
    }

    /// Intersect with `other`, keeping whichever side of each is tighter.
    fn narrow_with(&mut self, other: IntDomain) {
        if let Some(candidate) = other.low {
            self.low = Some(self.low.map_or(candidate, |held| held.max(candidate)));
        }
        if let Some(candidate) = other.high {
            self.high = Some(self.high.map_or(candidate, |held| held.min(candidate)));
        }
    }

    /// The values to enumerate, or `None` when this domain is open on either
    /// side, self-contradictory, or wider than `span_limit`.
    fn enumerable(self, span_limit: i64) -> Option<RangeInclusive<i64>> {
        let (low, high) = (self.low?, self.high?);
        // `saturating_sub` keeps a range spanning most of `i64` from wrapping
        // into a small (and therefore accepted) width.
        if low > high || high.saturating_sub(low) > span_limit {
            return None;
        }
        Some(low..=high)
    }
}

impl Solver {
    /// Look for terms confined to a short integer range and give the SAT core
    /// an explicit disjunction over that range.
    ///
    /// `true` means at least one new disjunction was asserted, and the caller
    /// owes the search a reset and a re-solve. `false` means there was nothing
    /// left to split and the candidate `sat` stands as it is.
    pub(super) fn split_narrow_int_domains(&mut self, manager: &mut TermManager) -> bool {
        if !self.arith.is_integer() || self.case_split_rounds >= MAX_REFINEMENT_ROUNDS {
            return false;
        }

        let mut targets = self.collect_split_targets();
        if targets.is_empty() {
            return false;
        }

        // Narrowest domain first: a term with two candidate values costs two
        // disjuncts, which is the cheapest split that can possibly help, so
        // the round's budget goes to those before anything wider. The term id
        // breaks ties, only so that two runs of the same query choose the same
        // terms.
        targets.sort_by_key(|(term, values)| (values.end() - values.start(), *term));
        targets.truncate(MAX_TERMS_PER_ROUND);

        for (term, values) in targets {
            self.assert_value_disjunction(term, values, manager);
        }
        self.case_split_rounds += 1;
        // The round ran: every target got its disjunction, so the re-solve
        // about to happen *does* branch on those domains and its verdict is
        // trustworthy again.
        self.case_split_skipped_targets = false;
        true
    }

    /// Record that a refinement round was declined by the affordability
    /// ceiling while this candidate model still had terms worth splitting.
    ///
    /// Deriving the targets is cheap relative to the re-solve the round was
    /// declined to avoid (one LP build plus at most
    /// [`MAX_LP_QUERIES_PER_ROUND`] optimisations, against a whole second
    /// search), and it is the only way to distinguish "this candidate is
    /// fully branched" from "I could not afford to check" — which is the
    /// difference between a `Sat` that may be reported and one that may not.
    /// Nothing is asserted here, so the caller owes no re-solve.
    pub(super) fn note_unaffordable_case_split(&mut self) {
        if !self.arith.is_integer() || self.case_split_rounds >= MAX_REFINEMENT_ROUNDS {
            return;
        }
        if !self.collect_split_targets().is_empty() {
            self.case_split_skipped_targets = true;
        }
    }

    /// The terms worth splitting this round, each with the values to enumerate.
    ///
    /// Two sources feed in, and they want exactly the same treatment.
    /// `numeric_uf_arg_terms` (filled by `purify_numeric_uf_args`) are the
    /// arguments the arithmetic and EUF solvers share. `lookup_index_terms`
    /// (flattened finite-map indices, see `encode::finite_map_ite`) are
    /// indices whose value decides which table entry a lookup resolves to —
    /// again something CDCL cannot branch on without an atom for it. Running
    /// both through one candidate set means the domain derivation, the
    /// per-term deduplication and the round budget are written once.
    ///
    /// Terms the single-atom reading leaves unbounded get a second chance from
    /// [`RootLevelLp`], which reads all the level-0 facts *together* instead of
    /// one at a time. That tableau is only built if some term actually needs
    /// it, and it is shared by every term that does.
    fn collect_split_targets(&self) -> Vec<(TermId, RangeInclusive<i64>)> {
        let domains = self.root_level_int_domains();
        let mut targets: Vec<(TermId, RangeInclusive<i64>)> = Vec::new();
        let mut unresolved: Vec<TermId> = Vec::new();
        for &term in self
            .numeric_uf_arg_terms
            .iter()
            .chain(self.lookup_index_terms.iter())
        {
            if self.case_split_terms.contains(&term) {
                continue;
            }
            let Some(values) = domains
                .get(&term)
                .and_then(|domain| domain.enumerable(MAX_ENUMERABLE_SPAN))
            else {
                unresolved.push(term);
                continue;
            };
            if let Some(values) = self.accept_range(term, values) {
                targets.push((term, values));
            }
        }

        if unresolved.is_empty() {
            return targets;
        }
        // The round can only *use* `MAX_TERMS_PER_ROUND` splits, and each LP
        // query is two full simplex optimisations, so querying every unresolved
        // term would pay an unbounded price for a bounded benefit. Sorting by
        // term id keeps which terms get asked stable across runs of the same
        // query.
        unresolved.sort_unstable();
        unresolved.truncate(MAX_LP_QUERIES_PER_ROUND);
        let Some(mut relaxation) = self.build_root_level_lp() else {
            return targets;
        };
        for term in unresolved {
            let (low, high) = relaxation.integer_bounds(term);
            // Start from whatever the single-atom reading did manage to learn:
            // both derivations are sound, so keeping the tighter side of each
            // is sound too, and a one-sided single-atom bound plus a one-sided
            // LP bound can together be enumerable when neither is alone.
            let mut domain = domains.get(&term).copied().unwrap_or_else(IntDomain::open);
            domain.narrow_with(IntDomain { low, high });
            let Some(values) = domain.enumerable(MAX_ENUMERABLE_SPAN) else {
                continue;
            };
            if let Some(values) = self.accept_range(term, values) {
                targets.push((term, values));
            }
        }
        targets
    }

    /// Hand back `values` unless the candidate model puts `term` outside it.
    ///
    /// The model the core just built is the authority on this term. A range
    /// that excludes it means this pass derived something the theory solver
    /// disagrees with, and declining is the cheap, safe response to a
    /// disagreement neither side can be shown to win.
    fn accept_range(
        &self,
        term: TermId,
        values: RangeInclusive<i64>,
    ) -> Option<RangeInclusive<i64>> {
        if let Some(in_model) = self.arith.value(term).and_then(|v| v.to_i64())
            && !values.contains(&in_model)
        {
            return None;
        }
        Some(values)
    }

    /// The LP relaxation of every level-0 arithmetic fact, or `None` when there
    /// is nothing to relax or the formula is past [`RootLevelLp`]'s size caps.
    ///
    /// A numeric equality is parsed with [`ArithConstraintType::Le`] as a
    /// placeholder (see the encoder's `TermKind::Eq` arm), so the recorded
    /// `Constraint` is what distinguishes `=` from `<=` and has to travel with
    /// each atom.
    fn build_root_level_lp(&self) -> Option<RootLevelLp> {
        let atoms: Vec<(&ParsedArithConstraint, bool)> = self
            .root_level_true_atoms()
            .map(|(var, parsed)| {
                let pins_exactly =
                    matches!(self.var_to_constraint.get(&var), Some(Constraint::Eq(_, _)));
                (parsed, pins_exactly)
            })
            .collect();
        RootLevelLp::build(&atoms)
    }

    /// Assert `(or (= term v) …)` over `values` and remember that `term` has
    /// been split, so a later round does not repeat the clause.
    fn assert_value_disjunction(
        &mut self,
        term: TermId,
        values: RangeInclusive<i64>,
        manager: &mut TermManager,
    ) {
        let disjuncts: Vec<Lit> = values
            .map(|value| {
                let literal_term = manager.mk_int(value);
                let equality = manager.mk_eq(term, literal_term);
                self.encode_depth(equality, manager, 0)
            })
            .collect();
        self.sat.add_clause(disjuncts);
        if self.case_split_terms.insert(term) {
            self.trail.push(TrailOp::CaseSplitTermAdded { term });
        }
    }

    /// Every term the formula's unconditional facts confine, and how far.
    ///
    /// See the module documentation for why only single-term atoms decided
    /// `true` at decision level 0 contribute. A term absent from the result is
    /// unconstrained as far as this pass can tell; a term present with an open
    /// side is bounded on one side only.
    fn root_level_int_domains(&self) -> FxHashMap<TermId, IntDomain> {
        let mut domains: FxHashMap<TermId, IntDomain> = FxHashMap::default();
        for (var, parsed) in self.root_level_true_atoms() {
            let Some((term, learned)) = self.domain_from_atom(var, parsed) else {
                continue;
            };
            if learned.is_open() {
                continue;
            }
            domains
                .entry(term)
                .or_insert_with(IntDomain::open)
                .narrow_with(learned);
        }
        domains
    }

    /// The arithmetic atoms that hold in every model of the input: assigned
    /// `true`, at decision level 0.
    ///
    /// The level matters as much as the polarity. An atom assigned above
    /// level 0 holds only inside the branch the search is currently in, and
    /// `false` says its *negation* holds, which is a different constraint that
    /// this pass does not know how to read off `ParsedArithConstraint`.
    fn root_level_true_atoms(
        &self,
    ) -> impl Iterator<Item = (oxiz_sat::Var, &ParsedArithConstraint)> + '_ {
        let trail = self.sat.trail();
        self.var_to_parsed_arith
            .iter()
            .filter(move |&(&var, _)| trail.level(var) == 0 && trail.value(var).is_true())
            .map(|(&var, parsed)| (var, parsed))
    }

    /// The domain one atom pins its term to, or `None` when the atom's shape
    /// is outside what this pass reads (several terms, a fractional
    /// coefficient, a vanishing coefficient).
    fn domain_from_atom(
        &self,
        var: oxiz_sat::Var,
        parsed: &ParsedArithConstraint,
    ) -> Option<(TermId, IntDomain)> {
        // Multi-term facts are excluded on soundness grounds, not for
        // convenience -- see the module documentation.
        let &[(term, coefficient)] = parsed.terms.as_slice() else {
            return None;
        };
        // `Rational64::to_i64` *truncates* a fractional value rather than
        // declining, so integrality has to be established from the
        // denominators before either conversion is trusted.
        if *coefficient.denom() != 1 || *parsed.constant.denom() != 1 {
            return None;
        }
        let (scale, offset) = (coefficient.to_i64()?, parsed.constant.to_i64()?);
        if scale == 0 {
            return None;
        }

        let pins_exactly = matches!(self.var_to_constraint.get(&var), Some(Constraint::Eq(_, _)));
        let domain = if pins_exactly {
            solve_for_equality(scale, offset)
        } else {
            // A strict integer inequality tightens by exactly one whatever the
            // coefficient's sign, because `scale * term` is itself an integer.
            match parsed.constraint_type {
                ArithConstraintType::Le => solve_for_inequality(scale, offset, true),
                ArithConstraintType::Lt => solve_for_inequality(scale, offset - 1, true),
                ArithConstraintType::Ge => solve_for_inequality(scale, offset, false),
                ArithConstraintType::Gt => solve_for_inequality(scale, offset + 1, false),
            }
        };
        Some((term, domain))
    }
}

/// The domain `scale * x = target` leaves `x`.
///
/// A `target` that is not a multiple of `scale` has no integer solution, and
/// the rounded endpoints then cross (`low > high`), which
/// [`IntDomain::enumerable`] reports as nothing to enumerate.
fn solve_for_equality(scale: i64, target: i64) -> IntDomain {
    // Both endpoints are the same real quotient, rounded the two opposite
    // ways, so neither depends on the coefficient's sign: an exact division
    // collapses them onto the single solution, and an inexact one leaves them
    // crossed.
    IntDomain {
        low: integer_quotient(target, scale, Rounding::Up),
        high: integer_quotient(target, scale, Rounding::Down),
    }
}

/// The domain `scale * x <= target` (`upper_bound`) or `scale * x >= target`
/// (otherwise) leaves `x`.
///
/// Dividing through by a negative coefficient reverses the relation, so which
/// side of `x` the bound lands on depends on both arguments' signs together.
fn solve_for_inequality(scale: i64, target: i64, upper_bound: bool) -> IntDomain {
    if upper_bound == (scale > 0) {
        IntDomain::from_above(integer_quotient(target, scale, Rounding::Down))
    } else {
        IntDomain::from_below(integer_quotient(target, scale, Rounding::Up))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rounding down must go toward negative infinity in all four sign
    /// quadrants -- the property that distinguishes it from Rust's `/`, which
    /// truncates toward zero and would answer `-2` for `-9 / 4`.
    #[test]
    fn rounding_down_goes_toward_negative_infinity() {
        let down = |n, d| integer_quotient(n, d, Rounding::Down);
        assert_eq!(down(9, 4), Some(2));
        assert_eq!(down(-9, 4), Some(-3));
        assert_eq!(down(9, -4), Some(-3));
        assert_eq!(down(-9, -4), Some(2));
        // Exact division is unaffected by the rounding direction.
        assert_eq!(down(-12, 4), Some(-3));
        assert_eq!(down(-12, -4), Some(3));
        // A remainder smaller than the divisor still rounds down, not to zero.
        assert_eq!(down(1, 5), Some(0));
        assert_eq!(down(-1, 5), Some(-1));
    }

    /// Rounding up is the mirror image: toward positive infinity.
    #[test]
    fn rounding_up_goes_toward_positive_infinity() {
        let up = |n, d| integer_quotient(n, d, Rounding::Up);
        assert_eq!(up(9, 4), Some(3));
        assert_eq!(up(-9, 4), Some(-2));
        assert_eq!(up(9, -4), Some(-2));
        assert_eq!(up(-9, -4), Some(3));
        assert_eq!(up(-12, 4), Some(-3));
        assert_eq!(up(1, 5), Some(1));
        assert_eq!(up(-1, 5), Some(0));
    }

    /// The `i64::MIN` negation edge has to answer `None` rather than panic
    /// under debug overflow checks: a missing bound only costs precision,
    /// since an open side simply makes the term ineligible for splitting.
    #[test]
    fn the_unrepresentable_negation_edge_declines() {
        let down = |n, d| integer_quotient(n, d, Rounding::Down);
        let up = |n, d| integer_quotient(n, d, Rounding::Up);
        // Rounding down negates both operands when the divisor is negative,
        // so a numerator of `i64::MIN` has no representable negation.
        assert_eq!(down(i64::MIN, -1), None);
        assert_eq!(down(i64::MIN, -4), None);
        // A positive divisor stays on the exact `div_euclid` path.
        assert_eq!(down(i64::MIN, 1), Some(i64::MIN));
        // Rounding up negates the numerator up front, so `i64::MIN` declines
        // whatever the divisor's sign.
        assert_eq!(up(i64::MIN, 1), None);
        assert_eq!(up(i64::MIN, -1), None);
        // And it declines when the *result* would be `-i64::MIN`.
        assert_eq!(up(i64::MAX, -1), Some(-i64::MAX));
    }

    /// A positive coefficient preserves the relation's direction: an upper
    /// bound on `c*x` stays an upper bound on `x`, rounded inward so what is
    /// left remains satisfiable over the integers.
    #[test]
    fn a_positive_coefficient_preserves_the_bound_direction() {
        // 3x <= 11  =>  x <= 3   (11/3 = 3.67, rounded down)
        assert_eq!(
            solve_for_inequality(3, 11, true),
            IntDomain::from_above(Some(3))
        );
        // 3x >= 11  =>  x >= 4   (rounded up)
        assert_eq!(
            solve_for_inequality(3, 11, false),
            IntDomain::from_below(Some(4))
        );
        // A negative right-hand side rounds the same way, toward the interior.
        // 4x <= -9  =>  x <= -3
        assert_eq!(
            solve_for_inequality(4, -9, true),
            IntDomain::from_above(Some(-3))
        );
        // 4x >= -9  =>  x >= -2
        assert_eq!(
            solve_for_inequality(4, -9, false),
            IntDomain::from_below(Some(-2))
        );
        // A unit coefficient is the identity case.
        assert_eq!(
            solve_for_inequality(1, -7, true),
            IntDomain::from_above(Some(-7))
        );
    }

    /// A negative coefficient flips which side of `x` the bound lands on --
    /// the case that quietly produces an unsound lemma if the sign is
    /// mishandled.
    #[test]
    fn a_negative_coefficient_flips_the_bound_direction() {
        // -3x <= 11  =>  x >= -3   (an upper bound on -3x bounds x below)
        assert_eq!(
            solve_for_inequality(-3, 11, true),
            IntDomain::from_below(Some(-3))
        );
        // -3x >= 11  =>  x <= -4
        assert_eq!(
            solve_for_inequality(-3, 11, false),
            IntDomain::from_above(Some(-4))
        );
        // Negative coefficient and negative right-hand side together.
        // -4x <= -9  =>  x >= 3
        assert_eq!(
            solve_for_inequality(-4, -9, true),
            IntDomain::from_below(Some(3))
        );
        // -4x >= -9  =>  x <= 2
        assert_eq!(
            solve_for_inequality(-4, -9, false),
            IntDomain::from_above(Some(2))
        );
    }

    /// Solving `scale * x = target` must place both endpoints correctly, and
    /// must produce a crossed (empty) range when there is no integer solution.
    #[test]
    fn an_equality_pins_both_endpoints() {
        // 5x = 20  =>  x = 4
        assert_eq!(
            solve_for_equality(5, 20),
            IntDomain {
                low: Some(4),
                high: Some(4)
            }
        );
        // -5x = 20  =>  x = -4
        assert_eq!(
            solve_for_equality(-5, 20),
            IntDomain {
                low: Some(-4),
                high: Some(-4)
            }
        );
        // -5x = -20  =>  x = 4
        assert_eq!(
            solve_for_equality(-5, -20),
            IntDomain {
                low: Some(4),
                high: Some(4)
            }
        );
        // 3x = 7 has no integer solution: the endpoints cross, and the
        // crossed range is not enumerable rather than being a contradiction.
        let no_solution = solve_for_equality(3, 7);
        assert_eq!(
            no_solution,
            IntDomain {
                low: Some(3),
                high: Some(2)
            }
        );
        assert!(no_solution.enumerable(MAX_ENUMERABLE_SPAN).is_none());
    }

    /// Narrowing must intersect with what is stored, never replace it, and
    /// must treat the two sides independently.
    #[test]
    fn narrowing_only_ever_tightens() {
        let mut domain = IntDomain::open();
        assert!(domain.is_open());

        domain.narrow_with(IntDomain::from_above(Some(20)));
        assert_eq!(domain.high, Some(20));
        // A weaker upper bound has no effect.
        domain.narrow_with(IntDomain::from_above(Some(30)));
        assert_eq!(domain.high, Some(20));
        // A stronger one does.
        domain.narrow_with(IntDomain::from_above(Some(12)));
        assert_eq!(domain.high, Some(12));
        // The lower side is tracked independently.
        domain.narrow_with(IntDomain::from_below(Some(-4)));
        assert_eq!((domain.low, domain.high), (Some(-4), Some(12)));
        domain.narrow_with(IntDomain::from_below(Some(-9)));
        assert_eq!((domain.low, domain.high), (Some(-4), Some(12)));
        // Both sides at once, from an equality-derived domain.
        domain.narrow_with(solve_for_equality(1, 5));
        assert_eq!((domain.low, domain.high), (Some(5), Some(5)));

        // A bound-free atom leaves the domain open, and an open domain is
        // never a split candidate.
        let mut untouched = IntDomain::open();
        untouched.narrow_with(IntDomain::open());
        assert!(untouched.is_open());
        assert!(untouched.enumerable(MAX_ENUMERABLE_SPAN).is_none());

        // Tightening past the crossing point reads as empty rather than as a
        // contradiction.
        let mut crossed = IntDomain {
            low: Some(0),
            high: Some(5),
        };
        crossed.narrow_with(IntDomain::from_below(Some(9)));
        assert_eq!((crossed.low, crossed.high), (Some(9), Some(5)));
        assert!(crossed.enumerable(MAX_ENUMERABLE_SPAN).is_none());
    }

    /// Only a two-sided domain no wider than the limit is enumerable.
    #[test]
    fn enumerability_needs_two_sides_and_a_short_span() {
        let two_valued = IntDomain {
            low: Some(0),
            high: Some(1),
        };
        assert_eq!(
            two_valued
                .enumerable(MAX_ENUMERABLE_SPAN)
                .map(|values| values.collect::<Vec<_>>()),
            Some(vec![0, 1])
        );
        // One open side is not enough, however tight the other is.
        assert!(
            IntDomain::from_below(Some(0))
                .enumerable(MAX_ENUMERABLE_SPAN)
                .is_none()
        );
        assert!(
            IntDomain::from_above(Some(0))
                .enumerable(MAX_ENUMERABLE_SPAN)
                .is_none()
        );
        // Exactly at the limit is still enumerable; one past it is not.
        let at_limit = IntDomain {
            low: Some(0),
            high: Some(MAX_ENUMERABLE_SPAN),
        };
        assert!(at_limit.enumerable(MAX_ENUMERABLE_SPAN).is_some());
        let past_limit = IntDomain {
            low: Some(0),
            high: Some(MAX_ENUMERABLE_SPAN + 1),
        };
        assert!(past_limit.enumerable(MAX_ENUMERABLE_SPAN).is_none());
        // A range spanning most of `i64` must not wrap into an accepted width.
        let enormous = IntDomain {
            low: Some(i64::MIN),
            high: Some(i64::MAX),
        };
        assert!(enormous.enumerable(MAX_ENUMERABLE_SPAN).is_none());
    }

    /// An unaffordable refinement round must mark the candidate unverified.
    ///
    /// This is the whole load-dependence fix in one assertion. The ceiling is
    /// a wall-clock test, so whether the round runs depends on how busy the
    /// machine is; if declining it silently left the candidate looking fully
    /// branched, the same query would answer `unsat` on an idle box and `sat`
    /// under load. `note_unaffordable_case_split` is what the `check_core`
    /// gate calls instead of short-circuiting the split away, and it must set
    /// the flag whenever targets existed.
    ///
    /// The formula below is satisfiable, so the search really does stop at a
    /// candidate model with a live split target (an `unsat` one would be
    /// refuted outright and never reach the gate). What is being pinned is the
    /// bookkeeping, not this formula's verdict.
    #[test]
    fn an_unaffordable_round_marks_the_candidate_unverified() {
        use oxiz_core::ast::TermManager;

        // `a` is pinned to `{2, 3}` and used as a UF argument: a shared term
        // with a narrow domain, i.e. exactly what `collect_split_targets`
        // looks for.
        let mut solver = Solver::new();
        // LIA mode: the refinement is integer-only, and `Solver::new` starts in
        // the real-arithmetic default.
        solver.set_logic("QF_UFLIA");
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let a = tm.mk_var("a", int_sort);
        let two = tm.mk_int(2);
        let three = tm.mk_int(3);
        let ge = tm.mk_ge(a, two);
        let le = tm.mk_le(a, three);
        solver.assert(ge, &mut tm);
        solver.assert(le, &mut tm);
        let fa = tm.mk_apply("f", vec![a], int_sort);
        let zero = tm.mk_int(0);
        let fa_ge = tm.mk_ge(fa, zero);
        solver.assert(fa_ge, &mut tm);

        let _ = solver.check(&mut tm);
        // Rewind the refinement bookkeeping to the state the ceiling creates:
        // a candidate model whose round has not been spent. `case_split_terms`
        // is cleared too, since a term already split is deliberately not a
        // target a second time.
        solver.case_split_rounds = 0;
        solver.case_split_skipped_targets = false;
        solver.case_split_terms.clear();

        solver.note_unaffordable_case_split();
        assert!(
            solver.case_split_skipped_targets,
            "a declined round with live targets must mark the candidate unverified"
        );
    }

    /// The converse: with nothing to split, a declined round leaves the
    /// candidate alone. Otherwise every slow `sat` in the solver would be
    /// downgraded to `unknown` and the gate would cost far more than it buys.
    #[test]
    fn an_unaffordable_round_with_no_targets_changes_nothing() {
        use oxiz_core::ast::TermManager;

        // Pure arithmetic, no uninterpreted function: no shared term, so
        // `collect_split_targets` finds nothing.
        let mut solver = Solver::new();
        // Same LIA mode as the test above, so this really is exercising "no
        // targets" rather than the integer-mode guard.
        solver.set_logic("QF_UFLIA");
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let zero = tm.mk_int(0);
        let ge = tm.mk_ge(x, zero);
        solver.assert(ge, &mut tm);
        let _ = solver.check(&mut tm);
        solver.case_split_rounds = 0;
        solver.case_split_skipped_targets = false;

        solver.note_unaffordable_case_split();
        assert!(
            !solver.case_split_skipped_targets,
            "no targets means nothing was skipped"
        );
    }
}
