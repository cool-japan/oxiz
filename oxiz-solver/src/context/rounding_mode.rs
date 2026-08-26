//! Cardinality axioms for the reserved `RoundingMode` sort.
//!
//! `oxiz_core::sort::SortKind::RoundingMode` gives rounding modes a sort of
//! their own, and `TermManager::mk_rounding_mode` gives the five IEEE 754
//! modes five nullary terms at that sort — but *neither* says anything about
//! how many elements the sort has. To the EUF core a rounding mode is just an
//! uninterpreted constant, and an uninterpreted sort is infinite.
//!
//! The finite five-element domain is supplied here, by two axioms that the
//! `Context` asserts on the solver:
//!
//! 1. **Closure**, one per declared `RoundingMode` constant `m`:
//!    `(or (= m RNE) (= m RNA) (= m RTP) (= m RTN) (= m RTZ))`.
//!    Without it a sixth value could exist. That matters concretely, not just
//!    theoretically: the parser compiles `(fp.add m x y)` into a five-way
//!    `ite` whose final, unguarded branch is `RTZ` (see
//!    `Parser::expand_symbolic_rm`), so a sixth value would be *evaluated as*
//!    `RTZ` while still comparing distinct from all five — and
//!    `(distinct m1 m2 m3 m4 m5 m6)` would answer `sat`.
//!
//! 2. **Distinctness**, once per solve whenever any rounding mode is in play:
//!    `(distinct RNE RNA RTP RTN RTZ)`.
//!    Without it the five mode constants could be collapsed into fewer
//!    classes, and `(distinct m1 m2 m3 m4 m5)` over five closed constants
//!    would wrongly answer `unsat`.
//!
//! Both together are what make the domain *exactly* five. Either one alone is
//! unsound in a different direction, so neither is optional.
//!
//! # Why these are asserted on the solver, not through `Context::assert`
//!
//! [`Context::assert`](super::Context::assert) also pushes onto
//! `Context::assertions`, which is what `(get-assertions)` reports back to the
//! user. These axioms are OxiZ's internal encoding of a built-in sort, not
//! something the user wrote, so they go straight to `Solver::assert` and stay
//! out of that list.

use super::Context;
use oxiz_core::ast::{RoundingMode, TermId};
use oxiz_core::sort::SortId;

impl Context {
    /// Whether `sort` is the reserved `RoundingMode` sort.
    pub(super) fn is_rounding_mode_sort(&self, sort: SortId) -> bool {
        sort == self.terms.sorts.rounding_mode_sort
    }

    /// Assert the closure axiom pinning `sym` to one of the five real modes.
    ///
    /// Called from [`Context::declare_const`](super::Context::declare_const)
    /// for every constant declared at the `RoundingMode` sort — which is the
    /// only position the parser accepts the sort in, precisely because this
    /// axiom needs a single symbol to attach to.
    pub(super) fn assert_rounding_mode_closure(&mut self, sym: TermId) {
        let alternatives: Vec<TermId> = RoundingMode::ALL
            .iter()
            .map(|&rm| {
                let mode = self.terms.mk_rounding_mode(rm);
                self.terms.mk_eq(sym, mode)
            })
            .collect();
        let closure = self.terms.mk_or(alternatives);
        self.solver.assert(closure, &mut self.terms);
    }

    /// Assert that the five rounding modes are pairwise distinct.
    ///
    /// Called from [`Context::check_sat`](super::Context::check_sat) whenever
    /// `TermManager::rounding_mode_used` reports that a mode term or a
    /// `RoundingMode` declaration exists, so a script that never mentions a
    /// rounding mode pays nothing.
    ///
    /// Deliberately unguarded by any "already asserted" flag and unscoped with
    /// respect to `push`/`pop`. The axiom is a *fact*, not a user assertion:
    /// re-stating it is logically idempotent (the term is hash-consed, so it is
    /// the identical `TermId` every time), and re-stating it after a `pop` that
    /// retracted an earlier copy is exactly the behaviour wanted — a flag would
    /// have to be un-set by that `pop` to stay correct, which is strictly more
    /// machinery for the same result. The price is a duplicate clause per
    /// `check-sat` in a script that both uses rounding modes and re-checks
    /// repeatedly; the SAT core tolerates duplicate clauses, and correctness
    /// never depends on the count.
    pub(super) fn assert_rounding_mode_distinctness(&mut self) {
        let modes: Vec<TermId> = RoundingMode::ALL
            .iter()
            .map(|&rm| self.terms.mk_rounding_mode(rm))
            .collect();
        let distinct = self.terms.mk_distinct(modes);
        self.solver.assert(distinct, &mut self.terms);
    }
}
