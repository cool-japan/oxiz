//! The bit-vector bridge between [`TheoryManager`] and `oxiz_theories`' bit-blasting
//! [`BvSolver`].
//!
//! Split out of `theory_manager.rs` to keep that file under the workspace's
//! 2000-line ceiling; it is the same `impl<'a> TheoryManager<'a>` block, moved
//! verbatim.
//!
//! Everything a bit-vector atom needs on its way from the CDCL(T) search into
//! the bit-blaster lives here: the sort lookup that decides whether an atom is
//! bit-vector-shaped at all, the recursive encoding of both operands, and the
//! three-way translation of the bit-blasted verdict back into a
//! [`TheoryCheckResult`]. `bv_run_check` is the funnel every bit-vector atom
//! passes through, and therefore where two soundness rules are enforced: an
//! atom the circuit was never told about (`asserted == false`) must not have
//! the bit-blaster's `Sat` read as evidence about it (U-Z10's sibling family),
//! and an exhausted or failed bit-blasted check must not read as "no conflict"
//! (U-Z12, once the embedded solver gained a budget).

use super::TheoryManager;
use crate::prelude::*;
use crate::solver::theory_bv_encode::{debug_verify_bv_circuits, encode_bv_term_recursive};
use oxiz_core::ast::{TermId, TermManager};
use oxiz_sat::TheoryCheckResult;

impl TheoryManager<'_> {
    /// Look up the BV bit-width of a term from its sort, if it has a BV sort.
    pub(super) fn bv_width_of(&self, term: TermId, manager: &TermManager) -> Option<u32> {
        manager
            .get(term)
            .and_then(|t| manager.sorts.get(t.sort))
            .and_then(|s| s.bitvec_width())
    }

    /// Bit-blast both operands of a BV constraint into the embedded SAT solver.
    ///
    /// Each side is encoded recursively; a bare leaf that the recursive encoder
    /// cannot handle falls back to a fresh BV variable of the operand's width.
    /// Returns `true` if both operands are BV-sorted with equal width (so that
    /// `assert_eq` / `assert_neq` may be called safely), `false` otherwise.
    pub(super) fn bit_blast_bv_pair(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        manager: &TermManager,
    ) -> bool {
        let (lw, rw) = match (
            self.bv_width_of(lhs, manager),
            self.bv_width_of(rhs, manager),
        ) {
            (Some(lw), Some(rw)) if lw == rw => (lw, rw),
            _ => return false,
        };
        let mut encoded: FxHashSet<TermId> = FxHashSet::default();
        if !encode_bv_term_recursive(self.bv, lhs, manager, &mut encoded) {
            self.bv.new_bv(lhs, lw);
        }
        if !encode_bv_term_recursive(self.bv, rhs, manager, &mut encoded) {
            self.bv.new_bv(rhs, rw);
        }
        true
    }

    /// Run the embedded BV SAT check after the caller has asserted a constraint.
    ///
    /// Records `constraint_term` so the conflict clause is non-empty, then
    /// returns `Some(Conflict(..))` if the embedded solver reports UNSAT and
    /// `None` otherwise (so the caller falls through to its conservative path).
    ///
    /// `operands` are the two sides of the atom just asserted.  When the check
    /// comes back SAT they are handed to [`debug_verify_bv_circuits`], the
    /// debug-only model-validity net: every bit-blasted node under them must
    /// reproduce its own operation concretely on the model the solver just
    /// found.  That is the check which distinguishes "the search is right" from
    /// "the circuit is wrong", and it costs nothing in release builds.
    ///
    /// `asserted` is the caller's own answer to "did an `assert_*` actually
    /// reach the circuit for this atom".  When it is `false` the circuit was
    /// never *told* about the atom, so its `Sat` is not evidence about it: the
    /// atom is recorded on [`Self::bv_atom_unmodelled`] and this returns `None`,
    /// letting the CDCL(T) loop continue while the owning `Solver` degrades a
    /// final `Sat` to `Unknown`.  The constraint term is deliberately *not*
    /// recorded — no clause depends on it, so it does not belong in a conflict
    /// explanation.
    pub(super) fn bv_run_check(
        &mut self,
        constraint_term: TermId,
        operands: (TermId, TermId),
        manager: &TermManager,
        asserted: bool,
    ) -> Option<TheoryCheckResult> {
        use oxiz_theories::Theory;
        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
        if !asserted {
            self.bv_atom_unmodelled = true;
            return None;
        }
        self.bv.record_constraint_term(constraint_term);
        match self.bv.check() {
            Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) => {
                Some(self.conflict_from_terms(&conflict_terms))
            }
            Ok(TheoryCheckResultEnum::Sat) => {
                debug_verify_bv_circuits(self.bv, operands.0, manager);
                debug_verify_bv_circuits(self.bv, operands.1, manager);
                None
            }
            // A propagation carries no verdict about this atom; fall through to
            // the caller's conservative path exactly as before.
            Ok(TheoryCheckResultEnum::Propagate(_)) => None,
            // An exhausted or failed bit-blasted check is NOT "no conflict".
            //
            // Before the embedded solver had a budget these two were
            // unreachable, and a `_ => None` arm was harmless.  The moment
            // `(set-option :timeout N)` / `(set-option :max-conflicts N)` can
            // stop an embedded `solve()` (U-Z12), reading its `Unknown` as
            // "consistent" would let the CDCL(T) loop fall through to
            // `TheoryCheckResult::Sat` at the tail of `process_constraint` and
            // report a `sat` that no bit-blasted check ever confirmed.  Setting
            // `resource_exhausted` and answering `Some(Sat)` stops the search
            // the way the conflict-limit path already does, and the flag makes
            // the owning `Solver` answer `unknown`.  `Unsat` is unaffected:
            // dropping a theory conflict only weakens the clause set, and
            // `Unsat` of a weakening implies `Unsat` of the original.
            //
            // Spelled out per variant rather than left as `_` so a new
            // `TheoryCheckResult` variant is a compile error here instead of
            // silently inheriting whichever behaviour the wildcard had.
            Ok(TheoryCheckResultEnum::Unknown) | Err(_) => {
                self.resource_exhausted = true;
                Some(TheoryCheckResult::Sat)
            }
        }
    }

    /// Bit-blast `lhs`/`rhs`, assert `lhs != b` at the bit level, and check.
    ///
    /// Returns `Some(Conflict(..))` on a detected BV theory conflict, `None`
    /// otherwise (including when the operands are not equal-width BV terms).
    pub(super) fn bv_check_neq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        constraint_term: TermId,
        manager: &TermManager,
    ) -> Option<TheoryCheckResult> {
        if !self.bit_blast_bv_pair(lhs, rhs, manager) {
            return None;
        }
        let asserted = self.bv.assert_neq(lhs, rhs);
        self.bv_run_check(constraint_term, (lhs, rhs), manager, asserted)
    }

    /// Bit-blast `lhs`/`rhs`, assert `lhs = b` at the bit level, and check.
    ///
    /// Returns `Some(Conflict(..))` on a detected BV theory conflict, `None`
    /// otherwise (including when the operands are not equal-width BV terms).
    pub(super) fn bv_check_eq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        constraint_term: TermId,
        manager: &TermManager,
    ) -> Option<TheoryCheckResult> {
        if !self.bit_blast_bv_pair(lhs, rhs, manager) {
            return None;
        }
        let asserted = self.bv.assert_eq(lhs, rhs);
        self.bv_run_check(constraint_term, (lhs, rhs), manager, asserted)
    }
}
