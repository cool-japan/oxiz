//! The reserved SMT-LIB `RoundingMode` sort's five terms.
//!
//! `RoundingMode` is a first-class sort in OxiZ (`SortKind::RoundingMode`),
//! but it deliberately adds **no** `TermKind` variant: its five inhabitants
//! are ordinary nullary `Var` terms interned at that sort. That single choice
//! is what makes `(declare-const m RoundingMode)` work without touching term
//! traversal, theory dispatch, substitution, printing or proof logging — a
//! rounding mode is, structurally, just another nullary symbol, and EUF
//! already decides equalities between nullary symbols.
//!
//! What the sort does *not* carry on its own is its cardinality. Nothing here
//! makes the five modes pairwise distinct or stops a sixth element from
//! existing; `oxiz-solver`'s `context::rounding_mode` asserts both facts.

use super::super::term::{RoundingMode, TermId};
use super::TermManager;

impl TermManager {
    /// The canonical SMT-LIB spelling of a rounding mode.
    ///
    /// The *long* form, deliberately: it is the name every mode term is
    /// interned under, so the short alias `RNE` and the long
    /// `roundNearestTiesToEven` denote the identical [`TermId`], and it is the
    /// spelling `get-model` emits — matching Z3's output.
    #[must_use]
    pub const fn rounding_mode_name(rm: RoundingMode) -> &'static str {
        match rm {
            RoundingMode::RNE => "roundNearestTiesToEven",
            RoundingMode::RNA => "roundNearestTiesToAway",
            RoundingMode::RTP => "roundTowardPositive",
            RoundingMode::RTN => "roundTowardNegative",
            RoundingMode::RTZ => "roundTowardZero",
        }
    }

    /// The term denoting rounding mode `rm`: a nullary
    /// [`Var`](crate::ast::TermKind::Var) at the reserved `RoundingMode`
    /// sort.
    ///
    /// Nullary means `NelsonOppen::theory_of` already classifies it as
    /// `Shared`, so EUF decides `(= m RNE)` with no theory-dispatch change and
    /// no new `TermKind` variant. The five modes are *distinct* only because
    /// the solver asserts it (see `oxiz-solver`'s `context::rounding_mode`);
    /// this layer only guarantees they are five different terms.
    pub fn mk_rounding_mode(&mut self, rm: RoundingMode) -> TermId {
        self.rounding_mode_used = true;
        let sort = self.sorts.rounding_mode_sort;
        self.mk_var(Self::rounding_mode_name(rm), sort)
    }

    /// Whether anything in this manager has introduced a rounding mode — a
    /// mode term, or a parser-accepted `RoundingMode` declaration.
    #[must_use]
    pub fn rounding_mode_used(&self) -> bool {
        self.rounding_mode_used
    }

    /// Record that a `RoundingMode`-sorted symbol was declared, even though no
    /// mode *term* has been built yet.
    pub fn note_rounding_mode_used(&mut self) {
        self.rounding_mode_used = true;
    }
}
