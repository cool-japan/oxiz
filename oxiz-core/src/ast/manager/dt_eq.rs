//! Equalities between two datatype **constructor applications**, decided at
//! construction (`#P2b-61`, `#P2b-76`).
//!
//! Datatypes are free (SMT-LIB 2.6 §4.2.3): two applications of different
//! constructors of one datatype are unequal in every model, and two
//! applications of the same constructor are equal exactly when their fields
//! are — constructors are injective.  So an equality of two constructor
//! applications is never an atom the search has to decide:
//!
//! * `(= (cons 1 x) nil)` is `false`;
//! * `(= (cons 1 x) (cons 1 y))` is `(= x y)`;
//! * `(= (cons 1 (cons 2 nil)) (cons 1 (cons 2 (cons 3 nil))))` is `false` —
//!   injectivity twice, then two different constructors.
//!
//! Folding it where the equality is built, rather than in one consumer, is what
//! lets the fact reach every place that manufactures such an equality: the
//! read-over-write lemma `(select (store a red 1) green)` builds `(= red
//! green)` itself and never asked the datatype axioms (`#P2b-61`), and the
//! datatype axioms' own congruence lemmas equate two constructor applications
//! whose difference lies several constructors down (`#P2b-76`: the second
//! example above answered `sat` on every build, because the axioms stop one
//! level below the assertion set).  Injectivity *through* an opaque term — two
//! applications the congruence closure merged via a constant — is the solver's
//! half (`solver::dt_refinement`).
//!
//! The decomposition is iterative (a worklist of field pairs), so a literal
//! list of any length folds in constant native stack.

use super::super::term::{TermId, TermKind};
use super::TermManager;
#[cfg(not(feature = "std"))]
use crate::prelude::{Vec, vec};

impl TermManager {
    /// The folded equality of two constructor applications of one sort:
    /// `false` when a constructor differs at some depth, otherwise the
    /// conjunction of the equalities of the fields that are not themselves
    /// both constructor applications.  `None` when `lhs` and `rhs` are not
    /// both constructor applications of one sort (or disagree in arity, which
    /// a well-sorted term never does), so the caller builds the plain atom.
    pub(super) fn fold_constructor_eq(&mut self, lhs: TermId, rhs: TermId) -> Option<TermId> {
        if !self.same_sort_constructors(lhs, rhs) {
            return None;
        }
        let mut pending: Vec<(TermId, TermId)> = vec![(lhs, rhs)];
        let mut conjuncts: Vec<TermId> = Vec::new();
        while let Some((left, right)) = pending.pop() {
            if left == right {
                continue;
            }
            if self.same_sort_constructors(left, right) {
                let (
                    Some(TermKind::DtConstructor {
                        constructor: c1,
                        args: a1,
                    }),
                    Some(TermKind::DtConstructor {
                        constructor: c2,
                        args: a2,
                    }),
                ) = (
                    self.get(left).map(|t| t.kind.clone()),
                    self.get(right).map(|t| t.kind.clone()),
                )
                else {
                    return None;
                };
                if c1 != c2 {
                    return Some(self.false_id);
                }
                if a1.len() != a2.len() {
                    return None;
                }
                // Pushed in reverse so the conjuncts come out left to right.
                for (&x, &y) in a1.iter().zip(a2.iter()).rev() {
                    pending.push((x, y));
                }
                continue;
            }
            let field = self.mk_eq(left, right);
            match self.get(field).map(|t| &t.kind) {
                Some(TermKind::False) => return Some(self.false_id),
                Some(TermKind::True) => {}
                _ => conjuncts.push(field),
            }
        }
        Some(self.mk_and(conjuncts))
    }

    /// The equality **atom** `(= lhs rhs)`, interned as a node and never
    /// folded — ordered as [`TermManager::mk_eq`] orders it, so it is the
    /// atom any unfolded equality of the two terms shares.
    ///
    /// For the one lemma that must *merge* two constructor applications in a
    /// congruence closure that treats them as opaque leaves: the datatype
    /// axioms' "equal arguments build equal values" (`⋀ aᵢ = bᵢ ⇒ C(a⃗) =
    /// C(b⃗)`).  Through [`TermManager::mk_eq`] its conclusion decomposes into
    /// its own premise and the lemma says nothing, so `(item b1) = red`
    /// beside `b1 = (full (item b1))` and `¬(b1 = (full red))` lost its
    /// refutation (`#P2b-76`'s fold, re-fix pass 15).
    pub fn mk_eq_atom(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if lhs == rhs {
            return self.true_id;
        }
        let (lhs, rhs) = if lhs.0 <= rhs.0 {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Eq(lhs, rhs), sort)
    }

    /// Whether `lhs` and `rhs` are both constructor applications of one sort.
    fn same_sort_constructors(&self, lhs: TermId, rhs: TermId) -> bool {
        match (self.get(lhs), self.get(rhs)) {
            (Some(l), Some(r)) => {
                l.sort == r.sort
                    && matches!(l.kind, TermKind::DtConstructor { .. })
                    && matches!(r.kind, TermKind::DtConstructor { .. })
            }
            _ => false,
        }
    }
}
