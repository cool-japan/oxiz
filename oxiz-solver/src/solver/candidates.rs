//! Ground candidate collection for MBQI instantiation.
//!
//! After Skolemization produces terms like `f(0, sk!0(0)) > 0`, we need to
//! walk the resulting term tree and register ground Apply sub-terms (especially
//! Skolem function applications) as MBQI candidates so that subsequent rounds
//! can instantiate other universals with Skolem application values.

use crate::prelude::*;
use oxiz_core::ast::{TermId, TermKind, TermManager};

use super::Solver;

impl Solver {
    /// Walk a term and register ground Apply sub-terms as MBQI candidates.
    ///
    /// After Skolemization produces `f(0, sk!0(0)) > 0`, the sub-term
    /// `sk!0(0)` must become a candidate for Int so that subsequent rounds
    /// can instantiate other universals with Skolem application values.
    pub(super) fn collect_ground_candidates_from_term(
        &mut self,
        term: TermId,
        manager: &TermManager,
    ) {
        let mut visited = FxHashSet::default();
        self.collect_ground_candidates_rec(term, manager, &mut visited);
    }

    pub(super) fn collect_ground_candidates_rec(
        &mut self,
        term: TermId,
        manager: &TermManager,
        visited: &mut FxHashSet<TermId>,
    ) {
        if !visited.insert(term) {
            return;
        }
        let Some(t) = manager.get(term) else {
            return;
        };
        match &t.kind {
            TermKind::Apply { func, args } => {
                // Only register Skolem function applications as candidates.
                // Non-Skolem Apply terms (like ack(0,1)) should NOT be
                // used as integer candidates — using them would create
                // nested applications (ack(ack(0,0), n)) that produce
                // spurious conflicts.
                let fname = manager.resolve_str(*func);
                if fname.starts_with("sk") || fname.starts_with("skf") {
                    self.mbqi.add_candidate(term, t.sort);
                }
                for &a in args.iter() {
                    self.collect_ground_candidates_rec(a, manager, visited);
                }
            }
            TermKind::Select(arr, idx) => {
                // Array select terms are useful candidates for array theory
                self.collect_ground_candidates_rec(*arr, manager, visited);
                self.collect_ground_candidates_rec(*idx, manager, visited);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &a in args {
                    self.collect_ground_candidates_rec(a, manager, visited);
                }
            }
            TermKind::Add(args) | TermKind::Mul(args) => {
                for &a in args.iter() {
                    self.collect_ground_candidates_rec(a, manager, visited);
                }
            }
            TermKind::Not(a) | TermKind::Neg(a) => {
                self.collect_ground_candidates_rec(*a, manager, visited);
            }
            TermKind::Implies(a, b)
            | TermKind::Eq(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b) => {
                self.collect_ground_candidates_rec(*a, manager, visited);
                self.collect_ground_candidates_rec(*b, manager, visited);
            }
            TermKind::Ite(c, t_br, e) => {
                self.collect_ground_candidates_rec(*c, manager, visited);
                self.collect_ground_candidates_rec(*t_br, manager, visited);
                self.collect_ground_candidates_rec(*e, manager, visited);
            }
            TermKind::Store(a, i, v) => {
                self.collect_ground_candidates_rec(*a, manager, visited);
                self.collect_ground_candidates_rec(*i, manager, visited);
                self.collect_ground_candidates_rec(*v, manager, visited);
            }
            _ => {}
        }
    }
}
