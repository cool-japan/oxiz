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
    ///
    /// Iterative (explicit work stack, `visited` set on the hash-consed DAG),
    /// so nesting depth cannot overflow the native call stack; children are
    /// pushed in reverse so registration keeps the original left-to-right
    /// pre-order.
    pub(super) fn collect_ground_candidates_from_term(
        &mut self,
        term: TermId,
        manager: &TermManager,
    ) {
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = vec![term];
        while let Some(term) = stack.pop() {
            if !visited.insert(term) {
                continue;
            }
            let Some(t) = manager.get(term) else {
                continue;
            };
            match &t.kind {
                TermKind::Apply { func, args } => {
                    // Only register Skolem function applications as candidates.
                    // Non-Skolem Apply terms (like ack(0,1)) should NOT be
                    // used as integer candidates — using them would create
                    // nested applications (ack(ack(0,0), n)) that produce
                    // spurious conflicts.
                    //
                    // Matched through `is_reserved_tag`, the read-side twin of
                    // the `reserved_name` the Skolemizer mints with: the old
                    // `starts_with("sk")` test both matched a user's own `skew`
                    // and would have gone silently dead when the Skolem symbols
                    // moved into the reserved `\oxiz.` class.
                    let fname = manager.resolve_str(*func);
                    if oxiz_core::smtlib::is_reserved_tag(fname, "sk")
                        || oxiz_core::smtlib::is_reserved_tag(fname, "skf")
                    {
                        self.mbqi.add_candidate(term, t.sort);
                    }
                    stack.extend(args.iter().rev().copied());
                }
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args) => {
                    stack.extend(args.iter().rev().copied());
                }
                TermKind::Not(a) | TermKind::Neg(a) => {
                    stack.push(*a);
                }
                // Array select terms are useful candidates for array theory.
                TermKind::Select(a, b)
                | TermKind::Implies(a, b)
                | TermKind::Eq(a, b)
                | TermKind::Lt(a, b)
                | TermKind::Le(a, b)
                | TermKind::Gt(a, b)
                | TermKind::Ge(a, b)
                | TermKind::Sub(a, b)
                | TermKind::Div(a, b)
                | TermKind::Mod(a, b) => {
                    stack.push(*b);
                    stack.push(*a);
                }
                TermKind::Ite(a, b, c) | TermKind::Store(a, b, c) => {
                    stack.push(*c);
                    stack.push(*b);
                    stack.push(*a);
                }
                _ => {}
            }
        }
    }
}
