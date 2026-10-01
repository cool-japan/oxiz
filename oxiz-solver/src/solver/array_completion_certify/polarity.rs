//! The maximal quantifier sub-terms of a goal, **with their polarity**
//! (`#P2b-51`, decision (54)(ii)).
//!
//! The certificate substitutes a truth value for every maximal quantifier
//! sub-term it has certified and then checks the assertions, so what it needs
//! to know about each sub-term is which value to certify.  Before this module
//! the walk knew one polarity only — "every connective above is an `and` /
//! `or`" — and declined every `exists` anywhere else.  But an `exists` under a
//! negation *is* a universal: `(not (exists ((j S)) φ))` holds exactly when
//! `∀j. ¬φ` does, and `(=> (exists ((j S)) φ) false)` is the same formula.
//! Declining it kept the candidate model, which printed `a` as `#b0` at every
//! point the search had not pinned beside `¬∃j. a[j] = #b0` (recheck 13,
//! `c04`, `d02`, `d05`).
//!
//! So the walk tracks polarity through `not`, an implication's antecedent and
//! `and` / `or`, and every sub-term whose polarity is fixed is certified at
//! the value that polarity needs:
//!
//! | sub-term | positive | negative |
//! |---|---|---|
//! | `∀x⃗. φ` | certified **true**: `φ` valid | certified **false**: a witness of `¬φ` |
//! | `∃x⃗. φ` | certified **true**: a witness of `φ` | certified **false**: `¬φ` valid |
//!
//! The negative cases are handed to the certificate as a synthetic *dual* —
//! `∃x⃗. ¬φ` for a negated `∀`, `∀x⃗. ¬φ` for a negated `∃` — listed with the
//! ordinary universals and existentials, so the peel, the sample, the fill
//! query and the pre-filter treat it exactly like a quantifier the script
//! spelled; the original sub-term is then substituted by `false`.  Certifying
//! a sub-term's actual value and substituting it is sound at *any* polarity,
//! so a sub-term under an `ite` condition, a Boolean `=` / `distinct` / `xor`
//! or an uninterpreted argument (both polarities) is certified `true` as
//! before; only which value is tried depends on the polarity.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};

/// The polarity of a sub-term's occurrence in an assertion.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Polarity {
    /// Every connective above is monotone (`and`, `or`, an implication's
    /// consequent) under an even number of negations.
    Positive,
    /// The same under an odd number of negations.
    Negative,
    /// Anything else: an `ite` condition, an operand of a Boolean `=`,
    /// `distinct` or `xor`, an argument of an uninterpreted function.
    Both,
}

impl Polarity {
    fn flip(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
            Self::Both => Self::Both,
        }
    }

    /// The polarity of a sub-term met at `self` and at `other`.
    fn join(self, other: Self) -> Self {
        if self == other { self } else { Self::Both }
    }
}

/// The maximal quantifier sub-terms of a goal, split by the value the
/// certificate proves for each.
#[derive(Default)]
pub(super) struct Quantifiers {
    /// Universals to certify `true`: the positive (or two-sided) `forall`
    /// sub-terms, and the dual `∀x⃗. ¬φ` of every negative `exists`.
    pub(super) universals: Vec<TermId>,
    /// Existentials to certify `true` by a witness: the positive (or
    /// two-sided) `exists` sub-terms, and the dual `∃x⃗. ¬φ` of every negative
    /// `forall`.
    pub(super) existentials: Vec<TermId>,
    /// `(original, dual)`: the original sub-term is `false` exactly when its
    /// dual (listed above) is `true`.
    pub(super) negations: Vec<(TermId, TermId)>,
}

/// Collect every maximal quantifier sub-term of `assertions` with its
/// polarity, or `None` when the walk meets something it cannot read (a term
/// the manager does not know).
pub(super) fn collect_quantifiers(
    assertions: &[TermId],
    manager: &mut TermManager,
) -> Option<Quantifiers> {
    let mut found: FxHashMap<TermId, Polarity> = FxHashMap::default();
    let mut stack: Vec<(TermId, Polarity)> = assertions
        .iter()
        .map(|&assertion| (assertion, Polarity::Positive))
        .collect();
    let mut visited: FxHashSet<(TermId, Polarity)> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some((current, polarity)) = stack.pop() {
        if !visited.insert((current, polarity)) {
            continue;
        }
        let data = manager.get(current)?;
        match &data.kind {
            TermKind::Forall { .. } | TermKind::Exists { .. } => {
                // Deliberately not descended into: the sub-term is handled as
                // a whole by the peel.
                let entry = found.entry(current).or_insert(polarity);
                *entry = entry.join(polarity);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                stack.extend(args.iter().map(|&arg| (arg, polarity)));
            }
            TermKind::Not(inner) => stack.push((*inner, polarity.flip())),
            TermKind::Implies(antecedent, consequent) => {
                stack.push((*antecedent, polarity.flip()));
                stack.push((*consequent, polarity));
            }
            TermKind::Ite(condition, then_branch, else_branch)
                if data.sort == manager.sorts.bool_sort =>
            {
                stack.push((*condition, Polarity::Both));
                stack.push((*then_branch, polarity));
                stack.push((*else_branch, polarity));
            }
            kind => {
                children.clear();
                children.extend(oxiz_core::ast::traversal::get_children(kind));
                stack.extend(children.iter().map(|&child| (child, Polarity::Both)));
            }
        }
    }

    // The walk visits a hash-consed DAG, so `stack` order decides the order
    // the map was filled.  Sort by term id so the certificate's query order is
    // a property of the terms and not of the traversal.
    let mut ordered: Vec<(TermId, Polarity)> = found.into_iter().collect();
    ordered.sort_unstable_by_key(|&(term, _)| term.raw());

    let mut out = Quantifiers::default();
    for (term, polarity) in ordered {
        let is_forall = matches!(
            manager.get(term).map(|d| &d.kind),
            Some(TermKind::Forall { .. })
        );
        match (is_forall, polarity) {
            (true, Polarity::Positive | Polarity::Both) => out.universals.push(term),
            (false, Polarity::Positive | Polarity::Both) => out.existentials.push(term),
            (true, Polarity::Negative) => {
                let dual = dual_of(term, manager)?;
                out.existentials.push(dual);
                out.negations.push((term, dual));
            }
            (false, Polarity::Negative) => {
                let dual = dual_of(term, manager)?;
                out.universals.push(dual);
                out.negations.push((term, dual));
            }
        }
    }
    Some(out)
}

/// The dual of a chain of consecutive quantifiers of one kind: `∀x⃗. φ` ↦
/// `∃x⃗. ¬φ` and `∃x⃗. φ` ↦ `∀x⃗. ¬φ`, with the chain's variables bound by one
/// quantifier in binder order (a name bound twice keeps both entries, which
/// the peel resolves to the innermost binding exactly as it does for the
/// chain itself).  A chain that alternates keeps its inner quantifier in the
/// body, and the peel declines it there, as it would the original.
fn dual_of(term: TermId, manager: &mut TermManager) -> Option<TermId> {
    let forall = matches!(manager.get(term)?.kind, TermKind::Forall { .. });
    let mut bound: Vec<(Spur, SortId)> = Vec::new();
    let mut current = term;
    loop {
        match manager.get(current).map(|d| d.kind.clone())? {
            TermKind::Forall { vars, body, .. } if forall => {
                bound.extend(vars.iter().copied());
                current = body;
            }
            TermKind::Exists { vars, body, .. } if !forall => {
                bound.extend(vars.iter().copied());
                current = body;
            }
            _ => break,
        }
    }
    let names: Vec<(String, SortId)> = bound
        .iter()
        .map(|&(name, sort)| (manager.resolve_str(name).to_string(), sort))
        .collect();
    let negated = manager.mk_not(current);
    let vars = names.iter().map(|(name, sort)| (name.as_str(), *sort));
    Some(if forall {
        manager.mk_exists(vars, negated)
    } else {
        manager.mk_forall(vars, negated)
    })
}
