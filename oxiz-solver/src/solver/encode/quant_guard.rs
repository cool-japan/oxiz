//! Polarity-complete handling of the quantifiers an assertion does **not**
//! state unconditionally.
//!
//! # The hole this closes (`#P2b-54`)
//!
//! [`Solver::encode`](crate::solver::Solver) is the Tseitin transform: it gives
//! a `Forall` / `Exists` sub-term one fresh Boolean variable and *no* defining
//! clause.  Nothing in the SAT core says what that variable means.  Its meaning
//! used to come from exactly one place —
//! [`Solver::register_asserted_quantifiers`](crate::solver::Solver), which
//! walks the *unconditionally asserted* spine
//! ([`term_walk::asserted_children`](super::super::term_walk::asserted_children):
//! `And` at positive polarity, `Or` at negative, `Not` flipping) and hands MBQI
//! the universals the assertion entails.
//!
//! Every other Boolean position was therefore an unconstrained Boolean:
//!
//! ```text
//! (declare-sort U 0)
//! (declare-fun f (U) U)
//! (assert (not (forall ((x U)) (= (f x) (f x)))))
//! (check-sat)   ; answered `sat`
//! ```
//!
//! The body is a tautology in every structure, so the assertion is
//! unsatisfiable in every structure — and the search satisfied it by setting
//! the quantifier's free literal to `false`.  The same held under `=>`, a
//! positive `or`, an `ite`, a Boolean `=`, and for `(not (exists …))`.
//!
//! # The rule
//!
//! A quantifier occurrence is an *obligation*, and which obligation it is
//! depends only on its polarity:
//!
//! | occurrence | obligation | discharged by |
//! |---|---|---|
//! | `forall` at positive polarity | universal | instantiation (MBQI / finite expansion) |
//! | `forall` at negative polarity | existential | Skolemisation |
//! | `exists` at positive polarity | existential | Skolemisation |
//! | `exists` at negative polarity | universal | instantiation |
//!
//! This pass replaces every such occurrence `Q` that the spine walk does *not*
//! already register by a fresh reserved Boolean constant `g`
//! ([`reserved_name`]`("qg", n)`) and emits the obligations that define `g`:
//!
//! * positive: `∀x. (g → φ)` — a plain *unconditional* universal, asserted
//!   beside the assertion.  Pushing `g →` inside the binder is an equivalence
//!   because `g` is closed, and it is what makes every instance the existing
//!   machinery produces come out guarded: an MBQI instance of it is
//!   `g → φ(t)`, a finite-expansion conjunct is `g → φ(t)`, and a `binder_row`
//!   read-over-write expansion rewrites `φ` in place.  So `g` can be set true
//!   only where the instantiation fixpoint — or, inside
//!   [`finite_expand`](super::finite_expand)'s budget, the whole-sort
//!   expansion — justifies it.
//! * negative: `g ∨ ¬φ(sk)` for fresh Skolem constants `sk`, which is the body
//!   asserted under the same Boolean context that `g` now carries.  `g` can be
//!   set false only with a witness in hand.
//!
//! An occurrence whose polarity the Boolean skeleton cannot pin down (a
//! quantifier under `xor`, under a Boolean `=`, in an `ite` condition, or as a
//! `UF` argument) gets **both**, which is the full definition `g ↔ Q` and is
//! sound in any position whatsoever.
//!
//! # Why this is equisatisfiable
//!
//! Write `F[Q]` for the assertion.  With the positive obligation alone, `g → Q`
//! holds in every model of the rewrite, so `F[g] ≤ F[Q]` at a positive
//! occurrence (monotonicity) — no wrong `sat`; and a model of `F[Q]` extends by
//! setting `g := Q`, which satisfies `∀x. (g → φ)` — no wrong `unsat`.  With
//! the negative obligation alone the argument is the mirror image: `Q ≤ g` in
//! every model of `g ∨ ¬φ(sk)`, `F` is antitone at a negative occurrence, and a
//! model of `F[Q]` extends by interpreting `sk` as a falsifying witness when
//! one exists (so `φ(sk) ≡ Q` in that model) and arbitrarily otherwise.  With
//! both, `g ↔ Q` holds outright and the rewrite is an equivalence.
//!
//! # What it declines, and what that costs
//!
//! * A quantifier occurring anywhere under another binder (`Forall`, `Exists`,
//!   `Let`, `Match`) in the same assertion is left alone: it may mention the
//!   enclosing binder's variables, so a *closed* `g` cannot stand for it and a
//!   Skolem *constant* cannot witness it.  Such a quantifier keeps the MBQI
//!   path — the enclosing quantifier's instances are ground, and this pass runs
//!   on them too (see
//!   [`Solver::prepare_ground_instance`](crate::solver::Solver)), so the inner
//!   quantifier is guarded once it is closed.
//! * More than [`MAX_GUARDED_QUANTIFIERS`] candidates in one assertion.
//!
//! Declining never fabricates a verdict: a quantifier that reaches `encode`
//! with neither a registration nor a guard sets
//! [`Solver::quantifier_literal_unconstrained`](crate::solver::Solver), and any
//! `Sat` resting on it is reported as `Unknown`.
//!
//! Reference: Z3's `ast/normal_forms/nnf.cpp` (polarity-driven NNF +
//! Skolemisation) and `smt/smt_quantifier.cpp`'s guarded instantiation
//! (`m_context.mk_iff` between a quantifier's literal and its instances).

use oxiz_core::ast::traversal::get_children;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::smtlib::reserved_name;

#[allow(unused_imports)]
use crate::prelude::*;

use super::super::term_walk::asserted_children;

/// Maximum number of quantifier occurrences one assertion may have guarded.
///
/// Mirrors [`binder_row`](super::binder_row)'s and
/// [`finite_expand`](super::finite_expand)'s caps for the same reason: each
/// candidate costs a walk and up to two derived assertions, and an adversarial
/// term must not turn one assertion into an unbounded number of them.
/// Declining past the cap costs completeness only — see the module doc.
const MAX_GUARDED_QUANTIFIERS: usize = 64;

/// The polarity of a sub-term occurrence inside one assertion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pol {
    /// Every occurrence seen so far is positive (`φ` is only ever *used*).
    Pos,
    /// Every occurrence seen so far is negative (`φ` is only ever *negated*).
    Neg,
    /// Both, or a position whose polarity the Boolean skeleton cannot pin
    /// down (`xor`, a Boolean `=`, an `ite` condition, a `UF` argument).
    Both,
}

impl Pol {
    /// The polarity of the same sub-term one `not` deeper.
    fn flip(self) -> Self {
        match self {
            Pol::Pos => Pol::Neg,
            Pol::Neg => Pol::Pos,
            Pol::Both => Pol::Both,
        }
    }

    /// The polarity of a sub-term seen at `self` in one place and `other` in
    /// another: `Both` unless the two agree.
    fn join(self, other: Self) -> Self {
        if self == other { self } else { Pol::Both }
    }

    /// Does this occurrence need the *universal* obligation?
    fn has_pos(self) -> bool {
        matches!(self, Pol::Pos | Pol::Both)
    }

    /// Does this occurrence need the *existential* obligation?
    fn has_neg(self) -> bool {
        matches!(self, Pol::Neg | Pol::Both)
    }
}

/// One assertion's rewrite: the term to encode in place of it, and the derived
/// obligations that define the fresh Boolean constants it now mentions.
pub(crate) struct Guarded {
    /// The assertion with every guarded quantifier replaced by its constant.
    pub(crate) term: TermId,
    /// The obligations, to be asserted beside `term` (never pushed onto
    /// `Solver::assertions`: they are derived, and an unsat core must not
    /// blame a term the user never wrote).
    pub(crate) obligations: Vec<TermId>,
}

/// `term` with every quantifier it does **not** state unconditionally replaced
/// by a fresh Boolean constant, plus the obligations defining those constants
/// — or `None` when every quantifier in `term` is already on the asserted
/// spine (the overwhelmingly common case, and a cheap one: two walks and no
/// allocation beyond them).
///
/// `next_skolem_id` is the solver's monotone fresh-symbol counter, threaded
/// for the reason [`exists_skolem`](super::exists_skolem) documents: two
/// rewrites that both started from zero would make two unrelated obligations
/// share one witness symbol, which is a strengthening that can turn `sat` into
/// `unsat`.
pub(crate) fn guard_conditional_quantifiers(
    term: TermId,
    manager: &mut TermManager,
    next_skolem_id: &mut u64,
) -> Option<Guarded> {
    let (order, polarities, under_binder) = occurrence_polarities(term, manager)?;
    let spine = unconditional_positive_quantifiers(term, manager);

    let candidates: Vec<TermId> = order
        .into_iter()
        .filter(|q| !spine.contains(q) && !under_binder.contains(q))
        .collect();
    if candidates.is_empty() || candidates.len() > MAX_GUARDED_QUANTIFIERS {
        return None;
    }

    let mut replacements: FxHashMap<TermId, TermId> = FxHashMap::default();
    let mut obligations: Vec<TermId> = Vec::new();

    for quantifier in candidates {
        let Some(pol) = polarities.get(&quantifier).copied() else {
            continue;
        };
        let Some((vars, body, patterns, is_exists)) = binder_parts(quantifier, manager) else {
            continue;
        };
        let guard_name = reserved_name("qg", &next_skolem_id.to_string());
        *next_skolem_id = next_skolem_id.checked_add(1)?;
        let bool_sort = manager.sorts.bool_sort;
        let guard = manager.mk_var(&guard_name, bool_sort);

        // The universal half: `g -> Q`.  For a `forall` it is `∀x. (g → φ)`;
        // for an `exists` at negative polarity the dual `¬g → ¬∃y. ψ` is
        // `∀y. (g ∨ ¬ψ)`.  Both are plain unconditional universals, so every
        // existing instantiation path guards its own instances for free.
        let universal_needed = if is_exists {
            pol.has_neg()
        } else {
            pol.has_pos()
        };
        // The existential half: a witness.  For a `forall` at negative
        // polarity `¬g → ¬∀x. φ` is `g ∨ ¬φ(sk)`; for an `exists` at positive
        // polarity `g → ∃y. ψ` is `¬g ∨ ψ(sk)`.
        let existential_needed = if is_exists {
            pol.has_pos()
        } else {
            pol.has_neg()
        };

        let mut built: Vec<TermId> = Vec::new();
        if universal_needed {
            let guarded_body = if is_exists {
                let negated = manager.mk_not(body);
                manager.mk_or(vec![guard, negated])
            } else {
                manager.mk_implies(guard, body)
            };
            // An `exists` turned into a `forall` keeps no trigger: its
            // patterns were written for the existential's own instantiation
            // and a trigger that does not cover every bound variable is worse
            // than none.  A `forall` keeps its patterns — `(=> g φ)` contains
            // every sub-term `φ` did, so every trigger still matches.
            let keep_patterns = if is_exists {
                Vec::new()
            } else {
                patterns.clone()
            };
            built.push(rebuild_forall(&vars, guarded_body, keep_patterns, manager)?);
        }
        if existential_needed {
            let witness_body = skolemize_body(&vars, body, manager, next_skolem_id)?;
            let obligation = if is_exists {
                let not_guard = manager.mk_not(guard);
                manager.mk_or(vec![not_guard, witness_body])
            } else {
                let negated = manager.mk_not(witness_body);
                manager.mk_or(vec![guard, negated])
            };
            built.push(obligation);
        }
        if built.is_empty() {
            continue;
        }
        obligations.extend(built);
        replacements.insert(quantifier, guard);
    }

    if replacements.is_empty() {
        return None;
    }
    let rewritten = manager.substitute(term, &replacements);
    if rewritten == term {
        return None;
    }
    Some(Guarded {
        term: rewritten,
        obligations,
    })
}

/// The `(vars, body, patterns, is_exists)` of a binder term.
type BinderParts = (
    Vec<(String, oxiz_core::sort::SortId)>,
    TermId,
    Vec<Vec<TermId>>,
    bool,
);

fn binder_parts(quantifier: TermId, manager: &TermManager) -> Option<BinderParts> {
    let kind = manager.get(quantifier).map(|t| t.kind.clone())?;
    let (vars, body, patterns, is_exists) = match kind {
        TermKind::Forall {
            vars,
            body,
            patterns,
        } => (vars, body, patterns, false),
        TermKind::Exists {
            vars,
            body,
            patterns,
        } => (vars, body, patterns, true),
        _ => return None,
    };
    let names = vars
        .iter()
        .map(|&(name, sort)| (manager.resolve_str(name).to_string(), sort))
        .collect();
    let patterns = patterns
        .iter()
        .map(|pattern| pattern.iter().copied().collect())
        .collect();
    Some((names, body, patterns, is_exists))
}

/// `∀vars. body`, rebuilt from the names and sorts of an existing binder.
fn rebuild_forall(
    vars: &[(String, oxiz_core::sort::SortId)],
    body: TermId,
    patterns: Vec<Vec<TermId>>,
    manager: &mut TermManager,
) -> Option<TermId> {
    if vars.is_empty() {
        return None;
    }
    let bound = vars.iter().map(|(name, sort)| (name.as_str(), *sort));
    Some(manager.mk_forall_with_patterns(bound, body, patterns))
}

/// `body` with every bound variable of `vars` replaced by a fresh Skolem
/// constant, or `None` when the binder is degenerate.
///
/// The substitution is [`TermManager::substitute`], which is capture-avoiding
/// and drops a mapping that a nested binder shadows — so a body that re-binds
/// one of `vars`' names keeps its own meaning.  The constants are minted
/// through [`reserved_name`] from the solver-wide counter, so they can collide
/// with neither a user symbol nor another obligation's witness.
fn skolemize_body(
    vars: &[(String, oxiz_core::sort::SortId)],
    body: TermId,
    manager: &mut TermManager,
    next_skolem_id: &mut u64,
) -> Option<TermId> {
    if vars.is_empty() {
        return None;
    }
    let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
    for (name, sort) in vars {
        let bound = manager.mk_var(name, *sort);
        let witness_name = reserved_name("sk", &next_skolem_id.to_string());
        let next = next_skolem_id.checked_add(1)?;
        *next_skolem_id = next;
        let witness = manager.mk_var(&witness_name, *sort);
        map.insert(bound, witness);
    }
    Some(manager.substitute(body, &map))
}

/// The quantifiers of `term` that [`Solver::register_asserted_quantifiers`]
/// registers as unconditional facts, i.e. the ones this pass must leave alone.
///
/// Deliberately the *same* walk that registration uses
/// ([`asserted_children`]), so the two can never disagree about which
/// occurrences are already tied to their meaning: an occurrence this set
/// misses would be guarded twice, and one it invents would be guarded by
/// nobody.
fn unconditional_positive_quantifiers(term: TermId, manager: &TermManager) -> FxHashSet<TermId> {
    let mut registered: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<(TermId, bool)> = vec![(term, true)];
    let mut visited: FxHashSet<(TermId, bool)> = FxHashSet::default();

    while let Some((current, positive)) = stack.pop() {
        if !visited.insert((current, positive)) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        if positive && matches!(kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
            registered.insert(current);
        }
        stack.extend(asserted_children(&kind, positive));
    }
    registered
}

/// Every quantifier sub-term of `term` that is **not** inside another binder,
/// in first-visit order, with the polarity of its occurrences; plus the
/// quantifiers that *are* inside a binder, which this pass must decline.
///
/// Returns `None` when `term` carries no quantifier at all, which is the
/// answer for almost every assertion and costs one structural walk.
type Occurrences = (Vec<TermId>, FxHashMap<TermId, Pol>, FxHashSet<TermId>);

fn occurrence_polarities(term: TermId, manager: &TermManager) -> Option<Occurrences> {
    let mut order: Vec<TermId> = Vec::new();
    let mut polarities: FxHashMap<TermId, Pol> = FxHashMap::default();
    let mut under_binder: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<(TermId, Pol)> = vec![(term, Pol::Pos)];
    let mut visited: FxHashSet<(TermId, u8)> = FxHashSet::default();

    while let Some((current, pol)) = stack.pop() {
        let tag = match pol {
            Pol::Pos => 0u8,
            Pol::Neg => 1u8,
            Pol::Both => 2u8,
        };
        if !visited.insert((current, tag)) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        match &kind {
            TermKind::Forall { body, patterns, .. } | TermKind::Exists { body, patterns, .. } => {
                if !polarities.contains_key(&current) {
                    order.push(current);
                }
                polarities
                    .entry(current)
                    .and_modify(|seen| *seen = seen.join(pol))
                    .or_insert(pol);
                // Everything below a binder is declined: it may mention the
                // bound variables, and a closed guard constant cannot stand
                // for a term that does.
                collect_inner_quantifiers(*body, manager, &mut under_binder);
                for pattern in patterns.iter() {
                    for &pattern_term in pattern.iter() {
                        collect_inner_quantifiers(pattern_term, manager, &mut under_binder);
                    }
                }
            }
            TermKind::Let { .. } | TermKind::Match { .. } => {
                // Same reason as a quantifier body: `Let` and `Match` bind
                // names too.
                for child in get_children(&kind) {
                    collect_inner_quantifiers(child, manager, &mut under_binder);
                }
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args.iter() {
                    stack.push((arg, pol));
                }
            }
            TermKind::Not(inner) => stack.push((*inner, pol.flip())),
            TermKind::Implies(lhs, rhs) => {
                stack.push((*lhs, pol.flip()));
                stack.push((*rhs, pol));
            }
            TermKind::Ite(cond, then_branch, else_branch) => {
                // The condition is used at both polarities by the `ite`'s two
                // Tseitin clauses; the branches inherit the position's own.
                stack.push((*cond, Pol::Both));
                stack.push((*then_branch, pol));
                stack.push((*else_branch, pol));
            }
            _ => {
                // `Eq` (this AST has no `Iff`, so a Boolean `=` is an `Eq`),
                // `Xor`, `Distinct`, a `UF` argument, an array element: every
                // one of them is a polarity boundary, and `Both` is the
                // answer that is sound at all of them.
                for child in get_children(&kind) {
                    stack.push((child, Pol::Both));
                }
            }
        }
    }

    if order.is_empty() && under_binder.is_empty() {
        return None;
    }
    Some((order, polarities, under_binder))
}

/// Add every quantifier sub-term reachable from `term` to `out`.
fn collect_inner_quantifiers(term: TermId, manager: &TermManager, out: &mut FxHashSet<TermId>) {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        if matches!(kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
            out.insert(current);
        }
        stack.extend(get_children(&kind));
    }
}

#[cfg(test)]
mod tests;
