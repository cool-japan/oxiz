//! Sound MBQI SAT certification for the (almost-)uninterpreted / bounded
//! fragment, via complete instantiation (Ge & de Moura, CAV 2009).
//!
//! # Why this module exists
//!
//! The counterexample-guided core of MBQI ([`super::counterexample`]) can
//! *refute* a quantified goal by finding an instantiation that the candidate
//! model falsifies, but on its own it never *certifies* satisfiability: for a
//! bound variable ranging over an infinite domain (Int, Real, ...) a finite
//! sample of "no counterexample found" proves nothing.  This module supplies
//! the missing certification step.
//!
//! # The idea: complete instantiation + ground saturation
//!
//! For the fragments the SMT-LIB logics UFLIA / UFLRA / AUFLIA target, a
//! universal `∀x⃗. φ` has a *finite, complete* instantiation set: instantiating
//! every bound variable over that set yields a ground formula that is
//! equisatisfiable with the original.  This module computes that complete set
//! and hands it to the ordinary ground solver as lemmas.  Certification is then
//! a saturation test performed by the caller: once **every** complete instance
//! has been emitted and the ground solver *still* reports a model, that model
//! satisfies the whole formula and `sat` is sound.
//!
//! Crucially, the `sat` conclusion rests on the ground solver's own SAT result
//! over the real assertions — never on evaluating the body under the completed
//! model (which may be an incomplete or macro-completed approximation).  A
//! universal instance is always a sound logical consequence, so adding the
//! complete set can only make an unsatisfiable problem reveal its conflict; it
//! can never turn `unsat` into `sat`.
//!
//! # The complete instantiation set
//!
//! * **Bounded box** — body `guard ⇒ C` where `guard` pins every Int variable
//!   to a concrete finite interval `[l, u]`.  The whole interval is enumerated:
//!   outside it the guard is false (implication vacuously true), inside every
//!   value is instantiated.  Exhaustive, hence trivially complete.
//!
//! * **Essentially uninterpreted** — every occurrence of every bound variable
//!   is a direct argument of an uninterpreted function or array `select`.  Each
//!   variable is instantiated over the ground terms already appearing at that
//!   argument position (harvested from the completed model's interpretation
//!   graph).  This is exactly Ge & de Moura's relevant set.
//!
//! * **Almost uninterpreted (guarded)** — body `guard ⇒ C` where the consequent
//!   `C` reads every bound variable *only* through uninterpreted functions /
//!   array operations (strictly essentially uninterpreted), and `guard` is a
//!   conjunction of:
//!     - **variable-vs-ground** comparisons `x ⊕ t` (`⊕ ∈ {≤,<,≥,>,=}`, `t`
//!       ground), including the disequality `¬(x = t)`.  The ground bound `t` is
//!       folded into the relevant instantiation set (see
//!       [`augment_guard_grounds`]) so no region boundary is missed;
//!     - **variable-vs-variable** comparisons `x ⊕ y` for the monotone-preserving
//!       relations `≤`, `≥`, `=` only.
//!       Strict `<` / `>` *between two variables*, and any bare variable in the
//!       consequent, are excluded because they are not preserved by the
//!       model-extension projection.  Each variable is instantiated over its sort's
//!       relevant set (UF-argument terms ∪ guard-ground constants); outside the guard
//!       region the implication is vacuously true.
//!
//! When *any* tracked quantifier falls outside these fragments (or is an
//! existential, which needs a witness rather than an instance) the module
//! reports [`CertifyResult::NotEligible`] and the caller keeps its normal
//! behaviour, ultimately answering `unknown`.  The module never fabricates
//! `sat`.

use crate::prelude::*;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use oxiz_core::ast::traversal::collect_free_vars_including_patterns;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::sort::SortId;

use super::model_completion::CompletedModel;
use super::{Instantiation, InstantiationReason, QuantifiedFormula};

/// Result of collecting the complete instantiation set for every tracked
/// quantifier.
#[derive(Debug)]
pub(crate) enum CertifyResult {
    /// The complete set of relevant/bounded instantiation lemmas across every
    /// eligible quantifier.  The caller deduplicates against instances already
    /// emitted in earlier rounds: if nothing is fresh the set is *saturated*
    /// and the ground solver's current model is a sound `sat` witness;
    /// otherwise the fresh lemmas drive the next round.
    Instances(Vec<Instantiation>),
    /// At least one quantifier is outside the certifiable fragment (or is an
    /// existential).  The caller must fall back to its normal handling.
    NotEligible,
}

/// Collect the complete instantiation set for every tracked quantifier, or
/// report that the goal is outside the certifiable fragment.
pub(crate) fn collect_fragment_instances(
    quantifiers: &[QuantifiedFormula],
    model: &CompletedModel,
    manager: &mut TermManager,
    cap: usize,
    generation: u32,
) -> CertifyResult {
    // A relevant instantiation term must be *ground* with respect to every
    // tracked quantifier: a term that mentions some quantifier's own bound
    // variable is not a legal instantiation value.  Such a term can leak into a
    // function's model interpretation as a *symbolic* application (e.g. `f(x)`
    // harvested from the un-instantiated body `f(x) ≥ 0`), and instantiating a
    // variable with itself would leave a free variable in the lemma — which
    // `substitute_tuple` rightly rejects, collapsing the whole certification to
    // `NotEligible`.  Excluding these terms up front keeps the relevant set
    // ground and lets the fragment saturate.
    let bound_names: FxHashSet<Spur> = quantifiers
        .iter()
        .flat_map(|q| q.bound_vars.iter().map(|(n, _)| *n))
        .collect();
    let mut relevant = collect_relevant_terms(model, manager, &bound_names);

    // Augment the relevant set with the ground terms that a bound variable is
    // compared against in a guard (`x ≤ c`, `i < n`, `i ≠ k`, ...).  This is the
    // second half of the almost-uninterpreted fragment's relevant set: without
    // the guard constants a region boundary can be missed, which would let a
    // point the guard admits — but that is never instantiated — escape the check
    // and turn a genuine `unsat` into a spurious `sat`.  Adding instantiation
    // points is always sound (a universal instance is a consequence), so this
    // can only ever *strengthen* the ground problem.
    for quantifier in quantifiers {
        if quantifier.is_universal && quantifier.can_instantiate() {
            augment_guard_grounds(quantifier, &mut relevant, manager);
        }
    }

    let mut saw_quantifier = false;
    let mut instances: Vec<Instantiation> = Vec::new();

    for quantifier in quantifiers {
        if !quantifier.can_instantiate() {
            continue;
        }
        saw_quantifier = true;

        // Only universal quantifiers admit a complete instantiation set.
        // A bare existential needs a witness, not an instance, so decline.
        if !quantifier.is_universal {
            return CertifyResult::NotEligible;
        }

        match universal_instances(quantifier, model, &relevant, manager, cap, generation) {
            Some(mut insts) => instances.append(&mut insts),
            None => return CertifyResult::NotEligible,
        }
    }

    if !saw_quantifier || instances.is_empty() {
        return CertifyResult::NotEligible;
    }
    CertifyResult::Instances(instances)
}

/// Build the complete instantiation set for one universal quantifier, or `None`
/// when it is outside the certifiable fragment.
fn universal_instances(
    quantifier: &QuantifiedFormula,
    model: &CompletedModel,
    relevant: &FxHashMap<SortId, Vec<TermId>>,
    manager: &mut TermManager,
    cap: usize,
    generation: u32,
) -> Option<Vec<Instantiation>> {
    // Prefer the exhaustive bounded-box domain (needs no model-extension
    // argument); otherwise fall back to the essentially-/almost-uninterpreted
    // relevant-term domain.
    let domains = match int_box_domains(quantifier, model, manager, cap) {
        Some(d) => d,
        None => eu_domains(quantifier, relevant, manager, cap)?,
    };

    let tuples = cartesian_product(&domains, cap)?;
    if tuples.is_empty() {
        // No relevant instantiation exists at all; we cannot certify.
        return None;
    }

    let mut instances = Vec::with_capacity(tuples.len());
    for tuple in &tuples {
        // A substitution that leaves a bound variable free is an internal error;
        // decline rather than emit a lemma with a stray variable.
        let ground = substitute_tuple(quantifier, tuple, manager)?;
        instances.push(make_instantiation(quantifier, tuple, ground, generation));
    }
    Some(instances)
}

/// Collect, per sort, the ground terms that appear as an argument of some
/// uninterpreted function or array `select` in the completed model.
///
/// These are exactly the "relevant terms" of Ge & de Moura: instantiating a
/// bound variable over them is sufficient for the essentially-uninterpreted
/// fragment.  Function-argument terms are read from the model's finite
/// interpretation graph (each entry's `args`, sorted by the function's declared
/// domain), and array indices from `select` keys in the assignment table.
fn collect_relevant_terms(
    model: &CompletedModel,
    manager: &TermManager,
    bound_names: &FxHashSet<Spur>,
) -> FxHashMap<SortId, Vec<TermId>> {
    let mut per_sort: FxHashMap<SortId, Vec<TermId>> = FxHashMap::default();
    let mut seen: FxHashSet<(SortId, TermId)> = FxHashSet::default();

    // Reject any candidate that mentions a quantifier's bound variable: it is a
    // symbolic (non-ground) application harvested from an un-instantiated body,
    // not a legal instantiation value.
    let mentions_bound = |term: TermId| -> bool {
        if bound_names.is_empty() {
            return false;
        }
        collect_free_vars_including_patterns(term, manager).iter().any(|&v| {
            matches!(manager.get(v).map(|t| &t.kind), Some(TermKind::Var(n)) if bound_names.contains(n))
        })
    };

    let mut push = |sort: SortId, term: TermId, per_sort: &mut FxHashMap<SortId, Vec<TermId>>| {
        if mentions_bound(term) {
            return;
        }
        if seen.insert((sort, term)) {
            per_sort.entry(sort).or_default().push(term);
        }
    };

    // Uninterpreted-function argument terms (grouped by the domain sort at each
    // argument position).
    for interp in model.function_interps.values() {
        for entry in &interp.entries {
            for (i, &arg) in entry.args.iter().enumerate() {
                let sort = interp
                    .domain
                    .get(i)
                    .copied()
                    .or_else(|| manager.get(arg).map(|t| t.sort));
                if let Some(sort) = sort {
                    push(sort, arg, &mut per_sort);
                }
            }
        }
    }

    // Array-select index terms.
    for &key in model.assignments.keys() {
        if let Some(t) = manager.get(key) {
            if let TermKind::Select(_, index) = &t.kind {
                let idx = *index;
                if let Some(it) = manager.get(idx) {
                    let sort = it.sort;
                    push(sort, idx, &mut per_sort);
                }
            }
        }
    }

    per_sort
}

/// Strip a chain of leading `Implies` premises that mention none of `vars`.
///
/// # Why the fragment analysis needs this
///
/// [`encode::quant_guard`](crate::solver::Solver) ties a conditionally placed
/// quantifier to a fresh Boolean constant `g` by rewriting `∀x. φ` into the
/// plain universal `∀x. (g → φ)`.  `g` is *closed* — it mentions no bound
/// variable and restricts no domain — so it is not a guard in the
/// almost-uninterpreted sense at all; it is bookkeeping wrapped around a body
/// that may well be in the fragment.  Every analysis below that reads "the
/// premise of a top-level `Implies`" would otherwise read `g` and stop there:
///
/// * [`augment_guard_grounds`] would miss the real guard's ground constants,
///   so a body `(=> (= i c) atom)` would contribute no `c` to the relevant
///   set and could not be certified;
/// * [`extract_int_bounds`] would find no `And` of comparisons and give up the
///   exhaustive integer box;
/// * [`is_eu_eligible`] would hand the inner `(= i c)` to [`strict_eu`] rather
///   than to [`premise_safe`], and a bound variable under `=` fails `strict_eu`
///   by design.
///
/// Peeling is sound for all three because it is an *equivalence-preserving*
/// re-reading of the same body: `g → ψ` and `ψ` agree wherever `g` holds, and
/// where `g` fails the body is true for **every** point of the domain at once,
/// so no instantiation point the analyses choose can ever be wrong about it.
/// The peeled body is used only to choose instantiation points and bounds;
/// every lemma emitted is still built from the *original* body
/// ([`substitute_tuple`] takes `quantifier.body`), so a guard that must be
/// carried is carried.
fn peel_ground_premises(body: TermId, vars: &FxHashSet<Spur>, manager: &TermManager) -> TermId {
    let mut current = body;
    // A hash-consed body is a DAG of bounded depth here, but the loop is
    // bounded anyway: each round strips one `Implies` node.
    for _ in 0..MAX_PEELED_PREMISES {
        let Some(TermKind::Implies(premise, consequent)) =
            manager.get(current).map(|t| t.kind.clone())
        else {
            return current;
        };
        if mentions_bound_var(premise, vars, manager) {
            return current;
        }
        current = consequent;
    }
    current
}

/// Bound on [`peel_ground_premises`]' loop.  `quant_guard` adds one premise per
/// polarity boundary it crosses and its own cap is far below this.
const MAX_PEELED_PREMISES: usize = 16;

/// Add, to the per-sort relevant set, every ground term that one of the
/// quantifier's bound variables is compared against in its guard.
///
/// Only the premise of a top-level `guard ⇒ _` body is inspected (that is where
/// the almost-uninterpreted bounds live), plus a non-implication body treated as
/// its own guard.  A comparison `x ⊕ t` / `t ⊕ x` (including a disequality
/// `¬(x = t)`) with `x` a bound variable and `t` ground contributes `t` to the
/// relevant set for `t`'s sort.
fn augment_guard_grounds(
    quantifier: &QuantifiedFormula,
    relevant: &mut FxHashMap<SortId, Vec<TermId>>,
    manager: &TermManager,
) {
    let var_names: FxHashSet<Spur> = quantifier.bound_vars.iter().map(|(n, _)| *n).collect();
    if var_names.is_empty() {
        return;
    }

    let body = peel_ground_premises(quantifier.body, &var_names, manager);
    let guard = match manager.get(body).map(|t| t.kind.clone()) {
        Some(TermKind::Implies(g, _)) => g,
        Some(_) => body,
        None => return,
    };

    let mut ground_terms: Vec<TermId> = Vec::new();
    collect_guard_ground_terms(guard, &var_names, manager, &mut ground_terms);

    for term in ground_terms {
        if let Some(sort) = manager.get(term).map(|t| t.sort) {
            let bucket = relevant.entry(sort).or_default();
            if !bucket.contains(&term) {
                bucket.push(term);
            }
        }
    }
}

/// Walk a guard conjunction (through `And` / `Or` / `Not`) and collect the
/// ground side of every `bound-variable ⊕ ground` comparison.
///
/// Iterative with an explicit heap stack: the guard shape is
/// caller-controlled input and the results flow through `out` with no error
/// channel, so a depth cap could only have silently dropped guard bounds —
/// shrinking the relevant instantiation set, which is exactly the failure
/// [`augment_guard_grounds`] exists to prevent.  Children are pushed in
/// reverse so `out` keeps the recursive version's left-to-right emission
/// order.  `visited` bounds shared-subterm re-expansion to linear; the
/// consumer deduplicates terms per sort bucket, so collapsing repeat visits
/// into one emission is behavior-preserving.
fn collect_guard_ground_terms(
    guard: TermId,
    vars: &FxHashSet<Spur>,
    manager: &TermManager,
    out: &mut Vec<TermId>,
) {
    let mut stack: Vec<TermId> = vec![guard];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(node) = manager.get(current) else {
            continue;
        };
        match &node.kind {
            TermKind::And(args) | TermKind::Or(args) => {
                for &a in args.iter().rev() {
                    stack.push(a);
                }
            }
            TermKind::Not(a) => stack.push(*a),
            TermKind::Le(l, r)
            | TermKind::Ge(l, r)
            | TermKind::Lt(l, r)
            | TermKind::Gt(l, r)
            | TermKind::Eq(l, r) => {
                push_guard_ground(*l, *r, vars, manager, out);
            }
            _ => {}
        }
    }
}

/// If exactly one side of a comparison is a bound variable and the other side is
/// ground (mentions no bound variable), record that ground side.
fn push_guard_ground(
    l: TermId,
    r: TermId,
    vars: &FxHashSet<Spur>,
    manager: &TermManager,
    out: &mut Vec<TermId>,
) {
    let lb = is_bound_var(l, vars, manager);
    let rb = is_bound_var(r, vars, manager);
    if lb && !rb && !mentions_bound_var(r, vars, manager) {
        out.push(r);
    } else if rb && !lb && !mentions_bound_var(l, vars, manager) {
        out.push(l);
    }
}

/// Build per-variable Int enumeration domains from concrete interval guards.
///
/// Succeeds only when the body is `guard ⇒ _` and every bound variable is of
/// sort Int with both a finite lower and upper bound extracted from `guard`
/// (the non-variable side is evaluated under the model, so `i < n` with a
/// declared `n = 5` yields `i ∈ [0, 4]`).  The product of interval sizes must
/// not exceed `cap`.
fn int_box_domains(
    quantifier: &QuantifiedFormula,
    model: &CompletedModel,
    manager: &mut TermManager,
    cap: usize,
) -> Option<Vec<Vec<TermId>>> {
    let bounds = extract_int_bounds(quantifier.body, &quantifier.bound_vars, model, manager)?;

    let mut domains = Vec::with_capacity(quantifier.bound_vars.len());
    let mut product: usize = 1;
    for &(name, sort) in quantifier.bound_vars.iter() {
        if sort != manager.sorts.int_sort {
            return None;
        }
        let (lo, hi) = bounds.get(&name)?;
        if hi < lo {
            return None;
        }
        let count = ((hi - lo) + 1u32).to_usize()?;
        product = product.checked_mul(count)?;
        if product > cap {
            return None;
        }
        let dom: Vec<TermId> = num_iter_inclusive(lo, hi)
            .map(|v| manager.mk_int(v))
            .collect();
        domains.push(dom);
    }
    Some(domains)
}

/// Inclusive `BigInt` range iterator `lo..=hi`.
fn num_iter_inclusive(lo: &BigInt, hi: &BigInt) -> impl Iterator<Item = BigInt> {
    let mut cur = lo.clone();
    let hi = hi.clone();
    core::iter::from_fn(move || {
        if cur > hi {
            None
        } else {
            let out = cur.clone();
            cur += 1;
            Some(out)
        }
    })
}

/// Build per-variable relevant-term domains for the essentially- /
/// almost-uninterpreted fragment.
///
/// Returns `None` when the quantifier is outside the fragment, when a bound
/// variable has no relevant term of its sort, or when the product of domain
/// sizes exceeds `cap`.
fn eu_domains(
    quantifier: &QuantifiedFormula,
    relevant: &FxHashMap<SortId, Vec<TermId>>,
    manager: &TermManager,
    cap: usize,
) -> Option<Vec<Vec<TermId>>> {
    let var_names: FxHashSet<Spur> = quantifier.bound_vars.iter().map(|(n, _)| *n).collect();
    let body = peel_ground_premises(quantifier.body, &var_names, manager);
    if !is_eu_eligible(body, &var_names, manager) {
        return None;
    }

    let mut domains = Vec::with_capacity(quantifier.bound_vars.len());
    let mut product: usize = 1;
    for &(_name, sort) in quantifier.bound_vars.iter() {
        let dom = relevant.get(&sort)?;
        if dom.is_empty() {
            return None;
        }
        product = product.checked_mul(dom.len())?;
        if product > cap {
            return None;
        }
        domains.push(dom.clone());
    }
    Some(domains)
}

/// Whether `body` is essentially- or almost-uninterpreted with respect to
/// `vars` (see the module docs).
fn is_eu_eligible(body: TermId, vars: &FxHashSet<Spur>, manager: &TermManager) -> bool {
    match manager.get(body).map(|t| t.kind.clone()) {
        Some(TermKind::Implies(premise, consequent)) => {
            premise_safe(premise, vars, manager) && strict_eu(consequent, vars, manager)
        }
        Some(_) => strict_eu(body, vars, manager),
        None => false,
    }
}

/// Strictly essentially uninterpreted: no bound variable occurs anywhere except
/// as a direct argument of an uninterpreted function or array operation.
fn strict_eu(term: TermId, vars: &FxHashSet<Spur>, manager: &TermManager) -> bool {
    eu_walk(term, vars, manager, false)
}

/// Premise position: like [`strict_eu`], but a comparison between two bound
/// variables using a monotone-preserving relation (`≤`, `≥`, `=`) is allowed
/// as a guard.
fn premise_safe(term: TermId, vars: &FxHashSet<Spur>, manager: &TermManager) -> bool {
    eu_walk(term, vars, manager, true)
}

/// Shared traversal for [`strict_eu`] / [`premise_safe`].
///
/// `allow_guard` enables the almost-uninterpreted var-vs-var guard exception.
///
/// # Iterative machine
///
/// This used to be a three-way native recursion (`eu_walk` ↔ `guard_cmp` ↔
/// `arg_ok`) returning `bool` — a return type with no error channel, where a
/// depth cap could only have fabricated a wrong fragment-classification
/// verdict, and a wrong `true` here would let an incomplete instantiation
/// set certify a spurious `sat`.  It now runs as a worklist of obligations
/// on an explicit heap stack, so nesting depth in the quantifier body costs
/// heap, never native stack.
///
/// Every check is a pure predicate over the term DAG and the whole
/// classification is one conjunction, so obligation order cannot change the
/// outcome: the walk answers `false` the moment any obligation fails, and
/// `true` only once the worklist drains.  `seen` deduplicates structural
/// obligations by `TermId` (`allow_guard` is fixed for a given walk), which
/// preserves the verdict — a repeated subterm re-adds identical conjuncts —
/// while bounding re-expansion of shared subterms of the hash-consed DAG to
/// linear.
fn eu_walk(term: TermId, vars: &FxHashSet<Spur>, manager: &TermManager, allow_guard: bool) -> bool {
    /// One pending conjunct of the classification.
    enum Obligation {
        /// The structural essentially-uninterpreted check for a term.
        Walk(TermId),
        /// A direct argument of an uninterpreted function / array operation:
        /// either a bound variable (exactly the allowed leaf position) or a
        /// subterm that must itself be essentially uninterpreted.  (The old
        /// `arg_ok`.)
        Arg(TermId),
        /// A comparison `l ⊕ r` (the old `guard_cmp`).  In a guard position
        /// (`allow_guard`) two extra forms are admitted beyond a plain
        /// essentially-uninterpreted comparison:
        ///
        /// * **bound-variable vs ground** — `x ⊕ t` (or `t ⊕ x`) where `t`
        ///   mentions no bound variable.  This is the interval/point bound
        ///   of the almost-uninterpreted fragment; its ground constant `t`
        ///   is added to the relevant instantiation set (see
        ///   [`augment_guard_grounds`]) so the region boundary is always
        ///   covered.
        /// * **bound-variable vs bound-variable** — `x ⊕ y`, but only for
        ///   the monotone-preserving non-strict relations (the `bool`
        ///   field, the old `allow_var_var`); strict `<` / `>` between two
        ///   variables is *not* preserved by the projection and is
        ///   rejected.
        ///
        /// Outside a guard (the consequent), neither exception applies and
        /// each side must itself be essentially uninterpreted, so a bare
        /// bound variable is rejected.
        Cmp(TermId, TermId, bool),
    }

    let mut stack: Vec<Obligation> = vec![Obligation::Walk(term)];
    let mut seen: FxHashSet<TermId> = FxHashSet::default();

    while let Some(obligation) = stack.pop() {
        match obligation {
            Obligation::Walk(t) => {
                if !seen.insert(t) {
                    continue;
                }
                let Some(node) = manager.get(t) else {
                    return false;
                };
                match &node.kind {
                    // A bare bound variable in a non-argument position is
                    // disallowed.  A free/declared constant (not in `vars`)
                    // is fine.
                    TermKind::Var(name) => {
                        if vars.contains(name) {
                            return false;
                        }
                    }

                    // Constants.
                    TermKind::True
                    | TermKind::False
                    | TermKind::IntConst(_)
                    | TermKind::RealConst(_)
                    | TermKind::BitVecConst { .. }
                    | TermKind::StringLit(_) => {}

                    // Direct arguments of uninterpreted functions / arrays: a
                    // bound variable here is exactly the allowed position.
                    TermKind::Apply { args, .. } => {
                        for &a in args.iter() {
                            stack.push(Obligation::Arg(a));
                        }
                    }
                    TermKind::Select(arr, idx) => {
                        stack.push(Obligation::Arg(*arr));
                        stack.push(Obligation::Arg(*idx));
                    }
                    TermKind::Store(arr, idx, val) => {
                        stack.push(Obligation::Arg(*arr));
                        stack.push(Obligation::Arg(*idx));
                        stack.push(Obligation::Arg(*val));
                    }

                    // Comparisons.  Non-strict relations (`≤`, `≥`, `=`)
                    // additionally permit a monotone-preserving var-vs-var
                    // guard; strict relations (`<`, `>`) do not (they are
                    // not preserved by the model-extension projection).
                    // Both permit a bound-variable-vs-ground guard.
                    TermKind::Le(l, r) | TermKind::Ge(l, r) | TermKind::Eq(l, r) => {
                        stack.push(Obligation::Cmp(*l, *r, true));
                    }
                    TermKind::Lt(l, r) | TermKind::Gt(l, r) => {
                        stack.push(Obligation::Cmp(*l, *r, false));
                    }

                    // Structural boolean / arithmetic: descend.  Any bare
                    // bound variable reached this way falls into the `Var`
                    // arm above and is rejected.
                    TermKind::Not(a) | TermKind::Neg(a) => stack.push(Obligation::Walk(*a)),
                    TermKind::And(args)
                    | TermKind::Or(args)
                    | TermKind::Add(args)
                    | TermKind::Mul(args) => {
                        for &a in args.iter() {
                            stack.push(Obligation::Walk(a));
                        }
                    }
                    TermKind::Implies(l, r)
                    | TermKind::Sub(l, r)
                    | TermKind::Div(l, r)
                    | TermKind::Mod(l, r) => {
                        stack.push(Obligation::Walk(*l));
                        stack.push(Obligation::Walk(*r));
                    }
                    TermKind::Ite(c, th, el) => {
                        stack.push(Obligation::Walk(*c));
                        stack.push(Obligation::Walk(*th));
                        stack.push(Obligation::Walk(*el));
                    }
                    TermKind::Distinct(args) => {
                        // Disequality is not projection-preserving; reject if
                        // any operand is a bound variable, otherwise allow
                        // (operands read via UF).
                        for &a in args.iter() {
                            if is_bound_var(a, vars, manager) {
                                return false;
                            }
                            stack.push(Obligation::Walk(a));
                        }
                    }

                    // Anything else (strings, floating point, datatypes,
                    // nested quantifiers, ...): conservatively require it to
                    // contain no bound variable at all.
                    _ => {
                        if mentions_bound_var(t, vars, manager) {
                            return false;
                        }
                    }
                }
            }
            Obligation::Arg(t) => {
                if is_bound_var(t, vars, manager) {
                    continue;
                }
                stack.push(Obligation::Walk(t));
            }
            Obligation::Cmp(l, r, allow_var_var) => {
                if allow_guard {
                    let lb = is_bound_var(l, vars, manager);
                    let rb = is_bound_var(r, vars, manager);
                    // Bound variable compared with a ground term.
                    if lb && !mentions_bound_var(r, vars, manager) {
                        continue;
                    }
                    if rb && !mentions_bound_var(l, vars, manager) {
                        continue;
                    }
                    // Monotone-preserving comparison between two bound
                    // variables.
                    if allow_var_var && lb && rb {
                        continue;
                    }
                }
                stack.push(Obligation::Walk(l));
                stack.push(Obligation::Walk(r));
            }
        }
    }
    true
}

/// Whether `term` is a `Var` naming one of the bound variables.
fn is_bound_var(term: TermId, vars: &FxHashSet<Spur>, manager: &TermManager) -> bool {
    matches!(manager.get(term).map(|t| &t.kind), Some(TermKind::Var(n)) if vars.contains(n))
}

/// Whether any bound variable in `vars` occurs anywhere in `term`.
///
/// Pattern-aware on purpose: every caller uses a `false` answer to license
/// treating `term` as bound-variable-free, so an occurrence hidden in a
/// quantifier trigger must still count.
fn mentions_bound_var(term: TermId, vars: &FxHashSet<Spur>, manager: &TermManager) -> bool {
    let free = collect_free_vars_including_patterns(term, manager);
    free.iter().any(|&v| is_bound_var(v, vars, manager))
}

/// Extract concrete Int `[lower, upper]` bounds for each bound variable from a
/// `guard ⇒ _` body.
///
/// Only the premise of a top-level `Implies` is inspected (so that outside the
/// interval the implication is vacuously true).  The non-variable side of each
/// comparison is evaluated under `model`, which lets bounds refer to declared
/// constants (`i < n` with `n = 5`).  Strict bounds are tightened to the
/// integer interval (`i < 5` ⇒ upper `4`, `i > 0` ⇒ lower `1`).
fn extract_int_bounds(
    body: TermId,
    bound_vars: &[(Spur, SortId)],
    model: &CompletedModel,
    manager: &TermManager,
) -> Option<FxHashMap<Spur, (BigInt, BigInt)>> {
    let var_names: FxHashSet<Spur> = bound_vars.iter().map(|(n, _)| *n).collect();
    let body = peel_ground_premises(body, &var_names, manager);
    let guard = match manager.get(body).map(|t| t.kind.clone())? {
        TermKind::Implies(g, _) => g,
        _ => return None,
    };

    let conjuncts: Vec<TermId> = match manager.get(guard).map(|t| t.kind.clone())? {
        TermKind::And(args) => args.to_vec(),
        _ => vec![guard],
    };

    let mut lowers: FxHashMap<Spur, BigInt> = FxHashMap::default();
    let mut uppers: FxHashMap<Spur, BigInt> = FxHashMap::default();

    for atom in conjuncts {
        let Some(kind) = manager.get(atom).map(|t| t.kind.clone()) else {
            continue;
        };
        let (l, r, rel) = match &kind {
            TermKind::Ge(l, r) => (*l, *r, Rel::Ge),
            TermKind::Gt(l, r) => (*l, *r, Rel::Gt),
            TermKind::Le(l, r) => (*l, *r, Rel::Le),
            TermKind::Lt(l, r) => (*l, *r, Rel::Lt),
            _ => continue,
        };

        // Normalise to `variable rel constant`; if the variable is on the right
        // the relation is reversed.
        let (var, konst, rel) = if let Some(name) = bound_var_name(l, &var_names, manager) {
            match model_int(r, model, manager) {
                Some(c) => (name, c, rel),
                None => continue,
            }
        } else if let Some(name) = bound_var_name(r, &var_names, manager) {
            match model_int(l, model, manager) {
                Some(c) => (name, c, rel.flip()),
                None => continue,
            }
        } else {
            continue;
        };

        match rel {
            Rel::Ge => tighten_lower(&mut lowers, var, konst),
            Rel::Gt => tighten_lower(&mut lowers, var, konst + 1),
            Rel::Le => tighten_upper(&mut uppers, var, konst),
            Rel::Lt => tighten_upper(&mut uppers, var, konst - 1),
        }
    }

    let mut bounds: FxHashMap<Spur, (BigInt, BigInt)> = FxHashMap::default();
    for (name, lo) in lowers {
        if let Some(hi) = uppers.get(&name) {
            if hi >= &lo {
                bounds.insert(name, (lo, hi.clone()));
            }
        }
    }
    if bounds.is_empty() {
        None
    } else {
        Some(bounds)
    }
}

/// A comparison relation, oriented as `variable rel constant`.
#[derive(Clone, Copy)]
enum Rel {
    Ge,
    Gt,
    Le,
    Lt,
}

impl Rel {
    /// Reverse the relation (used when the variable is on the right-hand side).
    fn flip(self) -> Self {
        match self {
            Rel::Ge => Rel::Le,
            Rel::Gt => Rel::Lt,
            Rel::Le => Rel::Ge,
            Rel::Lt => Rel::Gt,
        }
    }
}

fn tighten_lower(lowers: &mut FxHashMap<Spur, BigInt>, var: Spur, v: BigInt) {
    lowers
        .entry(var)
        .and_modify(|e| {
            if v > *e {
                *e = v.clone();
            }
        })
        .or_insert(v);
}

fn tighten_upper(uppers: &mut FxHashMap<Spur, BigInt>, var: Spur, v: BigInt) {
    uppers
        .entry(var)
        .and_modify(|e| {
            if v < *e {
                *e = v.clone();
            }
        })
        .or_insert(v);
}

/// If `term` is a bound variable, return its name.
fn bound_var_name(term: TermId, vars: &FxHashSet<Spur>, manager: &TermManager) -> Option<Spur> {
    match manager.get(term).map(|t| t.kind.clone()) {
        Some(TermKind::Var(n)) if vars.contains(&n) => Some(n),
        _ => None,
    }
}

/// Evaluate `term` to a concrete `BigInt` under the model.
fn model_int(term: TermId, model: &CompletedModel, manager: &TermManager) -> Option<BigInt> {
    let resolved = model.eval(term).unwrap_or(term);
    match manager.get(resolved).map(|t| t.kind.clone())? {
        TermKind::IntConst(n) => Some(n),
        _ => None,
    }
}

/// Enumerate the full cartesian product of `domains`, or `None` if it would
/// exceed `cap` entries.
fn cartesian_product(domains: &[Vec<TermId>], cap: usize) -> Option<Vec<Vec<TermId>>> {
    let mut total: usize = 1;
    for d in domains {
        total = total.checked_mul(d.len())?;
        if total > cap {
            return None;
        }
    }
    let mut out: Vec<Vec<TermId>> = vec![Vec::new()];
    for d in domains {
        let mut next = Vec::with_capacity(out.len() * d.len());
        for prefix in &out {
            for &v in d {
                let mut row = prefix.clone();
                row.push(v);
                next.push(row);
            }
        }
        out = next;
    }
    Some(out)
}

/// Substitute a tuple of values for the quantifier's bound variables and return
/// the fully-grounded body, or `None` if a bound variable survives (an internal
/// error we must not turn into a lemma).
fn substitute_tuple(
    quantifier: &QuantifiedFormula,
    tuple: &[TermId],
    manager: &mut TermManager,
) -> Option<TermId> {
    let mut term_subst: FxHashMap<TermId, TermId> = FxHashMap::default();
    for (i, &(name, sort)) in quantifier.bound_vars.iter().enumerate() {
        let value = *tuple.get(i)?;
        let name_str = manager.resolve_str(name).to_string();
        let var_id = manager.mk_var(&name_str, sort);
        term_subst.insert(var_id, value);
    }
    if term_subst.is_empty() {
        return Some(quantifier.body);
    }
    let result = manager.substitute(quantifier.body, &term_subst);
    // Pattern-aware: an eliminated bound variable surviving only inside a
    // trigger still leaves the lemma un-grounded, and the non-pattern-aware
    // query cannot see it.
    let free = collect_free_vars_including_patterns(result, manager);
    if term_subst.keys().any(|k| free.contains(k)) {
        return None;
    }
    Some(result)
}

/// Build an [`Instantiation`] lemma for a relevant/bounded tuple.
fn make_instantiation(
    quantifier: &QuantifiedFormula,
    tuple: &[TermId],
    ground: TermId,
    generation: u32,
) -> Instantiation {
    let mut subst: FxHashMap<Spur, TermId> = FxHashMap::default();
    for (i, &(name, _sort)) in quantifier.bound_vars.iter().enumerate() {
        if let Some(&value) = tuple.get(i) {
            subst.insert(name, value);
        }
    }
    Instantiation::with_reason(
        quantifier.term,
        subst,
        ground,
        generation,
        InstantiationReason::ModelBased,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interned name of a `Var` term.
    fn var_spur(m: &TermManager, v: TermId) -> Spur {
        match m.get(v).map(|t| &t.kind) {
            Some(TermKind::Var(n)) => *n,
            other => panic!("expected a variable, got {other:?}"),
        }
    }

    // ===== eu_walk (iterative fragment classification) =====
    //
    // `eu_walk` / `guard_cmp` / `arg_ok` used to be a three-way native
    // recursion returning `bool`; it now runs as an obligation worklist on
    // an explicit heap stack. These tests pin the classification verdicts
    // (behavior preservation), deep-input survival on a small thread stack,
    // and the `seen`-set bound on shared DAGs.

    #[test]
    fn eu_walk_semantic_pins() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let y = m.mk_var("y", int_sort);
        let c = m.mk_var("c", int_sort); // a declared constant, not bound
        let zero = m.mk_int(0);
        let five = m.mk_int(5);
        let vars: FxHashSet<Spur> = [var_spur(&m, x), var_spur(&m, y)].into_iter().collect();

        // x only as a UF argument: essentially uninterpreted.
        let f_x = m.mk_apply("f", [x], int_sort);
        let fx_ge = m.mk_ge(f_x, zero);
        assert!(strict_eu(fx_ge, &vars, &m));

        // A bare bound variable in a comparison is rejected outside a
        // guard, admitted inside one (var vs ground).
        let x_ge = m.mk_ge(x, zero);
        assert!(!strict_eu(x_ge, &vars, &m));
        assert!(premise_safe(x_ge, &vars, &m));

        // Guard against a ground declared constant.
        let x_le_c = m.mk_le(x, c);
        assert!(premise_safe(x_le_c, &vars, &m));
        assert!(!strict_eu(x_le_c, &vars, &m));

        // Var-vs-var guards: non-strict allowed, strict rejected.
        let x_le_y = m.mk_le(x, y);
        assert!(premise_safe(x_le_y, &vars, &m));
        let x_lt_y = m.mk_lt(x, y);
        assert!(!premise_safe(x_lt_y, &vars, &m));

        // Distinct rejects a bound operand, allows UF-read operands.
        let d_bound = m.mk_distinct([x, five]);
        assert!(!strict_eu(d_bound, &vars, &m));
        let d_ok = m.mk_distinct([f_x, five]);
        assert!(strict_eu(d_ok, &vars, &m));

        // Unhandled kinds (here: a nested quantifier) are conservatively
        // required to mention no bound variable at all.
        let z = m.mk_var("z", int_sort);
        let x_le_z = m.mk_le(x, z);
        let nested_bad = m.mk_forall([("z", int_sort)], x_le_z);
        assert!(!strict_eu(nested_bad, &vars, &m));
        let c_le_z = m.mk_le(c, z);
        let nested_ok = m.mk_forall([("z", int_sort)], c_le_z);
        assert!(strict_eu(nested_ok, &vars, &m));

        // Whole-body eligibility: guard => strictly-EU consequent.
        let fx_eq0 = m.mk_eq(f_x, zero);
        let good_body = m.mk_implies(x_le_c, fx_eq0);
        assert!(is_eu_eligible(good_body, &vars, &m));
        let bad_body = m.mk_implies(x_le_c, x_ge);
        assert!(!is_eu_eligible(bad_body, &vars, &m));
    }

    /// Deep-nesting regression: the classification must return on a 128 KiB
    /// stack (returning at all is the proof; the verdicts pin behavior).
    /// The old three-way recursion burned one native frame per level.
    #[test]
    fn eu_walk_deep_chains_return_on_small_stack() {
        // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
        // ~10 B-per-frame threshold is the pin, so never raise one alone.
        const DEPTH: usize = 12_500;

        std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut m = TermManager::new();
                let int_sort = m.sorts.int_sort;
                let x = m.mk_var("x", int_sort);
                let one = m.mk_int(1);
                let mut chain = x;
                for _ in 0..DEPTH {
                    chain = m.mk_sub(chain, one);
                }
                let bound: FxHashSet<Spur> = [var_spur(&m, x)].into_iter().collect();
                // x sits under `DEPTH` arithmetic nodes, not a UF: reject.
                assert!(!strict_eu(chain, &bound, &m));
                // With no bound variables the same chain is acceptable.
                let unbound: FxHashSet<Spur> = FxHashSet::default();
                assert!(strict_eu(chain, &unbound, &m));
                // A bound variable at the bottom of a deep Apply chain is a
                // legal UF-argument position: accept even though x is bound.
                let mut uf_chain = x;
                for _ in 0..DEPTH {
                    uf_chain = m.mk_apply("f", [uf_chain], int_sort);
                }
                assert!(strict_eu(uf_chain, &bound, &m));
            })
            .expect("spawn eu_walk thread")
            .join()
            .expect("deep eu_walk must return, not overflow");
    }

    /// Shared-DAG regression: each level references its child twice (2^60
    /// paths); the `seen` set must bound the walk to one visit per distinct
    /// term.
    #[test]
    fn eu_walk_shared_dag_add_doubling_is_deduplicated() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let mut t = x;
        for _ in 0..60 {
            let prev = t;
            t = m.mk_add([prev, prev]);
            assert_ne!(t, prev, "doubling must build a fresh Add node");
        }
        let unbound: FxHashSet<Spur> = FxHashSet::default();
        assert!(strict_eu(t, &unbound, &m));
        let bound: FxHashSet<Spur> = [var_spur(&m, x)].into_iter().collect();
        assert!(!strict_eu(t, &bound, &m));
    }

    // ===== collect_guard_ground_terms (iterative guard scan) =====

    #[test]
    fn guard_grounds_pin_content_and_emission_order() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let k = m.mk_var("k", int_sort); // declared constant
        let vars: FxHashSet<Spur> = [var_spur(&m, x)].into_iter().collect();
        let five = m.mk_int(5);
        let three = m.mk_int(3);

        let le5 = m.mk_le(x, five); // x <= 5   -> contributes 5
        let eq_k = m.mk_eq(x, k);
        let ne_k = m.mk_not(eq_k); // x != k   -> contributes k
        let le3x = m.mk_le(three, x); // 3 <= x   -> contributes 3
        let guard = m.mk_and([le5, ne_k, le3x]);

        let mut out = Vec::new();
        collect_guard_ground_terms(guard, &vars, &m, &mut out);
        // Left-to-right emission order, exactly like the recursive version.
        assert_eq!(out, vec![five, k, three]);
    }

    /// Deep-nesting regression for the guard scan.  The connective
    /// alternates per level so the `mk_and`/`mk_or` constructors cannot
    /// flatten the spine (each splices only its own kind).
    #[test]
    fn guard_grounds_deep_alternating_chain_returns_on_small_stack() {
        // Stack and depth scale together (1 MiB/50k -> 128 KiB/6.25k): the
        // ~21 B-per-frame threshold is the pin, so never raise one alone.
        std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut m = TermManager::new();
                let int_sort = m.sorts.int_sort;
                let x = m.mk_var("x", int_sort);
                let vars: FxHashSet<Spur> = [var_spur(&m, x)].into_iter().collect();
                const LEVELS: usize = 6_250;
                let c0 = m.mk_int(0);
                let mut consts = vec![c0];
                let mut guard = m.mk_le(x, c0);
                for i in 1..=LEVELS {
                    let ci = m.mk_int(i as i64);
                    consts.push(ci);
                    let cmp = m.mk_le(x, ci);
                    guard = if i % 2 == 0 {
                        m.mk_and([guard, cmp])
                    } else {
                        m.mk_or([guard, cmp])
                    };
                }
                let mut out = Vec::new();
                collect_guard_ground_terms(guard, &vars, &m, &mut out);
                // Deepest-leftmost first: the recursive emission order.
                assert_eq!(out, consts);
            })
            .expect("spawn guard-scan thread")
            .join()
            .expect("deep guard scan must return, not overflow");
    }

    /// Shared-DAG regression: alternating connectives with both operands
    /// aliased give 2^60 paths to one comparison; the visited set must
    /// bound the scan.  The consumer deduplicates ground terms per sort
    /// bucket, so emitting the comparison's bound once is
    /// behavior-preserving.
    #[test]
    fn guard_grounds_shared_dag_is_bounded_by_visited_set() {
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let vars: FxHashSet<Spur> = [var_spur(&m, x)].into_iter().collect();
        let seven = m.mk_int(7);
        let mut g = m.mk_le(x, seven);
        for i in 0..60 {
            let prev = g;
            g = if i % 2 == 0 {
                m.mk_or([prev, prev])
            } else {
                m.mk_and([prev, prev])
            };
            assert_ne!(g, prev, "doubling must build a fresh connective");
        }
        let mut out = Vec::new();
        collect_guard_ground_terms(g, &vars, &m, &mut out);
        assert_eq!(out, vec![seven]);
    }
}
