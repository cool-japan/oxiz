//! Exact finite-range expansion of quantifiers over a finite box.
//!
//! # What this does
//!
//! A quantifier whose bound variables range over a *concrete, finite* set is
//! not really a quantifier at all: it is shorthand for a finite conjunction
//! (`forall`) or disjunction (`exists`).  This module recognises that shape and
//! rewrites the assertion into that ground formula **before** the Tseitin
//! encoder runs, so the ordinary ground solver (arithmetic + arrays + EUF +
//! SAT) decides it directly.
//!
//! Two disjoint fragments qualify, and [`variable_domains`] decides which one
//! applies from the bound sorts alone:
//!
//! ```text
//! (forall ((i Int)) (=> (and (>= i l) (<= i u)) C(i)))  ≡  ⋀_{v=l..u} body(v)
//! (exists ((i Int)) (and (>= i l) (<= i u) C(i)))       ≡  ⋁_{v=l..u} body(v)
//!
//! (forall ((i (_ BitVec 1))) C(i))                      ≡  C(#b0) ∧ C(#b1)
//! (exists ((b Bool))         C(b))                      ≡  C(false) ∨ C(true)
//! ```
//!
//! An `Int` is infinite, so the first fragment has to *find* its box in a guard
//! the quantifier carries.  A `Bool` or a `(_ BitVec w)` is finite by its own
//! definition, so the second needs no guard, no entailment and no extraction:
//! the sort **is** the box.  That second case is `#P2b-47`, and why it is worth
//! having is in [`variable_domains`]'s own doc — it is the case where MBQI
//! publishes a model that falsifies its own quantified assertion.
//!
//! # Why this is an *equivalence*, not an approximation
//!
//! The rewrite emits the **whole substituted body**, guard included — never
//! just the consequent.  That is what makes the two directions hold without any
//! assumption that the extracted interval is *tight*:
//!
//! * Every extracted bound comes from a top-level conjunct of the guard
//!   (`forall`) or of the body (`exists`).  A conjunct is a *necessary*
//!   condition, so the interval `[l, u]` is guaranteed to **contain** the whole
//!   region in which the guard (resp. the body) can be true.
//! * `forall`: outside `[l, u]` some guard conjunct is false, hence the guard is
//!   false, hence `guard ⇒ C` is vacuously true.  Only the points inside the
//!   box can constrain anything, and every one of them is instantiated.
//! * `exists`: outside `[l, u]` some body conjunct is false, hence the body is
//!   false.  Only the points inside the box can witness the formula, and every
//!   one of them becomes a disjunct.
//!
//! Because the result is *logically equivalent* to the quantifier, the rewrite
//! is polarity-independent: it is legal at any position in the formula, not
//! only on the asserted spine, and it preserves both `sat` and `unsat`.  An
//! empty interval (`l > u`) is likewise exact: `forall` collapses to `true`
//! (nothing to check) and `exists` to `false` (no witness can exist).
//!
//! # Soundness of the bound extraction
//!
//! Only *ground* bounds are accepted:
//!
//! * an integer literal, or
//! * a term the caller supplies in `entailed`, i.e. a term some **top-level
//!   assertion** already pins to a concrete integer (`(assert (= n 5))`), which
//!   therefore holds in every model of the assertion set, or
//! * a constant-foldable combination of those (`(+ n 1)`, `(- 0 3)`, ...).
//!
//! Anything else — a bound that mentions another bound variable, an
//! unconstrained symbol, a non-Int sort — makes the quantifier ineligible and
//! it falls through to the existing MBQI path completely unchanged.  Declining
//! costs completeness only, never soundness.
//!
//! # Budget
//!
//! The product of the per-variable interval widths must not exceed the caller's
//! budget ([`SolverConfig::finite_expansion_budget`](crate::solver::types::SolverConfig),
//! default [`DEFAULT_FINITE_EXPANSION_BUDGET`]).  A wider box is left to MBQI.
//!
//! Reference: Z3's `qe/qe_lite.cpp` / `smt/smt_quantifier.cpp` bounded-domain
//! expansion, and the small-domain case of MBQI (`smt/smt_model_finder.cpp`).

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use oxiz_core::ast::traversal::{collect_free_vars_including_patterns, get_children};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::sort::SortId;
use smallvec::SmallVec;

#[allow(unused_imports)]
use crate::prelude::*;

/// Default cap on the number of ground instances a single quantifier may be
/// expanded into (the product of its bound variables' interval widths).
///
/// 64 comfortably covers the shapes that occur in practice — a loop index over
/// a small array, a pair of indices over a 8×8 window — while keeping the
/// ground formula small enough that the expansion is always cheaper than the
/// MBQI round-trip it replaces.
pub(crate) const DEFAULT_FINITE_EXPANSION_BUDGET: usize = 64;

/// Maximum number of outermost-first expansion sweeps.
///
/// Each sweep expands one level of quantifier nesting (see
/// [`expandable_quantifiers`]), so `n` sweeps clear `n` levels.  A deeper nest
/// simply keeps its innermost levels, which fall through to MBQI unchanged.
const MAX_EXPANSION_SWEEPS: usize = 8;

/// Maximum number of quantifier sub-terms one assertion may carry before the
/// expansion declines it outright.
///
/// The eligibility filter runs one free-variable walk per candidate, so without
/// a cap a single assertion holding thousands of quantifiers would make
/// `assert` quadratic in its own size.  Real assertions carry a handful.
const MAX_CANDIDATE_QUANTIFIERS: usize = 64;

/// Rewrite every bounded-integer quantifier inside `term` into its exactly
/// equivalent ground conjunction / disjunction.
///
/// Returns `None` when nothing was expandable, in which case the caller must
/// keep using `term` unchanged.
pub(crate) fn expand_finite_quantifiers(
    term: TermId,
    manager: &mut TermManager,
    budget: usize,
    entailed: &FxHashMap<TermId, BigInt>,
) -> Option<TermId> {
    if budget == 0 {
        return None;
    }

    let mut current = term;
    let mut changed = false;

    for _ in 0..MAX_EXPANSION_SWEEPS {
        let candidates = expandable_quantifiers(current, manager);
        if candidates.is_empty() {
            break;
        }

        let mut rewrites: FxHashMap<TermId, TermId> = FxHashMap::default();
        for quantifier in candidates {
            if let Some(expansion) = expand_one(quantifier, manager, budget, entailed) {
                rewrites.insert(quantifier, expansion);
            }
        }
        if rewrites.is_empty() {
            break;
        }

        current = manager.substitute(current, &rewrites);
        changed = true;
    }

    changed.then_some(current)
}

/// Collect the `Forall` / `Exists` sub-terms of `term` that may be replaced by
/// a ground expansion **in place**.
///
/// # Why a scope filter is required
///
/// [`TermManager::substitute`] is capture-*avoiding*: splicing a replacement
/// whose free variables include a name some enclosing binder binds makes it
/// alpha-rename that binder rather than let the occurrence be captured.  For an
/// inner quantifier that genuinely reads an outer bound variable — `∀i∈[0,1].
/// ∃j∈[0,1]. a[j] = i` — that renaming silently detaches the spliced body from
/// the `∀`, turning `i` into an unconstrained free constant and answering `sat`
/// for an unsatisfiable goal.
///
/// So a quantifier is a candidate only when its free variables are disjoint
/// from **every** name bound by a binder (`Forall` / `Exists` / `Let` /
/// `Match`) anywhere in `term`.  The over-approximation — "anywhere in `term`"
/// rather than "on the path to this occurrence" — is deliberate: the term is a
/// hash-consed DAG, so the same sub-term can sit under several different
/// binder paths at once, and the rewrite map is applied to all of them.
///
/// Nesting still expands, outermost first: expanding `∀i` substitutes integer
/// *literals* for `i` (a replacement with no free variables, so no capture is
/// possible), which leaves the inner `∃j` free of `i` and therefore eligible on
/// the next sweep.
///
/// Explicit heap stack: a hash-consed DAG can nest arbitrarily deep and a
/// native recursion here would overflow the call stack on adversarial input.
fn expandable_quantifiers(term: TermId, manager: &TermManager) -> Vec<TermId> {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut quantifiers: Vec<TermId> = Vec::new();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        if matches!(kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
            quantifiers.push(current);
            if quantifiers.len() > MAX_CANDIDATE_QUANTIFIERS {
                // The scope filter below costs one free-variable walk per
                // candidate, so an assertion carrying a huge number of
                // quantifiers is declined wholesale rather than made quadratic
                // in its own size.  Declining costs completeness only: every
                // quantifier keeps its ordinary MBQI path.
                return Vec::new();
            }
        }
        stack.extend(get_children(&kind));
    }
    if quantifiers.is_empty() {
        return quantifiers;
    }

    let bound_anywhere = binder_names(term, manager);
    quantifiers.retain(|&quantifier| {
        !collect_free_vars_including_patterns(quantifier, manager)
            .iter()
            .filter_map(|&free| match manager.get(free).map(|t| &t.kind) {
                Some(TermKind::Var(name)) => Some(*name),
                _ => None,
            })
            .any(|name| bound_anywhere.contains(&name))
    });
    quantifiers
}

/// Every variable name bound by a binder anywhere in `term`.
pub(super) fn binder_names(term: TermId, manager: &TermManager) -> FxHashSet<Spur> {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut names: FxHashSet<Spur> = FxHashSet::default();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        match &kind {
            TermKind::Forall { vars, .. } | TermKind::Exists { vars, .. } => {
                names.extend(vars.iter().map(|&(name, _)| name));
            }
            TermKind::Let { bindings, .. } => {
                names.extend(bindings.iter().map(|&(name, _)| name));
            }
            TermKind::Match { cases, .. } => {
                for case in cases {
                    names.extend(case.bindings.iter().copied());
                }
            }
            _ => {}
        }
        stack.extend(get_children(&kind));
    }

    names
}

/// Whether `term` has a `Forall` / `Exists` sub-term (itself included).
pub(crate) fn contains_quantifier(term: TermId, manager: &TermManager) -> bool {
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
            return true;
        }
        stack.extend(get_children(&kind));
    }
    false
}

/// One quantifier's ground expansion, or `None` when it is outside the
/// finite-range fragment.
fn expand_one(
    quantifier: TermId,
    manager: &mut TermManager,
    budget: usize,
    entailed: &FxHashMap<TermId, BigInt>,
) -> Option<TermId> {
    let (vars, body, is_exists) = match manager.get(quantifier).map(|t| t.kind.clone())? {
        TermKind::Forall { vars, body, .. } => (vars, body, false),
        TermKind::Exists { vars, body, .. } => (vars, body, true),
        _ => return None,
    };
    if vars.is_empty() {
        return None;
    }

    let domains = match variable_domains(&vars, body, is_exists, manager, budget, entailed)? {
        // Empty range: the quantifier is a constant.  `forall` over an empty
        // range is vacuously true, `exists` over one has no witness.
        Domains::Empty => {
            return Some(if is_exists {
                manager.mk_false()
            } else {
                manager.mk_true()
            });
        }
        Domains::Points(points) => points,
    };

    // Odometer over the box, variable 0 varying slowest.
    let mut offsets: Vec<usize> = vec![0; domains.len()];
    let product = domains.iter().map(Vec::len).product();
    let mut instances: Vec<TermId> = Vec::with_capacity(product);
    loop {
        let mut subst: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (index, &(name, sort)) in vars.iter().enumerate() {
            let offset = *offsets.get(index)?;
            let value = *domains.get(index)?.get(offset)?;
            let name_str = manager.resolve_str(name).to_string();
            let var_term = manager.mk_var(&name_str, sort);
            subst.insert(var_term, value);
        }

        let instance = manager.substitute(body, &subst);
        // A bound variable surviving the substitution would leave a stray free
        // variable in the ground formula.  That must never be encoded, so the
        // whole expansion is abandoned and the quantifier keeps its MBQI path.
        let free = collect_free_vars_including_patterns(instance, manager);
        if subst.keys().any(|key| free.contains(key)) {
            return None;
        }
        instances.push(instance);

        // Advance the odometer.
        let mut position = offsets.len();
        loop {
            if position == 0 {
                // Every point of the box has been emitted.
                return Some(if is_exists {
                    manager.mk_or(instances)
                } else {
                    manager.mk_and(instances)
                });
            }
            position -= 1;
            let width = domains.get(position)?.len();
            let slot = offsets.get_mut(position)?;
            *slot += 1;
            if *slot < width {
                break;
            }
            *slot = 0;
        }
    }
}

/// The ground points each bound variable ranges over.
enum Domains {
    /// One list of ground terms per bound variable, in the declaration order
    /// of `vars`.  Every list is non-empty.
    Points(Vec<Vec<TermId>>),
    /// Some variable's range is empty, so the quantifier is a constant.
    Empty,
}

/// The box `vars` ranges over, or `None` when the quantifier is outside the
/// finite fragment (or its box is wider than `budget`).
///
/// Two disjoint fragments are recognised, and which one applies is decided by
/// the bound sorts alone:
///
/// 1. **Every variable is `Int`** — the box has to be *found*, in a guard the
///    quantifier carries (see [`extract_bounds`]).  An `Int` is infinite, so
///    without such a guard there is no finite expansion.
/// 2. **Every variable has a finite sort** (`Bool`, `(_ BitVec w)`) — the box
///    *is* the sort.  Nothing has to be extracted, no guard has to exist, and
///    no entailment has to be established: `(forall ((i (_ BitVec 1))) C(i))`
///    **is** `(and C(#b0) C(#b1))`, by the meaning of the sort.
///
/// A quantifier mixing the two, or binding any other sort, is declined; that
/// costs completeness only, and it keeps its ordinary MBQI path.
///
/// # Why case 2 is worth having (`#P2b-47`)
///
/// It is the one case where MBQI is not merely slower but *wrong about the
/// model it publishes*.  MBQI answers `Satisfied` when no instance it chose to
/// build is violated by the candidate, which leaves every index it did not
/// choose unconstrained — and `(get-model)` then renders the array at those
/// indices from the sort default.  Fourteen of the 800 paired scripts in
/// `rc5/corpus/{q,qnoite}` answered a correct `sat` and printed a model that
/// falsifies their own quantified assertion for exactly that reason, every one
/// of them a single `(forall ((i (_ BitVec w))) …)` over a two- or four-element
/// index sort.  Expanding the sort makes the quantified script *be* its own
/// ground expansion, which is the form the same 800 scripts already decide
/// correctly, model included.
fn variable_domains(
    vars: &[(Spur, SortId)],
    body: TermId,
    is_exists: bool,
    manager: &mut TermManager,
    budget: usize,
    entailed: &FxHashMap<TermId, BigInt>,
) -> Option<Domains> {
    let int_sort = manager.sorts.int_sort;
    if vars.iter().all(|&(_, sort)| sort == int_sort) {
        let bounds = extract_bounds(body, vars, is_exists, manager, entailed)?;
        if bounds.iter().any(|(lo, hi)| lo > hi) {
            return Some(Domains::Empty);
        }
        let mut product: usize = 1;
        for (lo, hi) in &bounds {
            let width = ((hi - lo) + 1u32).to_usize()?;
            product = product.checked_mul(width)?;
            if product > budget {
                return None;
            }
        }
        let mut points: Vec<Vec<TermId>> = Vec::with_capacity(bounds.len());
        for (lo, hi) in &bounds {
            let width = ((hi - lo) + 1u32).to_usize()?;
            let mut column: Vec<TermId> = Vec::with_capacity(width);
            for offset in 0..width {
                column.push(manager.mk_int(lo + BigInt::from(offset)));
            }
            points.push(column);
        }
        return Some(Domains::Points(points));
    }

    // Case 2: the sorts themselves are the box.
    let mut cardinalities: Vec<usize> = Vec::with_capacity(vars.len());
    let mut product: usize = 1;
    for &(_, sort) in vars {
        let card = finite_sort_cardinality(sort, manager)?;
        product = product.checked_mul(card)?;
        if product > budget {
            return None;
        }
        cardinalities.push(card);
    }
    let mut points: Vec<Vec<TermId>> = Vec::with_capacity(vars.len());
    for (index, &(_, sort)) in vars.iter().enumerate() {
        let card = *cardinalities.get(index)?;
        let mut column: Vec<TermId> = Vec::with_capacity(card);
        for value in 0..card {
            column.push(finite_sort_element(sort, value, manager)?);
        }
        points.push(column);
    }
    Some(Domains::Points(points))
}

/// How many elements `sort` has, when it has finitely many *and* each of them
/// can be written as a ground term.
///
/// `Bool` and `(_ BitVec w)` are the only two sorts that qualify here.  A
/// declared (uninterpreted) sort is finite in every finite model but has no
/// ground spelling for its elements, so it is not enumerable and is declined;
/// `Int`, `Real`, `String`, `FloatingPoint` (whose `NaN` payloads and infinities
/// make the naive `2^(eb+sb)` count wrong) and `Array` are all declined too.
///
/// The width guard is not only about cost: `1usize << 64` is undefined, and a
/// width at or above 64 cannot have its cardinality represented in a `usize`
/// at all.  Every such sort is far past any sane `budget` anyway.
fn finite_sort_cardinality(sort: SortId, manager: &TermManager) -> Option<usize> {
    match manager.sorts.get(sort)?.kind {
        oxiz_core::sort::SortKind::Bool => Some(2),
        oxiz_core::sort::SortKind::BitVec(width) if width > 0 && width < 32 => {
            Some(1usize << width)
        }
        _ => None,
    }
}

/// Element number `index` of `sort`, counting from zero.
///
/// The order is the one the sort's own values have — `false` before `true`,
/// and unsigned increasing for a bit-vector — so the expansion of
/// `(forall ((i (_ BitVec 2))) C(i))` reads `#b00`, `#b01`, `#b10`, `#b11`.
/// Nothing depends on the order (a conjunction is commutative), but a *fixed*
/// order is what makes the expanded term identical on every run, which the
/// determinism gate requires.
fn finite_sort_element(sort: SortId, index: usize, manager: &mut TermManager) -> Option<TermId> {
    let kind = manager.sorts.get(sort)?.kind.clone();
    match kind {
        oxiz_core::sort::SortKind::Bool => Some(manager.mk_bool(index == 1)),
        oxiz_core::sort::SortKind::BitVec(width) => {
            Some(manager.mk_bitvec(BigInt::from(index), width))
        }
        _ => None,
    }
}

/// Extract a concrete `[lo, hi]` interval for **every** bound variable, in the
/// declaration order of `vars`.
///
/// * `forall`: only the premise of a top-level `Implies` is inspected, so that
///   outside the box the implication is vacuously true.
/// * `exists`: the body's own top-level conjuncts are inspected, so that
///   outside the box the body is false.
///
/// Returns `None` unless every variable gets both a lower and an upper bound.
fn extract_bounds(
    body: TermId,
    vars: &[(Spur, SortId)],
    is_exists: bool,
    manager: &TermManager,
    entailed: &FxHashMap<TermId, BigInt>,
) -> Option<Vec<(BigInt, BigInt)>> {
    let guard = if is_exists {
        body
    } else {
        match manager.get(body).map(|t| t.kind.clone())? {
            TermKind::Implies(premise, _) => premise,
            // `(or (not g) c)` — and its mirror `(or c (not g))` — is the same
            // implication after normalisation.  Either negated disjunct may be
            // read as the guard: `(or (not g) x)` ≡ `g ⇒ x`, so outside the
            // interval `g` pins, the disjunction is true either way.
            TermKind::Or(args) if args.len() == 2 => {
                args.iter()
                    .find_map(|&arg| match manager.get(arg).map(|t| t.kind.clone()) {
                        Some(TermKind::Not(inner)) => Some(inner),
                        _ => None,
                    })?
            }
            _ => return None,
        }
    };

    let names: FxHashSet<Spur> = vars.iter().map(|&(name, _)| name).collect();
    let conjuncts: SmallVec<[TermId; 8]> = match manager.get(guard).map(|t| t.kind.clone())? {
        TermKind::And(args) => args.iter().copied().collect(),
        _ => core::iter::once(guard).collect(),
    };

    let mut lowers: FxHashMap<Spur, BigInt> = FxHashMap::default();
    let mut uppers: FxHashMap<Spur, BigInt> = FxHashMap::default();

    for atom in conjuncts {
        let Some(kind) = manager.get(atom).map(|t| t.kind.clone()) else {
            continue;
        };
        let (lhs, rhs, rel) = match &kind {
            TermKind::Ge(l, r) => (*l, *r, Rel::Ge),
            TermKind::Gt(l, r) => (*l, *r, Rel::Gt),
            TermKind::Le(l, r) => (*l, *r, Rel::Le),
            TermKind::Lt(l, r) => (*l, *r, Rel::Lt),
            // `(= i c)` pins the variable to a single point.
            TermKind::Eq(l, r) => (*l, *r, Rel::Eq),
            _ => continue,
        };

        // Normalise to `variable rel ground`; a variable on the right flips it.
        let (name, value, rel) = if let Some(name) = bound_var_name(lhs, &names, manager) {
            match ground_int(rhs, manager, entailed) {
                Some(value) => (name, value, rel),
                None => continue,
            }
        } else if let Some(name) = bound_var_name(rhs, &names, manager) {
            match ground_int(lhs, manager, entailed) {
                Some(value) => (name, value, rel.flip()),
                None => continue,
            }
        } else {
            continue;
        };

        match rel {
            Rel::Ge => tighten(&mut lowers, name, value, true),
            Rel::Gt => tighten(&mut lowers, name, value + 1, true),
            Rel::Le => tighten(&mut uppers, name, value, false),
            Rel::Lt => tighten(&mut uppers, name, value - 1, false),
            Rel::Eq => {
                tighten(&mut lowers, name, value.clone(), true);
                tighten(&mut uppers, name, value, false);
            }
        }
    }

    let mut bounds = Vec::with_capacity(vars.len());
    for &(name, _) in vars {
        let lo = lowers.get(&name)?.clone();
        let hi = uppers.get(&name)?.clone();
        bounds.push((lo, hi));
    }
    Some(bounds)
}

/// A comparison relation, oriented as `variable rel ground`.
#[derive(Clone, Copy)]
enum Rel {
    Ge,
    Gt,
    Le,
    Lt,
    Eq,
}

impl Rel {
    /// Reverse the relation (used when the bound variable is on the right).
    fn flip(self) -> Self {
        match self {
            Rel::Ge => Rel::Le,
            Rel::Gt => Rel::Lt,
            Rel::Le => Rel::Ge,
            Rel::Lt => Rel::Gt,
            Rel::Eq => Rel::Eq,
        }
    }
}

/// Record a bound, keeping the tighter of the two when one already exists.
fn tighten(map: &mut FxHashMap<Spur, BigInt>, name: Spur, value: BigInt, is_lower: bool) {
    map.entry(name)
        .and_modify(|existing| {
            let tighter = if is_lower {
                value > *existing
            } else {
                value < *existing
            };
            if tighter {
                *existing = value.clone();
            }
        })
        .or_insert(value);
}

/// If `term` is one of the quantifier's own bound variables, its name.
fn bound_var_name(term: TermId, names: &FxHashSet<Spur>, manager: &TermManager) -> Option<Spur> {
    match manager.get(term).map(|t| &t.kind) {
        Some(TermKind::Var(name)) if names.contains(name) => Some(*name),
        _ => None,
    }
}

/// Fold `term` to a concrete integer, or `None` when it is not ground.
///
/// Recognises integer literals, terms the assertion set already pins to a
/// literal (`entailed`), and `+` / `-` / `*` / unary `-` combinations of those.
/// Explicit heap stack: the folded expression may nest arbitrarily deep.
fn ground_int(
    term: TermId,
    manager: &TermManager,
    entailed: &FxHashMap<TermId, BigInt>,
) -> Option<BigInt> {
    /// One step of the post-order fold.
    enum Step {
        /// Resolve `term`, pushing its children first when it is an operator.
        Visit(TermId),
        /// Combine the top `arity` values already on the value stack.
        Combine(Op, usize),
    }
    #[derive(Clone, Copy)]
    enum Op {
        Add,
        Sub,
        Mul,
        Neg,
    }

    /// Cap on folded nodes: a ground bound in a real script is tiny, and this
    /// keeps a pathological term from turning bound extraction into the
    /// dominant cost of `assert`.
    const FOLD_NODE_LIMIT: usize = 4096;

    let mut steps: Vec<Step> = vec![Step::Visit(term)];
    let mut values: Vec<BigInt> = Vec::new();
    let mut visited_nodes: usize = 0;

    while let Some(step) = steps.pop() {
        match step {
            Step::Visit(current) => {
                visited_nodes += 1;
                if visited_nodes > FOLD_NODE_LIMIT {
                    return None;
                }
                if let Some(value) = entailed.get(&current) {
                    values.push(value.clone());
                    continue;
                }
                match manager.get(current).map(|t| t.kind.clone())? {
                    TermKind::IntConst(value) => values.push(value),
                    TermKind::Neg(arg) => {
                        steps.push(Step::Combine(Op::Neg, 1));
                        steps.push(Step::Visit(arg));
                    }
                    TermKind::Sub(lhs, rhs) => {
                        steps.push(Step::Combine(Op::Sub, 2));
                        steps.push(Step::Visit(rhs));
                        steps.push(Step::Visit(lhs));
                    }
                    TermKind::Add(args) => {
                        steps.push(Step::Combine(Op::Add, args.len()));
                        for &arg in args.iter().rev() {
                            steps.push(Step::Visit(arg));
                        }
                    }
                    TermKind::Mul(args) => {
                        steps.push(Step::Combine(Op::Mul, args.len()));
                        for &arg in args.iter().rev() {
                            steps.push(Step::Visit(arg));
                        }
                    }
                    _ => return None,
                }
            }
            Step::Combine(op, arity) => {
                if values.len() < arity {
                    return None;
                }
                let operands = values.split_off(values.len() - arity);
                let folded = match op {
                    Op::Neg => -operands.first()?.clone(),
                    Op::Sub => operands.first()?.clone() - operands.get(1)?.clone(),
                    Op::Add => operands.iter().fold(BigInt::from(0), |acc, v| acc + v),
                    Op::Mul => operands.iter().fold(BigInt::from(1), |acc, v| acc * v),
                };
                values.push(folded);
            }
        }
    }

    match values.len() {
        1 => values.pop(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::ast::TermManager;

    /// `#P2b-47`: a `forall` over `(_ BitVec 2)` **is** the four-way
    /// conjunction over that sort's elements, with no guard anywhere.
    ///
    /// No `Int` quantifier can be expanded without one, which is why the
    /// bounded-`Int` fragment goes looking; this one needs nothing but the
    /// sort.
    #[test]
    fn forall_over_a_bit_vector_sort_expands_over_the_whole_sort() {
        let mut manager = TermManager::new();
        let index_sort = manager.sorts.bitvec(2);
        let elem_sort = manager.sorts.bitvec(1);
        let array_sort = manager.sorts.array(index_sort, elem_sort);

        let array = manager.mk_var("a", array_sort);
        let index = manager.mk_var("i", index_sort);
        let read = manager.mk_select(array, index);
        let one = manager.mk_bitvec(1u32, 1);
        let body = manager.mk_eq(read, one);
        let quantifier = manager.mk_forall([("i", index_sort)], body);

        let expanded =
            expand_finite_quantifiers(quantifier, &mut manager, 64, &FxHashMap::default())
                .expect("a bit-vector sort is its own finite box");
        match manager.get(expanded).map(|t| t.kind.clone()) {
            Some(TermKind::And(args)) => assert_eq!(args.len(), 4, "2^2 elements"),
            other => panic!("expected a 4-way conjunction, got {other:?}"),
        }
        assert!(!contains_quantifier(expanded, &manager));
    }

    /// The `exists` direction, and the `Bool` sort: two elements, a
    /// disjunction.
    #[test]
    fn exists_over_bool_expands_to_both_truth_values() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let predicate_arg = manager.mk_var("b", bool_sort);
        let body = manager.mk_apply("p", [predicate_arg], bool_sort);
        let quantifier = manager.mk_exists([("b", bool_sort)], body);

        let expanded =
            expand_finite_quantifiers(quantifier, &mut manager, 64, &FxHashMap::default())
                .expect("Bool is a two-element sort");
        match manager.get(expanded).map(|t| t.kind.clone()) {
            Some(TermKind::Or(args)) => assert_eq!(args.len(), 2),
            other => panic!("expected a 2-way disjunction, got {other:?}"),
        }
    }

    /// The budget is the whole guard against a blow-up here, because the sort
    /// alone decides the size: `(_ BitVec 8)` is 256 points, so at the default
    /// budget of 64 it must be declined and keep its MBQI path.
    #[test]
    fn a_bit_vector_sort_wider_than_the_budget_is_declined() {
        let mut manager = TermManager::new();
        let index_sort = manager.sorts.bitvec(8);
        let bool_sort = manager.sorts.bool_sort;
        let index = manager.mk_var("i", index_sort);
        let body = manager.mk_apply("p", [index], bool_sort);
        let quantifier = manager.mk_forall([("i", index_sort)], body);

        assert!(
            expand_finite_quantifiers(quantifier, &mut manager, 64, &FxHashMap::default())
                .is_none(),
            "256 points against a budget of 64 must decline"
        );
        // ... and the same quantifier IS expanded when the budget allows it,
        // so the decline above is the budget and not the sort.
        assert!(
            expand_finite_quantifiers(quantifier, &mut manager, 256, &FxHashMap::default())
                .is_some()
        );
    }

    /// A sort with no ground spelling for its elements is declined even though
    /// it is finite in every finite model: an uninterpreted sort's elements
    /// cannot be written down, so there is nothing to substitute.
    #[test]
    fn an_uninterpreted_sort_is_not_enumerable() {
        let mut manager = TermManager::new();
        let spur = manager.intern_str("U");
        let opaque = manager
            .sorts
            .intern(oxiz_core::sort::SortKind::Uninterpreted(spur));
        let bool_sort = manager.sorts.bool_sort;
        let element = manager.mk_var("u", opaque);
        let body = manager.mk_apply("p", [element], bool_sort);
        let quantifier = manager.mk_forall([("u", opaque)], body);

        assert!(
            expand_finite_quantifiers(quantifier, &mut manager, 64, &FxHashMap::default())
                .is_none()
        );
        assert!(finite_sort_cardinality(opaque, &manager).is_none());
    }

    /// Mixing an `Int` with a finite sort is outside both fragments: the box
    /// would have to be half found and half taken from the sort, and nothing
    /// here does that.
    #[test]
    fn a_mixed_int_and_bit_vector_binder_is_declined() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bv_sort = manager.sorts.bitvec(1);
        let bool_sort = manager.sorts.bool_sort;
        let i = manager.mk_var("i", int_sort);
        let b = manager.mk_var("b", bv_sort);
        let body = manager.mk_apply("p", [i, b], bool_sort);
        let quantifier = manager.mk_forall([("i", int_sort), ("b", bv_sort)], body);

        assert!(
            expand_finite_quantifiers(quantifier, &mut manager, 64, &FxHashMap::default())
                .is_none()
        );
    }

    /// `∀i. (0 ≤ i ∧ i ≤ 2) ⇒ p(i)` expands to the three-way conjunction.
    #[test]
    fn forall_literal_box_expands_to_conjunction() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let index = manager.mk_var("i", int_sort);
        let zero = manager.mk_int(0);
        let two = manager.mk_int(2);
        let lower = manager.mk_ge(index, zero);
        let upper = manager.mk_le(index, two);
        let guard = manager.mk_and([lower, upper]);
        let predicate = manager.mk_apply("p", [index], bool_sort);
        let body = manager.mk_implies(guard, predicate);
        let quantifier = manager.mk_forall([("i", int_sort)], body);

        let entailed = FxHashMap::default();
        let expanded = expand_finite_quantifiers(quantifier, &mut manager, 64, &entailed)
            .expect("bounded box must expand");

        match manager.get(expanded).map(|t| t.kind.clone()) {
            Some(TermKind::And(args)) => assert_eq!(args.len(), 3),
            other => panic!("expected a 3-way conjunction, got {other:?}"),
        }
        assert!(!contains_quantifier(expanded, &manager));
    }

    /// `∃i. (0 ≤ i ∧ i ≤ 3 ∧ p(i))` expands to the four-way disjunction that
    /// lets the ground solver pick the witness.
    #[test]
    fn exists_literal_box_expands_to_disjunction() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let index = manager.mk_var("i", int_sort);
        let zero = manager.mk_int(0);
        let three = manager.mk_int(3);
        let lower = manager.mk_ge(index, zero);
        let upper = manager.mk_le(index, three);
        let predicate = manager.mk_apply("p", [index], bool_sort);
        let body = manager.mk_and([lower, upper, predicate]);
        let quantifier = manager.mk_exists([("i", int_sort)], body);

        let entailed = FxHashMap::default();
        let expanded = expand_finite_quantifiers(quantifier, &mut manager, 64, &entailed)
            .expect("bounded box must expand");

        match manager.get(expanded).map(|t| t.kind.clone()) {
            Some(TermKind::Or(args)) => assert_eq!(args.len(), 4),
            other => panic!("expected a 4-way disjunction, got {other:?}"),
        }
    }

    /// An empty interval is still an exact rewrite: `∀` is vacuously true and
    /// `∃` has no witness at all.
    #[test]
    fn empty_interval_collapses_to_a_constant() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let index = manager.mk_var("i", int_sort);
        let five = manager.mk_int(5);
        let two = manager.mk_int(2);
        let lower = manager.mk_ge(index, five);
        let upper = manager.mk_le(index, two);
        let predicate = manager.mk_apply("p", [index], bool_sort);

        let guard = manager.mk_and([lower, upper]);
        let forall_body = manager.mk_implies(guard, predicate);
        let forall = manager.mk_forall([("i", int_sort)], forall_body);
        let expanded_forall =
            expand_finite_quantifiers(forall, &mut manager, 64, &FxHashMap::default())
                .expect("empty box must expand");
        assert!(matches!(
            manager.get(expanded_forall).map(|t| t.kind.clone()),
            Some(TermKind::True)
        ));

        let exists_body = manager.mk_and([lower, upper, predicate]);
        let exists = manager.mk_exists([("i", int_sort)], exists_body);
        let expanded_exists =
            expand_finite_quantifiers(exists, &mut manager, 64, &FxHashMap::default())
                .expect("empty box must expand");
        assert!(matches!(
            manager.get(expanded_exists).map(|t| t.kind.clone()),
            Some(TermKind::False)
        ));
    }

    /// A box wider than the budget is declined outright, so the quantifier
    /// keeps its ordinary MBQI path.
    #[test]
    fn over_budget_box_is_declined() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let index = manager.mk_var("i", int_sort);
        let zero = manager.mk_int(0);
        let hundred = manager.mk_int(100);
        let lower = manager.mk_ge(index, zero);
        let upper = manager.mk_le(index, hundred);
        let guard = manager.mk_and([lower, upper]);
        let predicate = manager.mk_apply("p", [index], bool_sort);
        let body = manager.mk_implies(guard, predicate);
        let quantifier = manager.mk_forall([("i", int_sort)], body);

        assert!(
            expand_finite_quantifiers(quantifier, &mut manager, 64, &FxHashMap::default())
                .is_none(),
            "a 101-point box must not be expanded under a budget of 64"
        );
    }

    /// A symbolic bound is only concrete once the caller supplies the value the
    /// assertion set entails for it; without that entry the quantifier falls
    /// through unchanged.
    #[test]
    fn symbolic_bound_needs_an_entailed_value() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let index = manager.mk_var("i", int_sort);
        let limit = manager.mk_var("n", int_sort);
        let zero = manager.mk_int(0);
        let lower = manager.mk_ge(index, zero);
        let upper = manager.mk_lt(index, limit);
        let guard = manager.mk_and([lower, upper]);
        let predicate = manager.mk_apply("p", [index], bool_sort);
        let body = manager.mk_implies(guard, predicate);
        let quantifier = manager.mk_forall([("i", int_sort)], body);

        assert!(
            expand_finite_quantifiers(quantifier, &mut manager, 64, &FxHashMap::default())
                .is_none(),
            "an unpinned `n` is not a concrete bound"
        );

        let mut entailed: FxHashMap<TermId, BigInt> = FxHashMap::default();
        entailed.insert(limit, BigInt::from(4));
        let expanded = expand_finite_quantifiers(quantifier, &mut manager, 64, &entailed)
            .expect("`n = 4` makes `i < n` the interval [0, 3]");
        match manager.get(expanded).map(|t| t.kind.clone()) {
            Some(TermKind::And(args)) => assert_eq!(args.len(), 4),
            other => panic!("expected a 4-way conjunction, got {other:?}"),
        }
    }

    /// An inner quantifier that reads an enclosing bound variable must not be
    /// spliced in place: `substitute` would alpha-rename the enclosing binder
    /// and detach the body from it.  It becomes eligible only after the outer
    /// quantifier has been expanded into ground instances.
    #[test]
    fn inner_quantifier_over_outer_variable_waits_for_the_outer_expansion() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let outer = manager.mk_var("x", int_sort);
        let inner = manager.mk_var("y", int_sort);
        let zero = manager.mk_int(0);
        let one = manager.mk_int(1);

        let inner_lower = manager.mk_ge(inner, zero);
        let inner_upper = manager.mk_le(inner, one);
        let related = manager.mk_apply("q", [outer, inner], bool_sort);
        let inner_body = manager.mk_and([inner_lower, inner_upper, related]);
        let exists = manager.mk_exists([("y", int_sort)], inner_body);

        // On its own the inner existential mentions the *free* `x`, but nothing
        // binds `x` in that term, so it is expandable.
        assert!(
            expand_finite_quantifiers(exists, &mut manager, 64, &FxHashMap::default()).is_some()
        );

        let outer_lower = manager.mk_ge(outer, zero);
        let outer_upper = manager.mk_le(outer, one);
        let outer_guard = manager.mk_and([outer_lower, outer_upper]);
        let outer_body = manager.mk_implies(outer_guard, exists);
        let forall = manager.mk_forall([("x", int_sort)], outer_body);

        let expanded = expand_finite_quantifiers(forall, &mut manager, 64, &FxHashMap::default())
            .expect("the outer box expands, and the inner one follows it");
        assert!(
            !contains_quantifier(expanded, &manager),
            "both levels must be ground after the sweeps"
        );
        // `x` was replaced by literals, so no free `x` may survive.
        let free = collect_free_vars_including_patterns(expanded, &manager);
        assert!(
            !free.contains(&outer),
            "the outer bound variable must not leak out as a free constant"
        );
    }
}
