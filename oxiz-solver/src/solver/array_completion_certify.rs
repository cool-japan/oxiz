//! Model **completion** for an array default under a binder, and the
//! quantifier-free **certificate** that has to pass before the `sat` it
//! licenses may be published (`#P2b-58`, decision (36)).
//!
//! # The gap this closes
//!
//! `finite_expand` decides a universal over a finite sort by writing it out at
//! every point, and it declines above a 64-point budget.  One bit above that
//! budget — `(_ BitVec 7)`, 128 points — a *satisfiable* quantified array
//! script stops being decided: MBQI reaches its fixpoint without ever being
//! able to say what the array is at the indices no ground term names, so the
//! candidate model leaves the array's **default** free and nothing may be
//! published from it.  `rk6/corpus/qmbqi120/q0074` is the minimal shape: its
//! quantifier reduces to `a1[i] = #b1` for every `i` over `(_ BitVec 7)`, which
//! is satisfiable exactly by the *constant* array `#b1` — an interpretation the
//! candidate model has no way to name.
//!
//! # What this module does
//!
//! Given the candidate model, it *completes* every array-sorted free variable
//! to a total interpretation — a constant array over a searched default, in the
//! spirit of Ge & de Moura's pins-plus-default model construction (CAV 2009)
//! and of [`crate::mbqi::model_certify`], which does the same thing for
//! uninterpreted functions over `Int` and `Real` — and then **certifies** the
//! completion before anything is published.
//!
//! # Why the certificate is sound, and why it is only ever one-directional
//!
//! Every query this module runs is a *validity* query, discharged by refuting
//! its negation with an ordinary quantifier-free solve:
//!
//! * a maximal `forall` sub-term `∀x⃗. ψ` is accepted as **true** only when
//!   `¬ψ[completion]`, with `x⃗` replaced by fresh reserved constants, is
//!   `Unsat`.  `Unsat` means `ψ[completion]` holds for *every* interpretation
//!   of whatever symbols are left in it, hence in particular under ours — so
//!   the conclusion survives any symbol this module failed to interpret.  The
//!   converse is **not** sound for the same reason (a `Sat` says some
//!   interpretation falsifies it, not that ours does), so a `forall` that
//!   cannot be certified true makes the whole attempt decline rather than be
//!   recorded as false.  `exists` is declined outright: certifying one true
//!   needs a witness, which is the direction this argument does not give.
//! * the assertions themselves are accepted only when
//!   `(or (not A₁[completion]) … (not Aₙ[completion]))` is `Unsat`, which is
//!   the same one-directional argument taken over all of them at once — one
//!   quantifier-free query, exactly as decision (36) asks.
//!
//! So a `sat` is published only behind a passing certificate, and a *wrong*
//! completion can never produce one: `rk11/atk/m4_body_false_at_a_point.smt2`
//! and `m4b_stored_point_conflict.smt2` are refuted rather than published.
//!
//! # What it is not
//!
//! It is not a decision procedure for the fragment and does not pretend to be:
//! the default pool is finite and the combination count is capped, so a script
//! whose satisfying interpretation is not a constant array over one of the
//! pooled values keeps its `unknown`.  Declining costs only completeness.
//! `#P2b-50` — a *declared* sort whose cardinality nothing pins — stays
//! separate and open; this module never invents a domain, it only interprets
//! an array whose index and element sorts are both already pinned.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::smtlib::reserved_name;
use oxiz_core::sort::{SortId, SortKind};
use rustc_hash::{FxHashMap, FxHashSet};

use super::array_axioms::CONST_ARRAY_FUNC;
use super::{Solver, SolverResult};

/// Array-sorted free variables one attempt may complete.
///
/// The search is a product over the pool below, so this is what keeps it
/// finite.  Three covers every script `#P2b-58` names (`q0106` has three
/// arrays under one binder) and bounds the product at `POOL^3`.
const MAX_COMPLETED_ARRAYS: usize = 3;

/// Default values tried per array.
const MAX_DEFAULT_CANDIDATES: usize = 6;

/// Default *combinations* across all completed arrays.
const MAX_COMBINATIONS: usize = 24;

/// Assertion-DAG nodes above which an attempt declines before it starts.
///
/// The certificate is a *validity* query per universal plus one for the
/// assertions, and a goal this module could actually discharge is small by
/// construction — the whole `#P2b-58` family is two to five assertions.  The
/// cap is what keeps a large benchmark from paying for a search that would
/// decline anyway.
const MAX_GOAL_NODES: usize = 4_096;

/// Quantifier-free certificate queries one attempt may spend in total.
///
/// Each query is a fresh, budgeted, quantifier-free solve; this bounds the
/// cost of the whole attempt independently of how the search goes.
const MAX_CERTIFICATE_QUERIES: usize = 96;

/// Boolean conflicts one certificate query may spend.
///
/// A deterministic budget, not a clock: the certificate's verdict must be a
/// property of the tree and not of the machine (decision (9)).  A query that
/// runs out answers `Unknown`, which this module reads as "not certified".
const CERTIFICATE_CONFLICT_BUDGET: u64 = 20_000;

/// Maximal quantifier sub-terms, plus the free symbols an attempt must pin.
struct Goal {
    /// The assertions, unchanged.
    assertions: Vec<TermId>,
    /// Every maximal `forall` sub-term of the assertion set, deduplicated and
    /// in first-encounter order so the attempt is deterministic.
    universals: Vec<TermId>,
    /// The array-sorted free variables to complete, in first-encounter order.
    arrays: Vec<TermId>,
    /// `scalar -> value`, the pins taken from the candidate model.
    scalar_pins: FxHashMap<TermId, TermId>,
}

impl Solver {
    /// Try to publish `Sat` for a quantified array goal by completing the
    /// arrays the candidate model leaves partial and certifying the result.
    ///
    /// Returns `true` only with a *certified* completion in hand, and installs
    /// it into [`Solver::model`] so `(get-model)` prints the interpretation the
    /// certificate verified rather than the partial candidate.  A `false`
    /// leaves the solver exactly as it was found.
    pub(super) fn certify_sat_by_array_completion(&mut self, manager: &mut TermManager) -> bool {
        if !self.has_array_ops || !self.has_quantifiers || self.assertions.is_empty() {
            return false;
        }
        let Some(model) = self.model.as_ref() else {
            return false;
        };
        let assignments = model.assignments().clone();
        let assertions = self.assertions.clone();
        if !is_small_enough_to_certify(&assertions, manager) {
            return false;
        }
        let Some(goal) = Goal::build(&assertions, &assignments, manager) else {
            return false;
        };
        let pools = goal.default_pools(&assignments, manager);
        if pools.iter().any(Vec::is_empty) {
            return false;
        }
        let logic = self.logic.clone();
        let mut queries = 0usize;
        for combination in Combinations::new(&pools) {
            let mut completion: FxHashMap<TermId, TermId> = FxHashMap::default();
            for (&array, &default) in goal.arrays.iter().zip(combination.iter()) {
                let Some(sort) = manager.get(array).map(|d| d.sort) else {
                    return false;
                };
                let interpretation = const_array(sort, default, manager);
                completion.insert(array, interpretation);
            }
            if goal.certificate_passes(&completion, manager, logic.as_deref(), &mut queries) {
                self.install_completed_model(&goal, &completion);
                return true;
            }
            if queries >= MAX_CERTIFICATE_QUERIES {
                return false;
            }
        }
        false
    }

    /// Record the certified interpretation in the published model.
    ///
    /// The array variable is bound to `((as const A) d)` — the term the
    /// certificate was discharged over, so what `(get-model)` prints and what
    /// was verified are the same object.
    fn install_completed_model(&mut self, goal: &Goal, completion: &FxHashMap<TermId, TermId>) {
        let Some(model) = self.model.as_mut() else {
            return;
        };
        for &array in &goal.arrays {
            if let Some(&value) = completion.get(&array) {
                model.set(array, value);
            }
        }
    }
}

impl Goal {
    /// Classify the assertion set, or decline.
    fn build(
        assertions: &[TermId],
        assignments: &FxHashMap<TermId, TermId>,
        manager: &TermManager,
    ) -> Option<Self> {
        let mut universals: Vec<TermId> = Vec::new();
        let mut seen_universals: FxHashSet<TermId> = FxHashSet::default();
        for &assertion in assertions {
            collect_maximal_quantifiers(assertion, manager, &mut universals, &mut seen_universals)?;
        }
        if universals.is_empty() {
            // Nothing under a binder, so this module has nothing to add over
            // the ordinary ground path.
            return None;
        }

        let mut arrays: Vec<TermId> = Vec::new();
        let mut scalar_pins: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        for &assertion in assertions {
            let mut free: Vec<TermId> = manager.free_vars_including_patterns(assertion);
            // `free_vars_including_patterns` answers from a set, so order it
            // by the id the term manager already assigned: the attempt's
            // combination order — and hence its verdict — must not depend on a
            // hash iteration order.
            free.sort_unstable_by_key(|term| term.raw());
            for var in free {
                if !seen.insert(var) {
                    continue;
                }
                let data = manager.get(var)?;
                if is_array_sort(data.sort, manager) {
                    arrays.push(var);
                    if arrays.len() > MAX_COMPLETED_ARRAYS {
                        return None;
                    }
                } else if let Some(&value) = assignments.get(&var) {
                    scalar_pins.insert(var, value);
                }
                // A scalar with no value in the candidate model is left free:
                // the certificate below is a *validity* query, so a symbol it
                // does not interpret can only make the query harder to
                // discharge, never the conclusion weaker.
            }
        }
        if arrays.is_empty() {
            return None;
        }
        Some(Self {
            assertions: assertions.to_vec(),
            universals,
            arrays,
            scalar_pins,
        })
    }

    /// The default values tried for each array, in a deterministic order.
    ///
    /// The pool is drawn from what the goal itself makes plausible — the
    /// element-sort values the candidate model already committed to, then the
    /// element-sort literals the script spells out, then the two ends of the
    /// sort — which is the same "values the goal makes plausible" rule
    /// [`crate::mbqi::model_certify`] uses for a function default.
    fn default_pools(
        &self,
        assignments: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> Vec<Vec<TermId>> {
        let mut literals: Vec<TermId> = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        for &assertion in &self.assertions {
            collect_constants(assertion, manager, &mut literals, &mut seen);
        }
        let mut model_values: Vec<TermId> = assignments.values().copied().collect();
        model_values.sort_unstable_by_key(|term| term.raw());
        model_values.dedup();

        self.arrays
            .iter()
            .map(|&array| {
                let element = element_sort(array, manager);
                let mut pool: Vec<TermId> = Vec::new();
                let mut pool_keys: FxHashSet<TermId> = FxHashSet::default();
                let push = |candidate: TermId,
                            pool: &mut Vec<TermId>,
                            keys: &mut FxHashSet<TermId>,
                            manager: &TermManager| {
                    if pool.len() >= MAX_DEFAULT_CANDIDATES {
                        return;
                    }
                    if !manager
                        .get(candidate)
                        .is_some_and(|d| Some(d.sort) == element)
                    {
                        return;
                    }
                    if keys.insert(candidate) {
                        pool.push(candidate);
                    }
                };
                for &value in &model_values {
                    push(value, &mut pool, &mut pool_keys, manager);
                }
                for &literal in &literals {
                    push(literal, &mut pool, &mut pool_keys, manager);
                }
                if let Some(element) = element {
                    for extreme in sort_extremes(element, manager) {
                        push(extreme, &mut pool, &mut pool_keys, manager);
                    }
                }
                pool
            })
            .collect()
    }

    /// The quantifier-free certificate for one completion.
    ///
    /// Every step is a validity query discharged by refuting its negation; see
    /// the module docs for why only that direction is taken.
    fn certificate_passes(
        &self,
        completion: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
        logic: Option<&str>,
        queries: &mut usize,
    ) -> bool {
        let mut substitution = self.scalar_pins.clone();
        substitution.extend(completion.iter().map(|(&k, &v)| (k, v)));

        // Step 1: every maximal `forall` must be certified *true* under the
        // completion.  The bound variables become fresh reserved constants, so
        // the query is quantifier-free and the bound name can never be
        // confused with a free one of the same name — which is exactly the
        // collision shape of `rk8/atk/f5_binder_collide_index.smt2`.
        let mut truths: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (position, &universal) in self.universals.iter().enumerate() {
            let Some(body) = peel_universal(universal, position, manager) else {
                return false;
            };
            let body = manager.substitute(body, &substitution);
            let negated = manager.mk_not(body);
            if !matches!(
                run_query(negated, manager, logic, queries),
                Some(SolverResult::Unsat)
            ) {
                return false;
            }
            truths.insert(universal, manager.mk_true());
        }

        // Step 2: one query for the assertions themselves.  `(or ¬A₁ … ¬Aₙ)`
        // is `Unsat` exactly when every `Aᵢ` holds under the completion.
        let mut disjuncts: Vec<TermId> = Vec::with_capacity(self.assertions.len());
        for &assertion in &self.assertions {
            let grounded = manager.substitute(assertion, &truths);
            let grounded = manager.substitute(grounded, &substitution);
            disjuncts.push(manager.mk_not(grounded));
        }
        let refutation = manager.mk_or(disjuncts);
        matches!(
            run_query(refutation, manager, logic, queries),
            Some(SolverResult::Unsat)
        )
    }
}

/// Run one quantifier-free certificate query, or `None` when the query budget
/// is spent.
///
/// The sub-solver is budgeted deterministically and carries no clock, so the
/// certificate reproduces on an idle and on a loaded machine alike.  It cannot
/// re-enter this module: the goal handed to it is quantifier-free by
/// construction, and the entry point above requires a quantifier.
fn run_query(
    goal: TermId,
    manager: &mut TermManager,
    logic: Option<&str>,
    queries: &mut usize,
) -> Option<SolverResult> {
    if *queries >= MAX_CERTIFICATE_QUERIES {
        return None;
    }
    *queries += 1;
    let mut solver = Solver::new();
    solver.set_logic(logic.unwrap_or("ALL"));
    solver.config.max_conflicts = CERTIFICATE_CONFLICT_BUDGET;
    solver.assert(goal, manager);
    Some(solver.check(manager))
}

/// The body of a chain of consecutive `forall`s, with every bound variable
/// replaced by a fresh reserved constant, or `None` when the sub-term is not a
/// universal this module handles.
///
/// A chain is peeled rather than a single binder, so
/// `(forall ((k …)) (forall ((k …)) …))` — decision (36)'s "the collision shape
/// with the array name bound **twice**" — is one multi-binder universal and not
/// a nested one.  A quantifier that survives the peel is a genuine
/// alternation and is declined, because the certificate below would not be
/// quantifier-free.
fn peel_universal(term: TermId, position: usize, manager: &mut TermManager) -> Option<TermId> {
    let mut current = term;
    let mut bound: Vec<(Spur, SortId)> = Vec::new();
    loop {
        let kind = manager.get(current).map(|d| d.kind.clone())?;
        match kind {
            TermKind::Forall { vars, body, .. } => {
                bound.extend(vars.iter().copied());
                current = body;
            }
            _ => break,
        }
    }
    if contains_quantifier(current, manager) {
        return None;
    }
    let mut rename: FxHashMap<TermId, TermId> = FxHashMap::default();
    for (index, (name, sort)) in bound.into_iter().enumerate() {
        let bound_name = manager.resolve_str(name).to_string();
        let bound_var = manager.mk_var(&bound_name, sort);
        let fresh = manager.mk_var(
            &reserved_name("qcert", &format!("{position}_{index}")),
            sort,
        );
        rename.insert(bound_var, fresh);
    }
    Some(manager.substitute(current, &rename))
}

/// Record every maximal quantifier sub-term of `term`, declining on an
/// `exists` (see the module docs) and on anything the walk cannot read.
fn collect_maximal_quantifiers(
    term: TermId,
    manager: &TermManager,
    out: &mut Vec<TermId>,
    seen: &mut FxHashSet<TermId>,
) -> Option<()> {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let data = manager.get(current)?;
        match &data.kind {
            TermKind::Forall { .. } => {
                if seen.insert(current) {
                    out.push(current);
                }
                // Deliberately not descended into: the sub-term is handled as
                // a whole by `peel_universal`.
            }
            TermKind::Exists { .. } => return None,
            _ => {
                children.clear();
                children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
                stack.extend(children.iter().copied());
            }
        }
    }
    // The walk visits a hash-consed DAG, so `stack` order decides the order
    // `out` is filled.  Sort it so the certificate's query order is a property
    // of the term ids and not of the traversal.
    out.sort_unstable_by_key(|term| term.raw());
    Some(())
}

/// Every constant (literal) sub-term of `term`, in first-encounter order.
fn collect_constants(
    term: TermId,
    manager: &TermManager,
    out: &mut Vec<TermId>,
    seen: &mut FxHashSet<TermId>,
) {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    let mut found: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        if matches!(
            data.kind,
            TermKind::BitVecConst { .. }
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::True
                | TermKind::False
        ) {
            found.push(current);
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    found.sort_unstable_by_key(|term| term.raw());
    for term in found {
        if seen.insert(term) {
            out.push(term);
        }
    }
}

/// Whether `term` carries a quantifier anywhere in it.
fn contains_quantifier(term: TermId, manager: &TermManager) -> bool {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        if matches!(data.kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
            return true;
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    false
}

/// Whether the goal is one an attempt can afford to try.
///
/// Two conditions, both about cost rather than about soundness.
///
/// * **Size.** See [`MAX_GOAL_NODES`].
/// * **No uninterpreted application.** An application of a symbol this module
///   does not interpret survives into the certificate's validity query, where
///   the sub-solver is free to interpret it any way it likes — so the query is
///   asking whether the goal holds for *every* interpretation of that symbol,
///   which a satisfiable-but-not-valid goal never does.  That is sound (it can
///   only decline) but it is certain to decline, so it is worth detecting up
///   front instead of after twenty-four combinations.  The array constant
///   `((as const A) d)` is an `Apply` under a reserved symbol and *is*
///   interpreted, so it does not count.
fn is_small_enough_to_certify(assertions: &[TermId], manager: &TermManager) -> bool {
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = assertions.to_vec();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        if visited.len() > MAX_GOAL_NODES {
            return false;
        }
        let Some(data) = manager.get(current) else {
            return false;
        };
        if matches!(data.kind, TermKind::Apply { .. }) && !is_const_array(current, manager) {
            return false;
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    true
}

/// Whether `sort` is an array sort.
fn is_array_sort(sort: SortId, manager: &TermManager) -> bool {
    manager
        .sorts
        .get(sort)
        .is_some_and(|s| matches!(s.kind, SortKind::Array { .. }))
}

/// The element sort of an array-sorted term.
fn element_sort(term: TermId, manager: &TermManager) -> Option<SortId> {
    let data = manager.get(term)?;
    match manager.sorts.get(data.sort)?.kind {
        SortKind::Array { range, .. } => Some(range),
        _ => None,
    }
}

/// The two ends of a sort, as a last resort for the default pool.
fn sort_extremes(sort: SortId, manager: &mut TermManager) -> Vec<TermId> {
    let kind = manager.sorts.get(sort).map(|s| s.kind.clone());
    match kind {
        Some(SortKind::BitVec(width)) => {
            let zero = manager.mk_bitvec(num_bigint::BigInt::from(0u8), width);
            let ones = manager.mk_bitvec(
                (num_bigint::BigInt::from(1u8) << width) - num_bigint::BigInt::from(1u8),
                width,
            );
            vec![zero, ones]
        }
        Some(SortKind::Bool) => vec![manager.mk_false(), manager.mk_true()],
        _ => Vec::new(),
    }
}

/// A bounded product over the per-array default pools.
struct Combinations<'a> {
    pools: &'a [Vec<TermId>],
    cursor: Vec<usize>,
    emitted: usize,
    done: bool,
}

impl<'a> Combinations<'a> {
    fn new(pools: &'a [Vec<TermId>]) -> Self {
        Self {
            pools,
            cursor: vec![0; pools.len()],
            emitted: 0,
            done: pools.is_empty(),
        }
    }
}

impl Iterator for Combinations<'_> {
    type Item = Vec<TermId>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.emitted >= MAX_COMBINATIONS {
            return None;
        }
        let mut item: Vec<TermId> = Vec::with_capacity(self.pools.len());
        for (pool, &index) in self.pools.iter().zip(self.cursor.iter()) {
            item.push(*pool.get(index)?);
        }
        self.emitted += 1;
        // Odometer, least significant position first.
        let mut position = self.cursor.len();
        loop {
            if position == 0 {
                self.done = true;
                break;
            }
            position -= 1;
            let limit = self.pools.get(position).map_or(0, Vec::len);
            let next = self.cursor.get_mut(position)?;
            *next += 1;
            if *next < limit {
                break;
            }
            *next = 0;
        }
        Some(item)
    }
}

/// The interpretation a completed array is published under: `((as const A) d)`.
pub(super) fn const_array(
    array_sort: SortId,
    default: TermId,
    manager: &mut TermManager,
) -> TermId {
    manager.mk_apply(CONST_ARRAY_FUNC, [default], array_sort)
}

/// Whether `term` is the array constant [`const_array`] builds.
///
/// The model formatter asks this before it prefers an explicit model entry
/// over the congruence class it would otherwise render an array from; see
/// `Context::array_model_value`.
pub(crate) fn is_const_array(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|data| match &data.kind {
        TermKind::Apply { func, args } => {
            args.len() == 1 && manager.resolve_str(*func) == CONST_ARRAY_FUNC
        }
        _ => false,
    })
}
