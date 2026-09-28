//! Decision (41): a default **plus finitely many pinned points**.
//!
//! The first search of `array_completion_certify` interprets each array as a
//! constant.  That is the whole of `#P2b-58`'s family, and it is exactly what
//! a body declines when it forces *one* index to differ from all the others:
//!
//! ```text
//! (forall ((i (_ BitVec 7))) (= (select a i) (ite (= i #b0000000) #b0 #b1)))
//! ```
//!
//! is satisfied by `a = (store ((as const …) #b1) #b0000000 #b0)` and by no
//! constant array at all.  This search widens the interpretation to Ge & de
//! Moura's full *pins plus default* shape: a default overridden at finitely
//! many points.
//!
//! # Where the points come from
//!
//! The ground index terms the goal already names, and nothing else: every
//! literal of the index sort the assertions spell (a `select`/`store` index,
//! the constant an `ite` condition or an implication guard compares the bound
//! variable with) and the value the candidate model gives each index-sort
//! constant (`Goal::scalar_pins`).  They are ordered by value, so the
//! interpretation — and the model printed from it — is a property of the goal
//! and not of term-id allocation or of a hash iteration order.
//!
//! # Where the values come from, and why a wrong one costs nothing
//!
//! The default and the value at each point are fresh reserved constants, and
//! one quantifier-free **fill** query asks for values that satisfy the
//! assertions with every maximal `forall` replaced by its instances at the
//! points plus one representative point the pins do not name (the smallest
//! such value of the sort) — the instance at that representative is what
//! constrains the default.  A satisfying assignment is only a *proposal*: the
//! concrete interpretation built from it is evaluated on the same sample and
//! then goes through the very certificate the constant search uses — a
//! validity query per universal over *every* point of its sort, plus one over
//! the assertions — before anything is installed.  A proposal that is wrong
//! anywhere is refused there, so the fill query can make the search find an
//! interpretation, never make it accept a wrong one.  A point whose proposed
//! value equals the default is dropped, so the printed interpretation carries
//! no redundant `store`.

use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::smtlib::reserved_name;
use oxiz_core::sort::{SortId, SortKind};
use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    CERTIFICATE_CONFLICT_BUDGET, Combinations, Goal, const_array, is_literal,
    peel_universal_with_vars,
};
use crate::solver::{Solver, SolverResult};

/// Pinned points one array may carry.  Every script the decision names needs
/// one or two; the cap bounds the fill query's size on a goal that spells
/// many index literals.
const MAX_PINS_PER_SORT: usize = 16;

/// Quantifier-free queries (the fill query and the certificate together) the
/// pinned search may spend, on top of the constant search's own allowance.
const MAX_PINNED_QUERIES: usize = 16;

/// Instances of one universal the fill query may carry (a product over the
/// binder's variables).
const MAX_FILL_INSTANCES: usize = 64;

impl Solver {
    /// The pinned search: one fill query over a **symbolic** default and a
    /// symbolic value at every pinned point proposes an interpretation, the
    /// evaluation pre-filter and then the certificate decide.  `None` when the
    /// goal names no point, the fill query is not `Sat`, or the proposal does
    /// not certify.
    ///
    /// One proposal rather than one per pooled default: the default is what
    /// the interpretation reads at every point the pins do not name, and the
    /// fill query already constrains it through the universal's instance at
    /// the representative unnamed point, so a search over the pool would only
    /// repeat the query the symbolic default answers once.
    pub(super) fn search_pinned_completion(
        &self,
        goal: &Goal,
        samples: &FxHashMap<TermId, TermId>,
        points: &FxHashMap<SortId, Vec<TermId>>,
        manager: &mut TermManager,
        logic: Option<&str>,
    ) -> Option<FxHashMap<TermId, TermId>> {
        let mut array_pins: Vec<Vec<TermId>> = Vec::with_capacity(goal.arrays.len());
        for &array in &goal.arrays {
            let index = index_sort(array, manager)?;
            // Finite index sorts only.  Over `Int` the fill query carries a
            // store chain over a *symbolic* constant array on each side of an
            // array equality, and one such query costs seconds: measured on
            // `bench/z3_parity/benchmarks/AUFLIA/array_update.smt2`, 25 ms
            // without this search and 104 s with it, where the constant
            // search alone answers in 33 ms.  The family this search exists
            // for (decision (41)) is the bit-vector one.
            if !matches!(
                manager.sorts.get(index).map(|s| &s.kind),
                Some(SortKind::BitVec(_) | SortKind::Bool)
            ) {
                return None;
            }
            array_pins.push(points.get(&index).cloned().unwrap_or_default());
        }
        if array_pins.iter().all(Vec::is_empty) {
            return None;
        }
        let mut symbolic: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut defaults: Vec<TermId> = Vec::with_capacity(goal.arrays.len());
        let mut slots: Vec<(usize, TermId, TermId)> = Vec::new();
        for (position, &array) in goal.arrays.iter().enumerate() {
            let sort = manager.get(array)?.sort;
            let element = super::element_sort(array, manager)?;
            let default = manager.mk_var(&reserved_name("qdef", &position.to_string()), element);
            let mut interpretation = const_array(sort, default, manager);
            for (pin, &index) in array_pins.get(position)?.iter().enumerate() {
                let symbol = manager.mk_var(
                    &reserved_name("qpin", &format!("{position}_{pin}")),
                    element,
                );
                interpretation = manager.mk_store(interpretation, index, symbol);
                slots.push((position, index, symbol));
            }
            symbolic.insert(array, interpretation);
            defaults.push(default);
        }
        let fill = goal.fill_query(&symbolic, samples, manager);
        let mut symbols = defaults.clone();
        symbols.extend(slots.iter().map(|&(_, _, symbol)| symbol));
        let mut queries = 0usize;
        let values = run_fill_query(fill, &symbols, manager, logic, &mut queries)?;
        let (default_values, pin_values) = values.split_at(defaults.len());

        let mut completion: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (position, &array) in goal.arrays.iter().enumerate() {
            let sort = manager.get(array)?.sort;
            // A default the fill model leaves unassigned is unconstrained by
            // every instance; any literal of the element sort will do, and the
            // certificate checks the one taken.
            let default = match default_values.get(position).copied().flatten() {
                Some(value) => value,
                None => {
                    let element = super::element_sort(array, manager)?;
                    super::sort_extremes(element, manager).into_iter().next()?
                }
            };
            let mut interpretation = const_array(sort, default, manager);
            for (&(owner, index, _), value) in slots.iter().zip(pin_values.iter()) {
                if owner != position {
                    continue;
                }
                if let Some(value) = *value
                    && value != default
                {
                    interpretation = manager.mk_store(interpretation, index, value);
                }
            }
            completion.insert(array, interpretation);
        }
        if self.completion_refuted_by_evaluation(goal, &completion, samples, manager) {
            return None;
        }
        goal.certificate_passes(&completion, manager, logic, &mut queries)
            .then_some(completion)
    }
}

impl Goal {
    /// Candidate points per sort: every literal of that sort the assertions
    /// spell, plus the candidate model's value of every constant of that sort,
    /// ordered by value and capped.
    pub(super) fn points_by_sort(&self, manager: &TermManager) -> FxHashMap<SortId, Vec<TermId>> {
        let mut found: Vec<TermId> = Vec::new();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = self.assertions.clone();
        let mut children: Vec<TermId> = Vec::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = manager.get(current) else {
                continue;
            };
            if is_literal(current, manager) {
                found.push(current);
            }
            children.clear();
            children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
            stack.extend(children.iter().copied());
        }
        found.extend(
            self.scalar_pins
                .values()
                .copied()
                .filter(|&v| is_literal(v, manager)),
        );

        let mut by_sort: FxHashMap<SortId, Vec<TermId>> = FxHashMap::default();
        for term in found {
            let Some(sort) = manager.get(term).map(|d| d.sort) else {
                continue;
            };
            by_sort.entry(sort).or_default().push(term);
        }
        for points in by_sort.values_mut() {
            points.sort_by_key(|point| literal_key(*point, manager));
            points.dedup();
            points.truncate(MAX_PINS_PER_SORT);
        }
        by_sort
    }

    /// Every maximal universal of the goal, mapped to the conjunction of its
    /// body's instances at the candidate points of each bound variable's sort
    /// and at one representative point those points do not name.  Shared by
    /// the fill query and by the evaluation pre-filter, so both look at the
    /// same finite sample.  `None` when a universal is not one the certificate
    /// could discharge anyway (an alternation survives the peel).
    pub(super) fn sample_instances(
        &self,
        points: &FxHashMap<SortId, Vec<TermId>>,
        manager: &mut TermManager,
    ) -> Option<FxHashMap<TermId, TermId>> {
        let mut instances: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (position, &universal) in self.universals.iter().enumerate() {
            let (body, vars) = peel_universal_with_vars(universal, position, manager)?;
            let mut lists: Vec<Vec<TermId>> = Vec::with_capacity(vars.len());
            for &var in &vars {
                let sort = manager.get(var)?.sort;
                let mut list = points.get(&sort).cloned().unwrap_or_default();
                if let Some(representative) = unnamed_point(sort, &list, manager) {
                    list.push(representative);
                }
                if list.is_empty() {
                    return None;
                }
                lists.push(list);
            }
            let mut conjuncts: Vec<TermId> = Vec::new();
            for combination in Combinations::with_cap(&lists, MAX_FILL_INSTANCES) {
                let map: FxHashMap<TermId, TermId> = vars
                    .iter()
                    .copied()
                    .zip(combination.iter().copied())
                    .collect();
                conjuncts.push(manager.substitute(body, &map));
            }
            instances.insert(universal, manager.mk_and(conjuncts));
        }
        Some(instances)
    }

    /// The fill query: the assertions under the symbolic interpretation, with
    /// every maximal universal replaced by its sampled instances.
    fn fill_query(
        &self,
        symbolic: &FxHashMap<TermId, TermId>,
        samples: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        let mut substitution = self.scalar_pins.clone();
        substitution.extend(symbolic.iter().map(|(&k, &v)| (k, v)));
        let mut parts: Vec<TermId> = Vec::with_capacity(self.assertions.len());
        for &assertion in &self.assertions {
            let instantiated = manager.substitute(assertion, samples);
            parts.push(manager.substitute(instantiated, &substitution));
        }
        manager.mk_and(parts)
    }
}

/// Run the fill query and read back the value it proposes for each symbol
/// (`None` where the model gives it no literal value — that point then takes
/// the default).  `None` overall when the query is not `Sat`.
fn run_fill_query(
    goal: TermId,
    symbols: &[TermId],
    manager: &mut TermManager,
    logic: Option<&str>,
    queries: &mut usize,
) -> Option<Vec<Option<TermId>>> {
    if *queries >= MAX_PINNED_QUERIES {
        return None;
    }
    *queries += 1;
    let mut solver = Solver::new();
    solver.set_logic(logic.unwrap_or("ALL"));
    solver.config.max_conflicts = CERTIFICATE_CONFLICT_BUDGET;
    solver.assert(goal, manager);
    if solver.check(manager) != SolverResult::Sat {
        return None;
    }
    let model = solver.model()?;
    Some(
        symbols
            .iter()
            .map(|&symbol| {
                model
                    .get(symbol)
                    .filter(|&value| is_literal(value, manager))
            })
            .collect(),
    )
}

/// The index sort of an array-sorted term.
fn index_sort(term: TermId, manager: &TermManager) -> Option<SortId> {
    let data = manager.get(term)?;
    match manager.sorts.get(data.sort)?.kind {
        SortKind::Array { domain, .. } => Some(domain),
        _ => None,
    }
}

/// The smallest value of `sort` that `named` does not contain — the
/// representative of the region every pinned interpretation reads as its
/// default — or `None` when the sort has no such value or is not one this
/// search enumerates.
fn unnamed_point(sort: SortId, named: &[TermId], manager: &mut TermManager) -> Option<TermId> {
    let kind = manager.sorts.get(sort).map(|s| s.kind.clone())?;
    let taken: FxHashSet<(u8, BigInt)> = named
        .iter()
        .filter_map(|&t| literal_key(t, manager))
        .collect();
    match kind {
        SortKind::BitVec(width) => {
            let limit = BigInt::from(1u8) << width;
            let mut value = BigInt::from(0u8);
            while value < limit {
                if !taken.contains(&(0, value.clone())) {
                    return Some(manager.mk_bitvec(value, width));
                }
                value += 1u8;
            }
            None
        }
        SortKind::Int => {
            let mut value = BigInt::from(0u8);
            loop {
                if !taken.contains(&(1, value.clone())) {
                    return Some(manager.mk_int(value));
                }
                value += 1u8;
            }
        }
        SortKind::Bool => {
            if !taken.contains(&(2, BigInt::from(0u8))) {
                Some(manager.mk_false())
            } else if !taken.contains(&(2, BigInt::from(1u8))) {
                Some(manager.mk_true())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// A total order on literals by value (bit-vectors, then integers, then
/// Booleans), so the points — and the store chain printed from them — do not
/// depend on the order terms were interned in.  Reals and anything else sort
/// last, by term id.
fn literal_key(term: TermId, manager: &TermManager) -> Option<(u8, BigInt)> {
    match &manager.get(term)?.kind {
        TermKind::BitVecConst { value, .. } => Some((0, value.clone())),
        TermKind::IntConst(value) => Some((1, value.clone())),
        TermKind::False => Some((2, BigInt::from(0u8))),
        TermKind::True => Some((2, BigInt::from(1u8))),
        _ => Some((3, BigInt::from(term.raw()))),
    }
}
