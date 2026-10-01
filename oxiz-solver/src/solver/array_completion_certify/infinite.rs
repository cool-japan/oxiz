//! Completing an array over an **`Int` / `Real`** index: the candidate's own
//! points over a pooled default (`#P2b-51`, decision (54)(ii)).
//!
//! The pinned search (`pinned`) proposes points and values with one fill
//! query and is confined to finite index sorts, because over `Int` that
//! query cost seconds.  So an `Int`- or `Real`-indexed goal no *constant*
//! array satisfies — `∀i. a[i] ≥ 3` beside `a[5] = 4`, `a[6] = 3` (recheck
//! 13, `d01`) — was declined, and the `sat` published the candidate's reads
//! over the sort default `0`, false at every point no term names.
//!
//! The candidate already holds the values the ground assertions need at the
//! points they name; what it lacks is a default the universals accept.  So
//! this search keeps each array's candidate reads at literal indices as the
//! pinned points and tries defaults drawn from the script's constants — the
//! candidate's values of the element sort, the literals the assertions spell,
//! and each of those `± 1` — in a deterministic order.  Every proposal goes
//! through the same evaluation pre-filter and the same certificate as every
//! other search (over an infinite index the certificate splits the universal
//! into the named points and the unnamed region, `split`), so a wrong
//! proposal costs a certificate and never a verdict.

use num_bigint::BigInt;
use num_rational::Rational64;
use num_traits::{CheckedAdd, CheckedSub};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::{SortId, SortKind};
use rustc_hash::{FxHashMap, FxHashSet};

use super::{Combinations, Goal, collect_constants, const_array, element_sort, is_literal};
use crate::solver::Solver;

/// Defaults tried per array (the base pool and its `± 1` neighbours).
const MAX_POINT_DEFAULTS: usize = 10;

/// Candidate points one array may carry.
const MAX_CANDIDATE_POINTS: usize = 16;

/// Certificates the search may spend on top of the other searches'.
const MAX_POINT_QUERIES: usize = 32;

impl Solver {
    /// The search; `None` unless some array of `goal` has an `Int` / `Real`
    /// index sort and a proposal certifies.
    pub(super) fn search_candidate_point_completion(
        &self,
        goal: &Goal,
        assignments: &FxHashMap<TermId, TermId>,
        samples: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
        logic: Option<&str>,
    ) -> Option<FxHashMap<TermId, TermId>> {
        if !goal
            .arrays
            .iter()
            .any(|&array| index_is_infinite(array, manager))
        {
            return None;
        }
        // Only where every universal is certified pointwise (`split`).
        let arrays: FxHashSet<TermId> = goal.arrays.iter().copied().collect();
        for (position, &universal) in goal.universals.iter().enumerate() {
            if !super::split::applies_to(universal, position, &arrays, manager) {
                return None;
            }
        }
        let pools = goal.neighbour_pools(assignments, manager);
        if pools.iter().any(Vec::is_empty) {
            return None;
        }
        let mut entries: Vec<Vec<(TermId, TermId)>> = Vec::with_capacity(goal.arrays.len());
        for &array in &goal.arrays {
            entries.push(if index_is_infinite(array, manager) {
                candidate_points(array, assignments, &goal.scalar_pins, manager)
            } else {
                Vec::new()
            });
        }
        let mut queries = 0usize;
        for combination in Combinations::new(&pools) {
            let mut completion: FxHashMap<TermId, TermId> = FxHashMap::default();
            for ((&array, &default), points) in goal
                .arrays
                .iter()
                .zip(combination.iter())
                .zip(entries.iter())
            {
                let sort = manager.get(array).map(|d| d.sort)?;
                let mut interpretation = const_array(sort, default, manager);
                for &(index, value) in points {
                    if value != default {
                        interpretation = manager.mk_store(interpretation, index, value);
                    }
                }
                completion.insert(array, interpretation);
            }
            if self.completion_refuted_by_evaluation(goal, &completion, samples, manager) {
                continue;
            }
            if goal.certificate_passes(self, &completion, manager, logic, &mut queries) {
                return Some(completion);
            }
            if queries >= MAX_POINT_QUERIES {
                break;
            }
        }
        None
    }
}

impl Goal {
    /// Per array: the candidate's element-sort values and the assertions'
    /// literals (as `default_pools` orders them), then each of those `± 1`
    /// over a numeric element sort; at most [`MAX_POINT_DEFAULTS`].
    fn neighbour_pools(
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

        let mut pools: Vec<Vec<TermId>> = Vec::with_capacity(self.arrays.len());
        for &array in &self.arrays {
            let Some(element) = element_sort(array, manager) else {
                pools.push(Vec::new());
                continue;
            };
            let base: Vec<TermId> = model_values
                .iter()
                .chain(literals.iter())
                .copied()
                .filter(|&value| is_literal(value, manager))
                .filter(|&value| manager.get(value).is_some_and(|d| d.sort == element))
                .collect();
            let mut pool: Vec<TermId> = Vec::new();
            let push = |value: TermId, pool: &mut Vec<TermId>| {
                if pool.len() < MAX_POINT_DEFAULTS && !pool.contains(&value) {
                    pool.push(value);
                }
            };
            for &value in &base {
                push(value, &mut pool);
            }
            for &value in &base {
                for neighbour in neighbours(value, element, manager) {
                    push(neighbour, &mut pool);
                }
            }
            if pool.is_empty() {
                for extreme in super::sort_extremes(element, manager) {
                    push(extreme, &mut pool);
                }
            }
            pools.push(pool);
        }
        pools
    }
}

/// `value ± 1` for an `Int` / `Real` literal of `sort`; nothing otherwise.
fn neighbours(value: TermId, sort: SortId, manager: &mut TermManager) -> Vec<TermId> {
    let kind = manager.get(value).map(|d| d.kind.clone());
    let sort_kind = manager.sorts.get(sort).map(|s| s.kind.clone());
    match (kind, sort_kind) {
        (Some(TermKind::IntConst(n)), Some(SortKind::Int)) => {
            vec![
                manager.mk_int(&n + BigInt::from(1u8)),
                manager.mk_int(&n - BigInt::from(1u8)),
            ]
        }
        (Some(TermKind::RealConst(r)), Some(SortKind::Real)) => {
            let one = Rational64::from_integer(1);
            let mut out = Vec::new();
            if let Some(up) = r.checked_add(&one) {
                out.push(manager.mk_real(up));
            }
            if let Some(down) = r.checked_sub(&one) {
                out.push(manager.mk_real(down));
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Whether an array-sorted term's index sort is `Int` or `Real`.
fn index_is_infinite(array: TermId, manager: &TermManager) -> bool {
    let Some(sort) = manager.get(array).map(|d| d.sort) else {
        return false;
    };
    match manager.sorts.get(sort).map(|s| &s.kind) {
        Some(SortKind::Array { domain, .. }) => matches!(
            manager.sorts.get(*domain).map(|s| &s.kind),
            Some(SortKind::Int | SortKind::Real)
        ),
        _ => false,
    }
}

/// The candidate model's reads of `array` at literal indices (a literal, or a
/// scalar the candidate pins to one) with literal values, one per index in
/// term-id order, at most [`MAX_CANDIDATE_POINTS`].
fn candidate_points(
    array: TermId,
    assignments: &FxHashMap<TermId, TermId>,
    scalar_pins: &FxHashMap<TermId, TermId>,
    manager: &TermManager,
) -> Vec<(TermId, TermId)> {
    let mut reads: Vec<(TermId, TermId, TermId)> = Vec::new();
    for (&term, &value) in assignments {
        let Some(TermKind::Select(read_array, index)) = manager.get(term).map(|d| &d.kind) else {
            continue;
        };
        if *read_array != array || !is_literal(value, manager) {
            continue;
        }
        let index = if is_literal(*index, manager) {
            *index
        } else if let Some(&pinned) = scalar_pins.get(index).filter(|&&v| is_literal(v, manager)) {
            pinned
        } else {
            continue;
        };
        reads.push((term, index, value));
    }
    reads.sort_unstable_by_key(|&(term, _, _)| term.raw());
    let mut out: Vec<(TermId, TermId)> = Vec::new();
    for (_, index, value) in reads {
        if out.len() >= MAX_CANDIDATE_POINTS {
            break;
        }
        if out.iter().any(|&(named, _)| named == index) {
            continue;
        }
        out.push((index, value));
    }
    out
}
