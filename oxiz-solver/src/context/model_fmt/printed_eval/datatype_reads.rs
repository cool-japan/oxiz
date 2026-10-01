//! Reads of a datatype-indexed array, folded over the printed store chain.
//!
//! The evaluator the query-free certificate runs on
//! (`solver::array_completion_certify::evaluate_closed`) has no value for a
//! datatype term, so a `select` whose index is a constructor value — `(select
//! e w)` with `w` printed as `(mk 0 red)` and `e` as a store chain keyed by
//! constructor values — was never folded, and a goal reading one (a negated
//! universal over a vacuous binder, `recheck14/corpus/g14a/s01255`) stayed
//! undecided: its correct model was withheld.
//!
//! Here such a read is folded before evaluation: the store chain is walked
//! from the outermost `store` in, each key compared with the index as VALUES
//! — constructors by name and then field by field, every other field by its
//! exact evaluation — so the first key equal to the index gives the read, a
//! key that differs is passed, and the constant array at the base gives its
//! default.  A comparison the values cannot decide (a field that does not
//! evaluate, an index that is not a constructor value) leaves the read as it
//! was.  Sound for the reason every evaluation here is: the model is the
//! printed one, and a store chain's reading is decided by value equality of
//! its keys.

use super::*;
use crate::prelude::{FxHashMap, FxHashSet};

/// Rewriting passes one term may take (a read whose array is itself read out
/// of another array needs one pass per level).
const MAX_FOLD_PASSES: usize = 8;

impl Context {
    /// `term` with every closed read of a datatype-indexed array replaced by
    /// the value its store chain gives it (see the module docs).
    pub(in crate::context) fn fold_datatype_reads(&mut self, term: TermId) -> TermId {
        let mut current = term;
        for _ in 0..MAX_FOLD_PASSES {
            let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
            let mut stack: Vec<TermId> = vec![current];
            let mut visited: FxHashSet<TermId> = FxHashSet::default();
            while let Some(node) = stack.pop() {
                if !visited.insert(node) {
                    continue;
                }
                let Some(data) = self.terms.get(node).cloned() else {
                    continue;
                };
                if let TermKind::Select(array, index) = data.kind
                    && self.is_datatype_term(index)
                    && let Some(value) = self.read_over_value_keys(array, index)
                {
                    map.insert(node, value);
                    continue;
                }
                stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
            }
            if map.is_empty() {
                break;
            }
            current = self.terms.substitute(current, &map);
        }
        current
    }

    /// Whether `term`'s sort is a datatype.
    fn is_datatype_term(&self, term: TermId) -> bool {
        self.terms
            .get(term)
            .and_then(|data| self.terms.sorts.get(data.sort))
            .is_some_and(|sort| matches!(sort.kind, SortKind::Datatype(_)))
    }

    /// The value `(select array index)` reads, where every key of `array`'s
    /// store chain the walk passes compares with `index` by value.
    fn read_over_value_keys(&mut self, array: TermId, index: TermId) -> Option<TermId> {
        let mut level = array;
        loop {
            match self.terms.get(level)?.kind.clone() {
                TermKind::Store(inner, key, value) => {
                    if self.values_equal(key, index)? {
                        return Some(value);
                    }
                    level = inner;
                }
                TermKind::Apply { func, args }
                    if args.len() == 1
                        && self.terms.resolve_str(func) == oxiz_core::smtlib::CONST_ARRAY_FUNC =>
                {
                    return args.first().copied();
                }
                _ => return None,
            }
        }
    }

    /// Whether two closed terms denote one value: `Some(true)` / `Some(false)`
    /// where the values decide it, `None` otherwise.  Constructor terms
    /// compare by constructor and then field by field (a worklist, so a long
    /// list costs no stack); any other pair by exact evaluation.
    fn values_equal(&mut self, left: TermId, right: TermId) -> Option<bool> {
        let mut pairs: Vec<(TermId, TermId)> = vec![(left, right)];
        let mut open = false;
        while let Some((a, b)) = pairs.pop() {
            if a == b {
                continue;
            }
            let kind_a = self.terms.get(a)?.kind.clone();
            let kind_b = self.terms.get(b)?.kind.clone();
            match (kind_a, kind_b) {
                (
                    TermKind::DtConstructor {
                        constructor: name_a,
                        args: fields_a,
                    },
                    TermKind::DtConstructor {
                        constructor: name_b,
                        args: fields_b,
                    },
                ) => {
                    if name_a != name_b {
                        return Some(false);
                    }
                    if fields_a.len() != fields_b.len() {
                        open = true;
                        continue;
                    }
                    pairs.extend(fields_a.iter().copied().zip(fields_b.iter().copied()));
                }
                (TermKind::DtConstructor { .. }, _) | (_, TermKind::DtConstructor { .. }) => {
                    open = true;
                }
                _ => {
                    if self.is_datatype_term(a)
                        || !self.terms.free_vars_including_patterns(a).is_empty()
                        || !self.terms.free_vars_including_patterns(b).is_empty()
                    {
                        open = true;
                        continue;
                    }
                    let value_a = crate::solver::array_completion_certify::evaluate_closed_value(
                        &self.solver,
                        a,
                        &mut self.terms,
                    );
                    let value_b = crate::solver::array_completion_certify::evaluate_closed_value(
                        &self.solver,
                        b,
                        &mut self.terms,
                    );
                    match (value_a, value_b) {
                        (Some(x), Some(y)) if x != y => return Some(false),
                        (Some(_), Some(_)) => {}
                        _ => open = true,
                    }
                }
            }
        }
        (!open).then_some(true)
    }
}
