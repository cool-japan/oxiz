//! Additional simplification infrastructure for tactic-driven preprocessing.

use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use smallvec::SmallVec;

/// Configuration for tactic-driven simplification passes.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimplificationConfig {
    /// Enable more expensive algebraic and Boolean rewrites.
    pub aggressive: bool,
}

/// Recursive simplifier that layers aggressive rewrites on top of `TermManager::simplify`.
pub struct AggressiveSimplifier<'a> {
    manager: &'a mut TermManager,
    config: SimplificationConfig,
}

impl<'a> AggressiveSimplifier<'a> {
    /// Create a new simplifier using the provided manager and configuration.
    pub fn new(manager: &'a mut TermManager, config: SimplificationConfig) -> Self {
        Self { manager, config }
    }

    /// Simplify a term recursively.
    pub fn simplify_term(&mut self, term: TermId) -> TermId {
        let mut cache = FxHashMap::default();
        self.simplify_term_cached(term, &mut cache)
    }

    fn simplify_term_cached(
        &mut self,
        term: TermId,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        if let Some(&cached) = cache.get(&term) {
            return cached;
        }

        let simplified = match self.manager.get(term).map(|t| t.kind.clone()) {
            None
            | Some(
                TermKind::True
                    | TermKind::False
                    | TermKind::IntConst(_)
                    | TermKind::RealConst(_)
                    | TermKind::BitVecConst { .. }
                    | TermKind::StringLit(_)
                    | TermKind::Var(_),
            ) => term,
            Some(TermKind::Not(arg)) => {
                let arg = self.simplify_term_cached(arg, cache);
                self.manager.mk_not(arg)
            }
            Some(TermKind::And(args)) => {
                let args = self.simplify_terms(args, cache);
                self.simplify_and(args)
            }
            Some(TermKind::Or(args)) => {
                let args = self.simplify_terms(args, cache);
                self.simplify_or(args)
            }
            Some(TermKind::Implies(lhs, rhs)) => {
                let lhs = self.simplify_term_cached(lhs, cache);
                let rhs = self.simplify_term_cached(rhs, cache);
                self.manager.mk_implies(lhs, rhs)
            }
            Some(TermKind::Eq(lhs, rhs)) => {
                let lhs = self.simplify_term_cached(lhs, cache);
                let rhs = self.simplify_term_cached(rhs, cache);
                self.simplify_eq(lhs, rhs)
            }
            Some(TermKind::Ite(cond, then_branch, else_branch)) => {
                let cond = self.simplify_term_cached(cond, cache);
                let then_branch = self.simplify_term_cached(then_branch, cache);
                let else_branch = self.simplify_term_cached(else_branch, cache);
                if then_branch == else_branch {
                    then_branch
                } else {
                    self.manager.mk_ite(cond, then_branch, else_branch)
                }
            }
            Some(TermKind::Distinct(args)) => {
                let args = self.simplify_terms(args, cache);
                self.simplify_distinct(args)
            }
            Some(TermKind::Add(args)) => {
                let args = self.simplify_terms(args, cache);
                let rebuilt = self.manager.mk_add(args);
                self.manager.simplify(rebuilt)
            }
            Some(TermKind::Sub(lhs, rhs)) => {
                let lhs = self.simplify_term_cached(lhs, cache);
                let rhs = self.simplify_term_cached(rhs, cache);
                let rebuilt = self.manager.mk_sub(lhs, rhs);
                self.manager.simplify(rebuilt)
            }
            Some(TermKind::Mul(args)) => {
                let args = self.simplify_terms(args, cache);
                let rebuilt = self.manager.mk_mul(args);
                self.manager.simplify(rebuilt)
            }
            Some(TermKind::Neg(arg)) => {
                let arg = self.simplify_term_cached(arg, cache);
                let rebuilt = self.manager.mk_neg(arg);
                self.manager.simplify(rebuilt)
            }
            Some(TermKind::Lt(lhs, rhs)) => {
                let lhs = self.simplify_term_cached(lhs, cache);
                let rhs = self.simplify_term_cached(rhs, cache);
                let rebuilt = self.manager.mk_lt(lhs, rhs);
                self.manager.simplify(rebuilt)
            }
            Some(TermKind::Le(lhs, rhs)) => {
                let lhs = self.simplify_term_cached(lhs, cache);
                let rhs = self.simplify_term_cached(rhs, cache);
                let rebuilt = self.manager.mk_le(lhs, rhs);
                self.manager.simplify(rebuilt)
            }
            Some(TermKind::Gt(lhs, rhs)) => {
                let lhs = self.simplify_term_cached(lhs, cache);
                let rhs = self.simplify_term_cached(rhs, cache);
                let rebuilt = self.manager.mk_gt(lhs, rhs);
                self.manager.simplify(rebuilt)
            }
            Some(TermKind::Ge(lhs, rhs)) => {
                let lhs = self.simplify_term_cached(lhs, cache);
                let rhs = self.simplify_term_cached(rhs, cache);
                let rebuilt = self.manager.mk_ge(lhs, rhs);
                self.manager.simplify(rebuilt)
            }
            Some(_) => self.manager.simplify(term),
        };

        cache.insert(term, simplified);
        simplified
    }

    fn simplify_terms(
        &mut self,
        args: SmallVec<[TermId; 4]>,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> SmallVec<[TermId; 4]> {
        args.into_iter()
            .map(|arg| self.simplify_term_cached(arg, cache))
            .collect()
    }

    fn simplify_and(&mut self, args: SmallVec<[TermId; 4]>) -> TermId {
        let baseline = self.manager.mk_and(args.clone());
        if !self.config.aggressive {
            return baseline;
        }

        if let Some(absorbed) = self.try_boolean_absorption_in_and(&args) {
            return self.manager.simplify(absorbed);
        }

        baseline
    }

    fn simplify_or(&mut self, args: SmallVec<[TermId; 4]>) -> TermId {
        let baseline = self.manager.mk_or(args.clone());
        if !self.config.aggressive {
            return baseline;
        }

        if let Some(absorbed) = self.try_boolean_absorption_in_or(&args) {
            return self.manager.simplify(absorbed);
        }

        if let Some(factored) = self.try_factor_or_of_ands(&args) {
            return self.manager.simplify(factored);
        }

        baseline
    }

    fn simplify_eq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let baseline = self.manager.mk_eq(lhs, rhs);
        if !self.config.aggressive {
            return baseline;
        }

        if let Some(rewritten) = self.try_solve_add_constant_eq(lhs, rhs) {
            return self.manager.simplify(rewritten);
        }
        if let Some(rewritten) = self.try_solve_add_constant_eq(rhs, lhs) {
            return self.manager.simplify(rewritten);
        }

        baseline
    }

    fn simplify_distinct(&mut self, args: SmallVec<[TermId; 4]>) -> TermId {
        let baseline = self.manager.mk_distinct(args.clone());
        if !self.config.aggressive {
            return baseline;
        }

        let mut seen = FxHashSet::default();
        for arg in args {
            if !seen.insert(arg) {
                return self.manager.mk_false();
            }
        }

        baseline
    }

    fn try_boolean_absorption_in_and(&mut self, args: &[TermId]) -> Option<TermId> {
        for &candidate in args {
            for &other in args {
                if candidate == other {
                    continue;
                }
                if let Some(term) = self.manager.get(other)
                    && let TermKind::Or(or_args) = &term.kind
                    && or_args.contains(&candidate)
                {
                    return Some(candidate);
                }
            }
        }
        None
    }

    fn try_boolean_absorption_in_or(&mut self, args: &[TermId]) -> Option<TermId> {
        for &candidate in args {
            for &other in args {
                if candidate == other {
                    continue;
                }
                if let Some(term) = self.manager.get(other)
                    && let TermKind::And(and_args) = &term.kind
                    && and_args.contains(&candidate)
                {
                    return Some(candidate);
                }
            }
        }
        None
    }

    fn try_factor_or_of_ands(&mut self, args: &[TermId]) -> Option<TermId> {
        for (left_idx, &left_term) in args.iter().enumerate() {
            let left_args = match self.manager.get(left_term).map(|term| &term.kind) {
                Some(TermKind::And(and_args)) => and_args.clone(),
                _ => continue,
            };

            for &right_term in &args[left_idx + 1..] {
                let right_args = match self.manager.get(right_term).map(|term| &term.kind) {
                    Some(TermKind::And(and_args)) => and_args.clone(),
                    _ => continue,
                };

                for &common in &left_args {
                    if right_args.contains(&common) {
                        let left_rest = without_one(&left_args, common);
                        let right_rest = without_one(&right_args, common);
                        let left_inner = self.mk_bool_join_or_true(left_rest, true);
                        let right_inner = self.mk_bool_join_or_true(right_rest, true);
                        let combined = self.manager.mk_or([left_inner, right_inner]);
                        return Some(self.manager.mk_and([common, combined]));
                    }
                }
            }
        }

        None
    }

    fn mk_bool_join_or_true(&mut self, args: SmallVec<[TermId; 4]>, as_and: bool) -> TermId {
        if args.is_empty() {
            self.manager.mk_true()
        } else if as_and {
            self.manager.mk_and(args)
        } else {
            self.manager.mk_or(args)
        }
    }

    fn try_solve_add_constant_eq(&mut self, add_side: TermId, const_side: TermId) -> Option<TermId> {
        let rhs_const = int_constant(self.manager, const_side)?;
        let add_args = match self.manager.get(add_side).map(|term| &term.kind) {
            Some(TermKind::Add(args)) => args.clone(),
            _ => return None,
        };

        let mut non_const = None;
        let mut constant_sum = BigInt::from(0_i32);
        for arg in add_args {
            if let Some(value) = int_constant(self.manager, arg) {
                constant_sum += value;
                continue;
            }
            if non_const.is_some() {
                return None;
            }
            non_const = Some(arg);
        }

        let lhs = non_const?;
        let rewritten_rhs = self.manager.mk_int(rhs_const - constant_sum);
        Some(self.manager.mk_eq(lhs, rewritten_rhs))
    }
}

fn int_constant(manager: &TermManager, term: TermId) -> Option<BigInt> {
    match manager.get(term).map(|t| &t.kind) {
        Some(TermKind::IntConst(value)) => Some(value.clone()),
        _ => None,
    }
}

fn without_one(args: &[TermId], needle: TermId) -> SmallVec<[TermId; 4]> {
    let mut removed = false;
    args.iter()
        .copied()
        .filter(|&arg| {
            if !removed && arg == needle {
                removed = true;
                false
            } else {
                true
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggressive_simplifier_handles_same_branch_ite() {
        let mut manager = TermManager::new();
        let cond = manager.mk_var("cond", manager.sorts.bool_sort);
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ite = manager.mk_ite(cond, x, x);

        let mut simplifier = AggressiveSimplifier::new(
            &mut manager,
            SimplificationConfig { aggressive: true },
        );
        let simplified = simplifier.simplify_term(ite);

        assert_eq!(simplified, x);
    }
}
