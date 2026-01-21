//! Ackermannization tactic for removing uninterpreted functions.

use super::core::*;
use crate::ast::{TermId, TermManager};
use crate::error::Result;

/// Ackermannization tactic - removes uninterpreted functions
pub struct AckermannizeTactic<'a> {
    manager: &'a mut TermManager,
}

/// A function application occurrence
#[derive(Debug, Clone)]
struct FuncApp {
    /// Fresh variable representing this application
    fresh_var: TermId,
    /// The arguments
    args: smallvec::SmallVec<[TermId; 4]>,
}

impl<'a> AckermannizeTactic<'a> {
    /// Create a new ackermannize tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Collect all function applications from a term
    fn collect_func_apps(
        &self,
        term_id: TermId,
        apps: &mut Vec<(lasso::Spur, smallvec::SmallVec<[TermId; 4]>, TermId)>,
        visited: &mut rustc_hash::FxHashSet<TermId>,
    ) {
        use crate::ast::TermKind;

        if visited.contains(&term_id) {
            return;
        }
        visited.insert(term_id);

        if let Some(term) = self.manager.get(term_id) {
            match &term.kind {
                TermKind::Apply { func, args } => {
                    apps.push((*func, args.clone(), term_id));
                    for &arg in args {
                        self.collect_func_apps(arg, apps, visited);
                    }
                }
                TermKind::Not(a) | TermKind::Neg(a) | TermKind::BvNot(a) => {
                    self.collect_func_apps(*a, apps, visited);
                }
                TermKind::BvExtract { arg, .. } => {
                    self.collect_func_apps(*arg, apps, visited);
                }
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::Distinct(args) => {
                    for &arg in args {
                        self.collect_func_apps(arg, apps, visited);
                    }
                }
                TermKind::Implies(a, b)
                | TermKind::Xor(a, b)
                | TermKind::Eq(a, b)
                | TermKind::Sub(a, b)
                | TermKind::Div(a, b)
                | TermKind::Mod(a, b)
                | TermKind::Lt(a, b)
                | TermKind::Le(a, b)
                | TermKind::Gt(a, b)
                | TermKind::Ge(a, b)
                | TermKind::Select(a, b)
                | TermKind::BvConcat(a, b)
                | TermKind::BvAnd(a, b)
                | TermKind::BvOr(a, b)
                | TermKind::BvXor(a, b)
                | TermKind::BvAdd(a, b)
                | TermKind::BvSub(a, b)
                | TermKind::BvMul(a, b)
                | TermKind::BvUdiv(a, b)
                | TermKind::BvSdiv(a, b)
                | TermKind::BvUrem(a, b)
                | TermKind::BvSrem(a, b)
                | TermKind::BvShl(a, b)
                | TermKind::BvLshr(a, b)
                | TermKind::BvAshr(a, b)
                | TermKind::BvUlt(a, b)
                | TermKind::BvUle(a, b)
                | TermKind::BvSlt(a, b)
                | TermKind::BvSle(a, b) => {
                    self.collect_func_apps(*a, apps, visited);
                    self.collect_func_apps(*b, apps, visited);
                }
                TermKind::Ite(c, t, e) | TermKind::Store(c, t, e) => {
                    self.collect_func_apps(*c, apps, visited);
                    self.collect_func_apps(*t, apps, visited);
                    self.collect_func_apps(*e, apps, visited);
                }
                TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => {
                    self.collect_func_apps(*body, apps, visited);
                }
                TermKind::Let { bindings, body } => {
                    for (_, t) in bindings {
                        self.collect_func_apps(*t, apps, visited);
                    }
                    self.collect_func_apps(*body, apps, visited);
                }
                // Constants and variables don't contain function applications
                _ => {}
            }
        }
    }

    /// Apply ackermannization to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        use rustc_hash::{FxHashMap, FxHashSet};

        // Collect all function applications
        let mut all_apps: Vec<(lasso::Spur, smallvec::SmallVec<[TermId; 4]>, TermId)> = Vec::new();
        let mut visited = FxHashSet::default();

        for &assertion in &goal.assertions {
            self.collect_func_apps(assertion, &mut all_apps, &mut visited);
        }

        // No function applications found
        if all_apps.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Group applications by function symbol
        let mut func_groups: FxHashMap<lasso::Spur, Vec<FuncApp>> = FxHashMap::default();
        let mut term_to_var: FxHashMap<TermId, TermId> = FxHashMap::default();

        for (var_counter, (func, args, term_id)) in all_apps.into_iter().enumerate() {
            // Create a fresh variable for this application
            let Some(term) = self.manager.get(term_id) else {
                continue; // Skip if term not found
            };
            let sort = term.sort;
            let var_name = format!("!ack_{}", var_counter);
            let fresh_var = self.manager.mk_var(&var_name, sort);

            term_to_var.insert(term_id, fresh_var);

            func_groups
                .entry(func)
                .or_default()
                .push(FuncApp { fresh_var, args });
        }

        // Generate functional consistency constraints
        // For each pair of applications of the same function:
        // (a1 = b1 ∧ ... ∧ an = bn) => (f(a) = f(b))
        let mut constraints: Vec<TermId> = Vec::new();

        for apps in func_groups.values() {
            for i in 0..apps.len() {
                for j in (i + 1)..apps.len() {
                    let app_i = &apps[i];
                    let app_j = &apps[j];

                    // Only compare if they have the same arity
                    if app_i.args.len() != app_j.args.len() {
                        continue;
                    }

                    // Build: (a1 = b1) ∧ (a2 = b2) ∧ ... => (var_i = var_j)
                    let mut arg_eqs: Vec<TermId> = Vec::new();
                    for k in 0..app_i.args.len() {
                        let eq = self.manager.mk_eq(app_i.args[k], app_j.args[k]);
                        arg_eqs.push(eq);
                    }

                    let antecedent = if arg_eqs.len() == 1 {
                        arg_eqs[0]
                    } else {
                        self.manager.mk_and(arg_eqs)
                    };

                    let consequent = self.manager.mk_eq(app_i.fresh_var, app_j.fresh_var);
                    let constraint = self.manager.mk_implies(antecedent, consequent);
                    constraints.push(constraint);
                }
            }
        }

        // Substitute function applications with their fresh variables in the goal
        let mut new_assertions: Vec<TermId> = Vec::new();

        for &assertion in &goal.assertions {
            let substituted = self.manager.substitute(assertion, &term_to_var);
            new_assertions.push(substituted);
        }

        // Add the functional consistency constraints
        new_assertions.extend(constraints);

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessAckermannizeTactic;

impl Tactic for StatelessAckermannizeTactic {
    fn name(&self) -> &str {
        "ackermannize"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Eliminates uninterpreted functions by adding functional consistency constraints"
    }
}
