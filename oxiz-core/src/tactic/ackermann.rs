//! Ackermannization tactic for removing uninterpreted functions.

use super::core::*;
use crate::ast::{TermId, TermManager};
use crate::error::Result;
#[allow(unused_imports)]
use crate::prelude::*;

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

    /// Collect all function applications from a term.
    ///
    /// # Soundness: quantifier-bound arguments
    ///
    /// Ackermannization replaces every application `f(a)` by a *ground* fresh
    /// variable and adds ground functional-consistency constraints
    /// `a = b => v_a = v_b`. That is only sound when the arguments `a` are
    /// themselves ground: an application `f(x)` whose argument `x` is bound by
    /// an enclosing `forall`/`exists` denotes a *different* value for each
    /// binding of `x`, so collapsing it into one ground variable (and adding
    /// ground congruence constraints) is unsound.
    ///
    /// An application whose arguments reference any quantifier-bound variable
    /// is therefore **not** collected; instead its function symbol is recorded
    /// in `tainted`. The caller then drops *every* application of a tainted
    /// symbol — even its ground occurrences — because Ackermannizing only some
    /// occurrences of `f` while leaving quantified ones as real `f` would
    /// decouple the fresh variables from `f` (the linking constraint
    /// `v_a = f(a)` is absent), again unsoundly. If nothing survives, the
    /// tactic is `NotApplicable`.
    ///
    /// # Why `bound_names` is goal-global rather than scope-tracked
    ///
    /// This used to carry a `bound: &mut Vec<Spur>` push/truncate scope stack
    /// so that a name only counted as bound *within* its binder. Combined
    /// with the `visited` memo — which is keyed on `TermId` alone and shared
    /// across the whole goal — that was unsound: because terms are
    /// hash-consed, a subterm `f(x)` occurring both at the top level and
    /// inside `(forall ((x Int)) ...)` is one single `TermId`. Whichever
    /// occurrence was reached first won; if that was the top-level one, the
    /// walk recorded `f(x)` as a ground application and `visited` then
    /// suppressed the quantified occurrence that would have tainted `f`. The
    /// rewrite step replaces terms by `TermId`, so `f(x)` inside the binder
    /// was replaced by the ground fresh variable too — exactly the unsoundness
    /// the taint mechanism exists to prevent.
    ///
    /// `bound_names` is now the set of *every* name bound by *any*
    /// quantifier anywhere in the goal (see
    /// [`Self::collect_bound_names`]). That is a deliberate
    /// over-approximation: a symbol applied to a free variable that merely
    /// happens to share a name with some unrelated binder elsewhere is
    /// tainted and left alone. Over-tainting only makes the tactic decline to
    /// eliminate a symbol (incompleteness, reported honestly as
    /// `NotApplicable`), whereas under-tainting corrupts the formula — and it
    /// makes the scope-insensitive `visited` memo correct by construction,
    /// since "does this term reference a bound name" no longer depends on the
    /// path by which the term was reached.
    ///
    /// # Iterative walk
    ///
    /// The descent uses an explicit heap stack: the return type is `()`, so a
    /// depth cap could only silently stop collecting applications partway
    /// through — leaving some occurrences of `f` Ackermannized and others
    /// not, which is precisely the decoupling described above. Children come
    /// from [`crate::ast::traversal::get_children`], which matches `TermKind`
    /// exhaustively; the previous hand-written match ended in `_ => {}` and so
    /// never descended into `Str*`, `Fp*`, `DtConstructor`/`DtSelector`/
    /// `DtTester` or `Match` nodes, silently missing every application nested
    /// under one of those.
    ///
    /// Reference: Z3's `ackermannize_bv_tactic` / `ackr_helper` only collect
    /// ground applications.
    fn collect_func_apps(
        &self,
        term_id: TermId,
        bound_names: &crate::prelude::FxHashSet<crate::interner::Spur>,
        apps: &mut Vec<(
            crate::interner::Spur,
            smallvec::SmallVec<[TermId; 4]>,
            TermId,
        )>,
        tainted: &mut crate::prelude::FxHashSet<crate::interner::Spur>,
        visited: &mut crate::prelude::FxHashSet<TermId>,
    ) {
        use crate::ast::TermKind;
        use crate::ast::traversal::get_children;

        let mut stack = vec![term_id];
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            let Some(term) = self.manager.get(id) else {
                continue;
            };
            if let TermKind::Apply { func, args } = &term.kind {
                let refs_bound = !bound_names.is_empty()
                    && args
                        .iter()
                        .any(|&a| self.references_bound_var(a, bound_names));
                if refs_bound {
                    // Quantified occurrence: exclude this symbol entirely.
                    tainted.insert(*func);
                } else {
                    apps.push((*func, args.clone(), id));
                }
            }
            stack.extend(get_children(&term.kind));
        }
    }

    /// Every variable name bound by any `Forall`/`Exists` anywhere in
    /// `roots`' term DAGs.
    ///
    /// See [`Self::collect_func_apps`] for why the taint test uses this
    /// goal-global set instead of a lexical scope stack.
    fn collect_bound_names(
        &self,
        roots: &[TermId],
    ) -> crate::prelude::FxHashSet<crate::interner::Spur> {
        use crate::ast::TermKind;
        use crate::ast::traversal::get_children;

        let mut names = crate::prelude::FxHashSet::default();
        let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();
        let mut stack: Vec<TermId> = roots.to_vec();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            let Some(term) = self.manager.get(id) else {
                continue;
            };
            if let TermKind::Forall { vars, .. } | TermKind::Exists { vars, .. } = &term.kind {
                names.extend(vars.iter().map(|(name, _)| *name));
            }
            stack.extend(get_children(&term.kind));
        }
        names
    }

    /// Whether `term` references any variable name in `bound` (every name
    /// bound by any quantifier in the goal). Used to detect applications that
    /// depend on bound variables — see [`Self::collect_func_apps`].
    fn references_bound_var(
        &self,
        term: TermId,
        bound: &crate::prelude::FxHashSet<crate::interner::Spur>,
    ) -> bool {
        use crate::ast::TermKind;
        use crate::ast::traversal::collect_subterms;

        // `collect_subterms` walks the whole (hash-consed) subterm DAG once;
        // we then check whether any `Var` node's name is bound.
        for sub in collect_subterms(term, self.manager) {
            if let Some(t) = self.manager.get(sub)
                && let TermKind::Var(name) = &t.kind
                && bound.contains(name)
            {
                return true;
            }
        }
        false
    }

    /// Apply ackermannization to a goal.
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        Ok(self.apply_mut_with_converter(goal)?.0)
    }

    /// Apply ackermannization to a goal, additionally returning a
    /// [`ModelConverter`] that lifts a model of the transformed sub-goal back
    /// to a model over the original goal's variables (dropping the fresh
    /// Ackermann variables that are not part of the original signature).
    ///
    /// Returns `(TacticResult, None)` when the result is not a `SubGoals`
    /// transformation (nothing was eliminated), and `(SubGoals, Some(conv))`
    /// otherwise.
    pub fn apply_mut_with_converter(
        &mut self,
        goal: &Goal,
    ) -> Result<(TacticResult, Option<Box<dyn ModelConverter>>)> {
        use crate::prelude::{FxHashMap, FxHashSet};

        // Collect ground function applications, tracking symbols with any
        // quantifier-bound-argument occurrence (see `collect_func_apps`).
        let mut all_apps: Vec<(
            crate::interner::Spur,
            smallvec::SmallVec<[TermId; 4]>,
            TermId,
        )> = Vec::new();
        let mut tainted: FxHashSet<crate::interner::Spur> = FxHashSet::default();
        let bound_names = self.collect_bound_names(&goal.assertions);
        let mut visited = FxHashSet::default();

        for &assertion in &goal.assertions {
            self.collect_func_apps(
                assertion,
                &bound_names,
                &mut all_apps,
                &mut tainted,
                &mut visited,
            );
        }

        // Drop every application of a symbol that had a quantified
        // (bound-variable-dependent) occurrence — Ackermannizing it would be
        // unsound.
        if !tainted.is_empty() {
            all_apps.retain(|(func, _, _)| !tainted.contains(func));
        }

        // No (ground) function applications left to eliminate.
        if all_apps.is_empty() {
            return Ok((TacticResult::NotApplicable, None));
        }

        // Group applications by function symbol
        let mut func_groups: FxHashMap<crate::interner::Spur, Vec<FuncApp>> = FxHashMap::default();
        let mut term_to_var: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut fresh_vars: FxHashSet<TermId> = FxHashSet::default();

        for (var_counter, (func, args, term_id)) in all_apps.into_iter().enumerate() {
            // Create a fresh variable for this application
            let Some(term) = self.manager.get(term_id) else {
                continue; // Skip if term not found
            };
            let sort = term.sort;
            let var_name = format!("!ack_{}", var_counter);
            let fresh_var = self.manager.mk_var(&var_name, sort);

            term_to_var.insert(term_id, fresh_var);
            fresh_vars.insert(fresh_var);

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

        let converter: Box<dyn ModelConverter> = Box::new(AckermannModelConverter { fresh_vars });

        Ok((
            TacticResult::SubGoals(vec![Goal {
                assertions: new_assertions,
                precision: goal.precision,
            }]),
            Some(converter),
        ))
    }
}

/// Model converter for [`AckermannizeTactic`].
///
/// Ackermannization introduces fresh `!ack_k` variables that replace ground
/// function applications; these are *not* part of the original goal's
/// signature. Given a model of the transformed sub-goal, this converter
/// projects those fresh variables out, returning a model over the original
/// variables. (The eliminated function symbols' interpretations are
/// recoverable from the projected-out fresh-variable values — each fresh
/// variable is exactly the value of its application — but a function table is
/// not representable in the variable-only [`TacticModel`], so it is not
/// reconstructed here.)
#[derive(Debug, Clone)]
struct AckermannModelConverter {
    fresh_vars: crate::prelude::FxHashSet<TermId>,
}

impl ModelConverter for AckermannModelConverter {
    fn convert(&self, model: &TacticModel, _manager: &mut TermManager) -> TacticModel {
        let mut out = TacticModel::new();
        for (&var, &value) in &model.values {
            if !self.fresh_vars.contains(&var) {
                out.set(var, value);
            }
        }
        out
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessAckermannizeTactic;

impl Tactic for StatelessAckermannizeTactic {
    fn name(&self) -> &str {
        "ackermannize"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // Ackermannization must allocate fresh variables and congruence
        // constraints, which requires `&mut TermManager` access that the
        // manager-free `Tactic::apply` signature does not provide. Rather
        // than silently return the goal unchanged (which would dishonestly
        // claim a successful transformation), report that this path does not
        // apply. Callers holding a `&mut TermManager` should use
        // `AckermannizeTactic::apply_mut` (or the registry's `create_managed`
        // path) for the real transformation.
        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        "Eliminates uninterpreted functions by adding functional consistency constraints \
         (requires a TermManager; the manager-free path is NotApplicable)"
    }
}

#[cfg(test)]
mod group_c1_tests {
    use super::*;
    use crate::ast::{TermKind, TermManager};

    /// Regression: `collect_func_apps`' hand-written match ended in `_ => {}`,
    /// so it never descended into `Str*`/`Fp*`/`Dt*`/`Match` nodes. An
    /// application of `f` hidden under one of those was neither collected nor
    /// able to taint `f`, so the *other* occurrences of `f` were
    /// Ackermannized while that one stayed a real `f` -- exactly the
    /// decoupling `collect_func_apps`' own doc comment says is unsound.
    #[test]
    fn applications_under_a_datatype_constructor_are_seen() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let a = manager.mk_int(1);
        let f_a = manager.mk_apply("f", [a], int_sort);
        // Bury f(1) under a datatype constructor, which the old walk skipped.
        let wrapped = manager.mk_dt_constructor("mk", [f_a], int_sort);
        let b = manager.mk_int(2);
        let f_b = manager.mk_apply("f", [b], int_sort);
        let eq = manager.mk_eq(wrapped, f_b);

        let tactic = AckermannizeTactic::new(&mut manager);
        let bound = tactic.collect_bound_names(&[eq]);
        let mut apps = Vec::new();
        let mut tainted = crate::prelude::FxHashSet::default();
        let mut visited = crate::prelude::FxHashSet::default();
        tactic.collect_func_apps(eq, &bound, &mut apps, &mut tainted, &mut visited);

        assert_eq!(
            apps.len(),
            2,
            "both f-applications must be collected, including the one under \
             the constructor: {apps:?}"
        );
    }

    /// Regression: taint used to be computed against a lexical scope stack
    /// while `visited` was keyed on `TermId` alone. Because terms are
    /// hash-consed, `f(x)` occurring both free and under `forall x` is one
    /// `TermId`; whichever occurrence was reached first won, so the
    /// quantified one could be suppressed and `f` never tainted. Taint is now
    /// goal-global, which over-approximates safely.
    #[test]
    fn a_shared_application_under_a_binder_taints_its_symbol() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let zero = manager.mk_int(0);
        let free_use = manager.mk_eq(f_x, zero);
        let quantified_use = manager.mk_forall([("x", int_sort)], free_use);
        let p = manager.mk_apply("P", [x], bool_sort);
        let _ = p;

        // The free use is listed first so it is reached before the binder.
        let goal = Goal::new(vec![free_use, quantified_use]);
        let mut tactic = AckermannizeTactic::new(&mut manager);
        let result = tactic
            .apply_mut(&goal)
            .expect("test operation should succeed");

        assert!(
            matches!(result, TacticResult::NotApplicable),
            "f has a quantifier-bound-argument occurrence, so nothing may be \
             Ackermannized; got {result:?}"
        );
    }

    /// `collect_func_apps` walks with an explicit heap stack: a term far
    /// deeper than any native stack could hold must return rather than abort.
    #[test]
    fn collect_func_apps_survives_a_deep_chain_on_a_tiny_stack() {
        const DEPTH: usize = 7_500;

        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let mut chain = manager.mk_int(0);
                for _ in 0..DEPTH {
                    chain = manager.mk_apply("f", [chain], int_sort);
                }

                let tactic = AckermannizeTactic::new(&mut manager);
                let bound = tactic.collect_bound_names(&[chain]);
                let mut apps = Vec::new();
                let mut tainted = crate::prelude::FxHashSet::default();
                let mut visited = crate::prelude::FxHashSet::default();
                tactic.collect_func_apps(chain, &bound, &mut apps, &mut tainted, &mut visited);
                apps.len()
            })
            .expect("test thread must spawn");

        assert_eq!(handle.join().ok(), Some(DEPTH));
    }

    /// Sanity pin: a purely ground goal still Ackermannizes.
    #[test]
    fn ground_applications_are_still_eliminated() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let a = manager.mk_int(1);
        let b = manager.mk_int(2);
        let f_a = manager.mk_apply("f", [a], int_sort);
        let f_b = manager.mk_apply("f", [b], int_sort);
        let eq = manager.mk_eq(f_a, f_b);

        let goal = Goal::new(vec![eq]);
        let mut tactic = AckermannizeTactic::new(&mut manager);
        let result = tactic
            .apply_mut(&goal)
            .expect("test operation should succeed");
        let TacticResult::SubGoals(goals) = result else {
            panic!("expected ground applications to be eliminated, got {result:?}");
        };
        // No `Apply` node may survive in the transformed assertions.
        for &assertion in &goals[0].assertions {
            for sub in crate::ast::traversal::collect_subterms(assertion, tactic.manager) {
                assert!(
                    !matches!(
                        tactic.manager.get(sub).map(|t| &t.kind),
                        Some(TermKind::Apply { .. })
                    ),
                    "an uninterpreted application survived Ackermannization"
                );
            }
        }
    }
}
