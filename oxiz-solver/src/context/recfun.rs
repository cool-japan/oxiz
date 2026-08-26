//! `define-fun-rec` / `define-funs-rec`: registry plus the fuel-bounded
//! unfolding driver that discharges the definitional axiom.
//!
//! # What a recursive definition means here
//!
//! `(define-fun-rec f ((x S)) R body)` asserts the axiom
//! `forall x. f(x) = body`. Nothing else constrains `f`: the parser
//! deliberately registers it as an ordinary declared symbol so that call sites
//! build real `Apply` nodes, and it never expands the body at those call sites
//! (expansion would not terminate). Discharging the axiom is this module's job.
//!
//! # The verdict contract (read this before changing anything below)
//!
//! The driver instantiates the axiom at finitely many argument tuples and hands
//! the instances to the ordinary solver. Instances are *consequences* of the
//! axiom, so the instantiated problem is a **relaxation** of the real one:
//!
//! * **`unsat` is returned immediately and is final.** A refutation that uses
//!   only consequences of the axiom refutes the original problem too.
//! * **`sat` is returned only with a certificate.** The relaxation can be
//!   satisfiable when the real problem is not (the solver is free to invent a
//!   value for an `f`-application whose defining instance was never
//!   instantiated), so a bare `sat` is never published. Two certificates are
//!   accepted:
//!   - **C1, saturation** — the unfolding closed: every `f`-application
//!     reachable from the assertions got its instance, so no application the
//!     assertions can talk about is left free.
//!
//!     Note precisely what this does *not* say. The axiom quantifies over all
//!     arguments, and closure only covers the reachable ones, so a definition
//!     that is unsatisfiable at some *unreachable* argument is not caught:
//!     `f(n) = ite(n = 0, 0, f(n) + 1)` with only `(= (f 0) 0)` asserted
//!     saturates at `f(0) = 0` and is reported `sat`, even though the axiom has
//!     no model at any `n /= 0`. Z3 4.15.4 answers `sat` here too — this is the
//!     shared reading of `define-fun-rec` as a definition constraining the
//!     applications a problem actually makes, not a standalone theory.
//!   - **C2, model re-computation** — a ground evaluator recomputes every
//!     recursive application occurring in the user's assertions directly from
//!     the definitions and finds that the model already agrees with it (see
//!     [`eval`]).
//! * **Everything else is `unknown`, and it is reached in finite time.** Both
//!   budgets are finite — the fuel schedule bounds the number of instances per
//!   round and the number of rounds, and the certifier's evaluation budget
//!   bounds the re-computation — so a definition that does not terminate is
//!   answered `unknown`, **never** by hanging. An *inconsistent* definition
//!   such as `f(x) = f(x) + 1` is still correctly `unsat`, because its single
//!   instance is already contradictory.
//!
//! The one thing that must never happen is a bare `sat`/`unsat` derived from a
//! problem where the definition was dropped: that silently answers a strictly
//! weaker question.
//!
//! # Where the instances live
//!
//! Instances are asserted **directly on the solver**, inside a scratch scope,
//! and never into `Context::assertions` — `(get-assertions)` must keep
//! reporting exactly the script's own assertions, byte for byte. The scratch
//! scope is deliberately left *open* on an accepted verdict so `(get-model)` /
//! `(get-value ..)` read the model that the instances helped build; it is
//! discharged by [`Context::invalidate_last_check`] (which every
//! `assert`/`push`/`pop` calls before touching the solver's scope stack) and at
//! the top of the driver itself.

mod eval;

#[allow(unused_imports)]
use crate::prelude::*;
use crate::solver::SolverResult;
use oxiz_core::ast::traversal::get_children;
use oxiz_core::ast::{TermId, TermKind};
use oxiz_core::error::Result;
use oxiz_core::smtlib::RecFunDecl;
use oxiz_core::sort::SortId;

use super::Context;

/// Instance budgets, tried in order. Each round re-runs the whole solve with a
/// larger unfolding; the schedule is finite, which is half of why the driver
/// always terminates.
///
/// It stops at 32 on purpose. Structural unfolding of a *symbolic* argument
/// stacks one `ite` per level, and the solve cost grows exponentially with the
/// stack — measured on `(and (>= k 0) (= (sum k) 6))` as 9 ms / 13 ms / 215 ms
/// / 3.6 s / 84 s for 4 / 8 / 16 / 32 / 64 instances. Beyond this point deeper
/// unfolding is not a way to reach an answer, it is a way to reach the
/// timeout; the refinement loop's learned ground instances are what actually
/// converge those queries.
const FUEL_SCHEDULE: &[usize] = &[4, 8, 16, 32];

/// Solver calls the driver may make for one `check-sat`.
const MAX_ROUNDS: usize = 16;

/// Ceiling on the concrete applications carried between refinement rounds.
const MAX_LEARNED_APPS: usize = 256;

/// Hard ceiling on the instances produced in a single unfolding round,
/// independent of the fuel. Reaching it marks the round *truncated*, which
/// disables the saturation certificate — a worklist drained by truncation is
/// not a worklist that closed.
const MAX_INSTANCES: usize = 100_000;

/// One recursive definition in scope.
#[derive(Debug, Clone)]
pub(super) struct RecDef {
    /// The defined function's name.
    name: String,
    /// `(name, sort-string)` per formal parameter, for `(get-model)` output.
    params: Vec<(String, String)>,
    /// The declared return sort, stringified, for `(get-model)` output.
    ret_sort_name: String,
    /// Each formal parameter's interned `Var`, in declaration order — the only
    /// sound substitution keys for `body` (see `RecFunDecl::formal_vars`).
    formal_vars: Vec<TermId>,
    /// The definition's right-hand side.
    body: TermId,
    /// The resolved return sort, used to rebuild application keys.
    ret_sort: SortId,
}

/// The recursive definitions in scope, plus the scratch-scope bookkeeping.
#[derive(Debug, Default)]
pub(super) struct RecFunState {
    /// Definitions in declaration order.
    defs: Vec<RecDef>,
    /// Name -> index into `defs`.
    name_to_index: crate::prelude::HashMap<String, usize>,
    /// `defs.len()` at each [`Context::push`], so [`Context::pop`] can retract
    /// the definitions introduced inside the scope.
    scope_stack: Vec<usize>,
    /// Whether a scratch solver scope opened by the driver is still open and
    /// owes a `solver.pop()`.
    pending_pop: bool,
}

impl RecFunState {
    /// Whether any recursive definition is in scope.
    pub(super) fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Record the definition count at a `push`.
    pub(super) fn push_scope(&mut self) {
        self.scope_stack.push(self.defs.len());
    }

    /// Retract the definitions introduced since the matching `push`.
    pub(super) fn pop_scope(&mut self) {
        if let Some(len) = self.scope_stack.pop() {
            self.truncate_defs(len);
        }
    }

    /// Drop every definition made inside a `push` (used by `reset-assertions`,
    /// which pops all levels but keeps top-level definitions), and forget the
    /// scratch scope without popping it — the caller resets the solver, which
    /// empties the scope stack outright.
    pub(super) fn retract_to_base(&mut self) {
        let base = self.scope_stack.first().copied().unwrap_or(self.defs.len());
        self.truncate_defs(base);
        self.scope_stack.clear();
        self.pending_pop = false;
    }

    /// Forget every definition and the scratch scope (used by `reset`).
    pub(super) fn clear_all(&mut self) {
        self.defs.clear();
        self.name_to_index.clear();
        self.scope_stack.clear();
        self.pending_pop = false;
    }

    /// Shrink `defs` to `len`, keeping `name_to_index` consistent.
    fn truncate_defs(&mut self, len: usize) {
        while self.defs.len() > len {
            if let Some(def) = self.defs.pop() {
                // Only drop the index entry if it still points at the popped
                // definition: a redefinition inside the scope may have
                // overwritten an outer entry, and that outer definition is
                // still live.
                if self.name_to_index.get(&def.name) == Some(&self.defs.len()) {
                    self.name_to_index.remove(&def.name);
                }
            }
        }
        // Rebuild the shadowed entries an inner redefinition may have hidden.
        for (index, def) in self.defs.iter().enumerate() {
            self.name_to_index.insert(def.name.clone(), index);
        }
    }

    /// The definition applied by `name`, if any.
    #[cfg(test)]
    fn lookup(&self, name: &str) -> Option<&RecDef> {
        self.name_to_index.get(name).and_then(|&i| self.defs.get(i))
    }
}

/// What certificate C2 concluded about a candidate model.
pub(super) enum Certification {
    /// The model agrees with the definitions everywhere it had to.
    Certified,
    /// The model disagrees with the definitions. `applications` are the
    /// concrete applications whose true values the certifier established on the
    /// way; their defining instances are consequences of the axiom that refute
    /// this model.
    Refuted {
        /// Concrete applications with a known true value.
        applications: Vec<TermId>,
    },
    /// The re-computation could not be completed (divergence, or a budget ran
    /// out), so nothing was learned about the model either way.
    Inconclusive,
}

/// The outcome of one unfolding round.
struct Unfolding {
    /// The instantiated definitional equations, ready to assert.
    instances: Vec<TermId>,
    /// Applications discovered but never instantiated. Empty *and* not
    /// truncated means the unfolding closed (certificate C1).
    boundary: Vec<TermId>,
    /// Whether [`MAX_INSTANCES`] cut the round short. A truncated round may
    /// leave an empty boundary purely because the worklist was abandoned, so it
    /// must never be read as saturation.
    truncated: bool,
}

impl Context {
    /// Register a `define-fun-rec` / `define-funs-rec` group.
    ///
    /// Each function is also declared the way `declare-fun` / `declare-const`
    /// would declare it, so introspection (`get_fun_signature`,
    /// `declared_function_names`, `(get-model)`) sees the same symbol the
    /// parser resolved call sites against.
    ///
    /// # Errors
    ///
    /// Returns an error when a parameter or return sort expression does not
    /// resolve.
    pub(super) fn define_funs_rec(&mut self, decls: Vec<RecFunDecl>) -> Result<()> {
        // A new definition supersedes any cached verdict, and discharges the
        // scratch scope before anything else touches the solver.
        self.invalidate_last_check();
        for decl in decls {
            let ret_sort = self.parse_sort_name(&decl.ret_sort)?;
            let arg_sorts: Vec<SortId> = decl
                .params
                .iter()
                .map(|(_, sort_name)| self.parse_sort_name(sort_name))
                .collect::<Result<_>>()?;
            if arg_sorts.is_empty() {
                // A nullary recursive definition resolves as a `Var` at its
                // call sites (the parser put it in `constants`), so it must be
                // declared as a constant here for the terms to coincide.
                self.declare_const(&decl.name, ret_sort);
            } else {
                self.declare_fun(&decl.name, arg_sorts, ret_sort);
            }
            let index = self.recfun.defs.len();
            self.recfun.defs.push(RecDef {
                name: decl.name.clone(),
                params: decl.params,
                ret_sort_name: decl.ret_sort,
                formal_vars: decl.formal_vars,
                body: decl.body,
                ret_sort,
            });
            self.recfun.name_to_index.insert(decl.name, index);
        }
        Ok(())
    }

    /// Pop the driver's scratch solver scope if one is still open.
    ///
    /// Called from [`Context::invalidate_last_check`], i.e. before every
    /// `assert` / `push` / `pop` touches the solver, and at the top of the
    /// driver. `reset` / `reset_assertions` *forget* the scope instead (see
    /// [`RecFunState::clear_all`]), because they reset the solver outright.
    pub(super) fn discharge_recfun_scope(&mut self) {
        if self.recfun.pending_pop {
            self.recfun.pending_pop = false;
            self.solver.pop();
        }
    }

    /// `check-sat` with recursive definitions in scope.
    ///
    /// See this module's doc comment for the verdict contract; the loop below
    /// is a direct transcription of it, plus one refinement step. When a `sat`
    /// fails to certify, the certifier has computed the *true* value of one or
    /// more concrete applications — `sum(4)`, `sum(3)`, … — straight from the
    /// definitions. Their defining instances are consequences of the axiom just
    /// like any other, so feeding them into the next round's seeds refutes
    /// exactly the model that was just rejected, without ruling out any real
    /// one. That is what lets a query over a *symbolic* argument
    /// (`(and (>= k 0) (= (sum k) 6))`) converge instead of being chased to
    /// ever-deeper symbolic unfoldings, where each round costs exponentially
    /// more than the last.
    ///
    /// More fuel is spent only when a round taught us nothing new, so the whole
    /// loop is bounded by [`MAX_ROUNDS`] solves.
    pub(super) fn check_sat_recfun(&mut self) -> SolverResult {
        self.check_sat_recfun_with(&[])
    }

    /// The driver, shared by `(check-sat)` and every assumption-guarded solve.
    ///
    /// `(check-sat-assuming ..)` and `(get-consequences ..)` funnel through
    /// [`Context::check_with_assumptions_raw`], which does *not* reach
    /// `check_sat`. Without this entry point they would run the plain solver
    /// with every recursive application unconstrained and answer confidently
    /// and wrongly — the same defect the parser's old hard rejection existed to
    /// prevent, reintroduced one command over.
    ///
    /// The assumptions join the roots: an application mentioned only in an
    /// assumption still has to be unfolded, and still has to be certified.
    pub(super) fn check_sat_recfun_with(&mut self, assumptions: &[TermId]) -> SolverResult {
        self.discharge_recfun_scope();
        let assumptions: Vec<TermId> = assumptions.to_vec();
        let mut roots = self.assertions.clone();
        roots.extend(assumptions.iter().copied());
        // The applications a `sat` verdict has to be certified against: every
        // recursive application the *user's* assertions and assumptions
        // mention. Instances introduce more, but those are consequences, not
        // obligations.
        let root_apps = self.collect_recfun_apps(&roots);

        // Concrete applications learned from rejected models, carried across
        // rounds.
        let mut learned: Vec<TermId> = Vec::new();
        let mut learned_seen: FxHashSet<TermId> = FxHashSet::default();
        let mut fuel_index = 0usize;

        for _round in 0..MAX_ROUNDS {
            let Some(&fuel) = FUEL_SCHEDULE.get(fuel_index) else {
                break;
            };
            // The learned instances are ground and cheap; they must not crowd
            // out the structural unfolding the fuel is meant to pay for.
            let budget = fuel.saturating_add(learned.len());
            let unfolding = self.unfold_recfun(&roots, &learned, budget);
            self.solver.push();
            self.recfun.pending_pop = true;
            for &inst in &unfolding.instances {
                self.solver.assert(inst, &mut self.terms);
            }
            match self.check_round(&assumptions) {
                // A refutation from consequences of the axiom refutes the
                // original problem. Final, and the scope stays open so
                // `(get-unsat-core)` / `(get-proof)` see the state it was
                // derived from.
                SolverResult::Unsat => return SolverResult::Unsat,
                SolverResult::Sat => {
                    if !unfolding.truncated && unfolding.boundary.is_empty() {
                        // C1: the unfolding closed, so the instantiated problem
                        // *is* the original one.
                        return SolverResult::Sat;
                    }
                    match self.certify_recfun_sat(&root_apps) {
                        Certification::Certified => return SolverResult::Sat,
                        Certification::Refuted { applications } => {
                            let mut progress = false;
                            for app in applications {
                                if learned.len() >= MAX_LEARNED_APPS {
                                    break;
                                }
                                if learned_seen.insert(app) {
                                    learned.push(app);
                                    progress = true;
                                }
                            }
                            if !progress {
                                fuel_index = fuel_index.saturating_add(1);
                            }
                        }
                        // Nothing was learned: only a deeper unfolding can help.
                        Certification::Inconclusive => {
                            fuel_index = fuel_index.saturating_add(1);
                        }
                    }
                }
                SolverResult::Unknown => {
                    fuel_index = fuel_index.saturating_add(1);
                }
            }
            // Uncertified `sat`, or `unknown`: retract the round's instances
            // and try again.
            self.discharge_recfun_scope();
        }
        SolverResult::Unknown
    }

    /// One round's solve: assumption-guarded when the caller had assumptions,
    /// otherwise the plain check core.
    ///
    /// Both spellings keep the gates their own non-recursive path applies —
    /// `check_sat_core` for the plain check, and the rounding-mode axioms that
    /// [`Context::check_with_assumptions_raw`] documents as the single funnel
    /// for assumption-guarded solves.
    fn check_round(&mut self, assumptions: &[TermId]) -> SolverResult {
        if assumptions.is_empty() {
            return self.check_sat_core();
        }
        if self.terms.rounding_mode_used() {
            self.assert_rounding_mode_distinctness();
        }
        self.solver
            .check_with_assumptions(assumptions, &mut self.terms)
    }

    /// Instantiate the definitional axiom breadth-first from `roots`, stopping
    /// at `fuel` instances.
    ///
    /// The worklist is explicit: the term graph's depth is input-controlled, so
    /// nothing here may recurse natively.
    fn unfold_recfun(&mut self, roots: &[TermId], learned: &[TermId], fuel: usize) -> Unfolding {
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut queue: std::collections::VecDeque<TermId> = std::collections::VecDeque::new();

        // Concrete applications a rejected model taught us about, first: their
        // instances are what refute that model, so they must not be squeezed
        // out by the structural unfolding.
        for &app in learned {
            if seen.insert(app) {
                queue.push_back(app);
            }
        }

        // A nullary definition's axiom has exactly one ground instance, and it
        // holds whether or not the script ever mentions the symbol — so it is
        // seeded unconditionally. Without this, `(define-fun-rec c () Int
        // (+ c 1))` would be reported `sat` for a script that never names `c`,
        // even though the definition alone is contradictory.
        let nullary: Vec<(String, SortId)> = self
            .recfun
            .defs
            .iter()
            .filter(|def| def.formal_vars.is_empty())
            .map(|def| (def.name.clone(), def.ret_sort))
            .collect();
        for (name, sort) in nullary {
            let term = self.terms.mk_var(&name, sort);
            if seen.insert(term) {
                queue.push_back(term);
            }
        }

        for app in self.collect_recfun_apps(roots) {
            if seen.insert(app) {
                queue.push_back(app);
            }
        }

        let mut instances = Vec::new();
        let mut boundary: Vec<TermId> = Vec::new();
        let mut truncated = false;
        while let Some(app) = queue.pop_front() {
            if instances.len() >= fuel {
                queue.push_front(app);
                break;
            }
            if instances.len() >= MAX_INSTANCES {
                truncated = true;
                queue.push_front(app);
                break;
            }
            let Some(body) = self.instantiate_recfun_app(app) else {
                // Not an application of a definition in scope after all (an
                // arity mismatch, or a definition retracted by `pop`). It goes
                // to the boundary rather than being dropped: an application
                // that reached the worklist and left it unconstrained is
                // exactly what makes the round *not* saturated, and silently
                // discarding it would let an empty boundary claim a closure
                // that never happened.
                boundary.push(app);
                continue;
            };
            instances.push(self.terms.mk_eq(app, body));
            for next in self.collect_recfun_apps(&[body]) {
                if seen.insert(next) {
                    queue.push_back(next);
                }
            }
        }

        boundary.extend(queue);
        Unfolding {
            instances,
            boundary,
            truncated,
        }
    }

    /// The right-hand side of `app`'s defining instance: the definition body
    /// with the formal parameters replaced by `app`'s actual arguments, then
    /// normalized.
    fn instantiate_recfun_app(&mut self, app: TermId) -> Option<TermId> {
        let (index, args) = self.recfun_app_parts(app)?;
        let def = self.recfun.defs.get(index)?.clone();
        if args.len() != def.formal_vars.len() {
            return None;
        }
        let mut subst: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (&formal, &actual) in def.formal_vars.iter().zip(args.iter()) {
            subst.insert(formal, actual);
        }
        let body = if subst.is_empty() {
            def.body
        } else {
            self.terms.substitute(def.body, &subst)
        };
        Some(self.normalize_instance(body))
    }

    /// Constant-fold a freshly instantiated body, *including inside function
    /// application arguments*.
    ///
    /// This is what makes an unfolding of a ground recursion terminate.
    /// `TermManager::simplify` deliberately does not rewrite below an `Apply`
    /// node, so without this pass substituting `n := 5` into `fact(n - 1)`
    /// yields `fact(5 - 1)`, then `fact((5 - 1) - 1)`, … — a chain of pairwise
    /// distinct application terms whose base-case `ite` condition never becomes
    /// syntactically decidable, so the worklist never drains and the saturation
    /// certificate can never fire. Folding the arguments turns the same chain
    /// into `fact(4)`, `fact(3)`, … `fact(0)`, whose body folds to a constant
    /// and closes the unfolding.
    ///
    /// The rewrite is sound because `simplify` is semantics-preserving: `x` and
    /// `simplify(x)` denote the same value in every model, so `g(x)` and
    /// `g(simplify(x))` do too, by congruence.
    ///
    /// Applications are processed children-first so that a nested application's
    /// already-folded form is what its parent's argument sees.
    fn normalize_instance(&mut self, term: TermId) -> TermId {
        let mut rewrites: FxHashMap<TermId, TermId> = FxHashMap::default();
        for node in self.post_order(term) {
            let Some(t) = self.terms.get(node) else {
                continue;
            };
            let TermKind::Apply { func, args } = &t.kind else {
                continue;
            };
            let sort = t.sort;
            let name = self.terms.resolve_str(*func).to_string();
            let args: Vec<TermId> = args.iter().copied().collect();
            let mut folded = Vec::with_capacity(args.len());
            for arg in args {
                let arg = if rewrites.is_empty() {
                    arg
                } else {
                    self.terms.substitute(arg, &rewrites)
                };
                folded.push(self.terms.simplify(arg));
            }
            let rebuilt = self.terms.mk_apply(&name, folded, sort);
            if rebuilt != node {
                rewrites.insert(node, rebuilt);
            }
        }
        let term = if rewrites.is_empty() {
            term
        } else {
            self.terms.substitute(term, &rewrites)
        };
        self.terms.simplify(term)
    }

    /// Every subterm of `term`, children before parents, each visited once.
    fn post_order(&self, term: TermId) -> Vec<TermId> {
        let mut order = Vec::new();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        // `(node, expanded)`: the second visit emits the node.
        let mut stack: Vec<(TermId, bool)> = vec![(term, false)];
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                order.push(node);
                continue;
            }
            if !visited.insert(node) {
                continue;
            }
            stack.push((node, true));
            if let Some(t) = self.terms.get(node) {
                for child in get_children(&t.kind) {
                    if !visited.contains(&child) {
                        stack.push((child, false));
                    }
                }
            }
        }
        order
    }

    /// Every application of an in-scope recursive definition occurring in
    /// `roots`, deduplicated.
    fn collect_recfun_apps(&self, roots: &[TermId]) -> Vec<TermId> {
        let mut found = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = roots.to_vec();
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            if self.recfun_app_parts(node).is_some() {
                found.push(node);
            }
            if let Some(t) = self.terms.get(node) {
                for child in get_children(&t.kind) {
                    stack.push(child);
                }
            }
        }
        found
    }

    /// Decompose `term` into `(definition index, arguments)` when it applies an
    /// in-scope recursive definition.
    ///
    /// A nullary definition is a `Var`, not an `Apply`: the parser registers it
    /// among the constants so its self-reference resolves, and the solver
    /// declares it with `declare_const`, so the two mint the same term.
    fn recfun_app_parts(&self, term: TermId) -> Option<(usize, Vec<TermId>)> {
        let t = self.terms.get(term)?;
        let (name, args): (&str, Vec<TermId>) = match &t.kind {
            TermKind::Apply { func, args } => (
                self.terms.resolve_str(*func),
                args.iter().copied().collect(),
            ),
            TermKind::Var(name) => (self.terms.resolve_str(*name), Vec::new()),
            _ => return None,
        };
        let index = *self.recfun.name_to_index.get(name)?;
        let def = self.recfun.defs.get(index)?;
        if def.formal_vars.len() != args.len() {
            return None;
        }
        Some((index, args))
    }

    /// The `(define-fun-rec ..)` lines `(get-model)` appends for the recursive
    /// definitions in scope.
    ///
    /// A name redefined in the same scope leaves the superseded `RecDef` in
    /// `defs` (it has to stay, so `pop` can restore it), so only the definition
    /// the name currently resolves to is printed — a model listing two
    /// interpretations for one symbol would be nonsense.
    pub(super) fn recfun_model_lines(&self) -> Vec<String> {
        let printer = oxiz_core::smtlib::Printer::new(&self.terms);
        self.recfun
            .defs
            .iter()
            .enumerate()
            .filter(|(index, def)| self.recfun.name_to_index.get(&def.name) == Some(index))
            .map(|(_, def)| {
                let params: Vec<String> = def
                    .params
                    .iter()
                    .map(|(name, sort)| format!("({name} {sort})"))
                    .collect();
                format!(
                    "  (define-fun-rec {} ({}) {} {})",
                    def.name,
                    params.join(" "),
                    def.ret_sort_name,
                    printer.print_term(def.body)
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pop` must retract exactly the definitions made inside the scope, and
    /// leave the outer ones (and the name index) intact.
    #[test]
    fn scope_stack_retracts_exactly_the_inner_definitions() {
        let mut state = RecFunState::default();
        let def = |name: &str| RecDef {
            name: name.to_string(),
            params: Vec::new(),
            ret_sort_name: "Int".to_string(),
            formal_vars: Vec::new(),
            body: TermId::new(0),
            ret_sort: SortId::new(0),
        };
        state.defs.push(def("outer"));
        state.name_to_index.insert("outer".to_string(), 0);
        state.push_scope();
        state.defs.push(def("inner"));
        state.name_to_index.insert("inner".to_string(), 1);

        state.pop_scope();
        assert_eq!(state.defs.len(), 1);
        assert!(state.lookup("outer").is_some(), "outer definition survives");
        assert!(state.lookup("inner").is_none(), "inner definition is gone");
    }

    /// `reset-assertions` keeps top-level definitions and drops scoped ones.
    #[test]
    fn retract_to_base_keeps_only_top_level_definitions() {
        let mut state = RecFunState::default();
        let def = |name: &str| RecDef {
            name: name.to_string(),
            params: Vec::new(),
            ret_sort_name: "Int".to_string(),
            formal_vars: Vec::new(),
            body: TermId::new(0),
            ret_sort: SortId::new(0),
        };
        state.defs.push(def("top"));
        state.name_to_index.insert("top".to_string(), 0);
        state.push_scope();
        state.defs.push(def("scoped"));
        state.name_to_index.insert("scoped".to_string(), 1);
        state.pending_pop = true;

        state.retract_to_base();
        assert_eq!(state.defs.len(), 1);
        assert!(state.lookup("top").is_some());
        assert!(state.lookup("scoped").is_none());
        assert!(
            !state.pending_pop,
            "the scratch scope must be forgotten, not left owing a pop"
        );
        assert!(state.scope_stack.is_empty());
    }
}
