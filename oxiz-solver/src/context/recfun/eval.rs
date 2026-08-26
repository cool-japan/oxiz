//! Certificate C2: re-compute every recursive application under the candidate
//! model and check that the model already agrees.
//!
//! # Why a `sat` needs this at all
//!
//! The unfolding driver hands the solver *instances* of the definitional
//! axiom, so the problem it actually solves is a relaxation: any application
//! whose instance was not produced is left free, and the solver may invent a
//! value for it. A model of the relaxation is a model of the original problem
//! only if it happens to interpret every recursive application the way the
//! definition does. This module checks exactly that, by evaluating each
//! application in the user's assertions straight from the definitions.
//!
//! # Why this evaluator terminates on inputs that the definitions do not
//!
//! Three mechanisms, all of them load-bearing:
//!
//! * **`ite` is short-circuited.** The condition is evaluated first and only
//!   the taken branch is pushed. This is what lets `fact(0)` finish: its body
//!   is `(ite (= 0 0) 1 (* 0 (fact (- 0 1))))`, and an eager evaluator would
//!   descend into `fact(-1)`, then `fact(-2)`, forever. `and` / `or` short-
//!   circuit for the same reason.
//! * **Re-entry is divergence.** An application key is marked
//!   [`Slot::InProgress`] while its body is being evaluated; meeting the same
//!   key again means the definition is unfolding into itself with no progress,
//!   which is answered `None` (uncertified) rather than followed.
//! * **Two finite budgets.** [`EVAL_BUDGET`] bounds evaluation steps and
//!   [`MAX_EVAL_APPS`] bounds distinct application evaluations, so a definition
//!   that diverges *without* repeating a key — `loop(x) = loop(x + 1)` — still
//!   stops.
//!
//! Failing to certify is never a wrong answer: it only costs the round its
//! `sat`, and an exhausted fuel schedule reports `unknown`.
//!
//! The walk carries its own heap stack. Term depth is input-controlled, so
//! native recursion is not an option.

#[allow(unused_imports)]
use crate::prelude::*;
use crate::solver::Model;
use oxiz_core::ast::traversal::get_children;
use oxiz_core::ast::{TermId, TermKind, TermManager};

use super::{Certification, Context, RecDef};

/// Evaluation steps allowed per certification attempt.
const EVAL_BUDGET: u64 = 200_000;

/// Distinct recursive applications evaluated per certification attempt. This is
/// the budget that actually bites for a non-repeating divergence such as
/// `loop(x) = loop(x + 1)`, where every unfolding produces a fresh key and no
/// re-entry is ever detected.
///
/// Sized for the recursion depths a real definition reaches — a `fact`/`fib`
/// certification needs one evaluation per distinct argument — while keeping the
/// cost of *failing* to certify bounded: the whole schedule re-attempts the
/// certification once per fuel round, so a budget an order of magnitude larger
/// buys nothing and makes an honest `unknown` take an order of magnitude longer
/// to reach.
const MAX_EVAL_APPS: usize = 2_000;

/// Memo entry for one application key.
enum Slot {
    /// The body is being evaluated; meeting this again is divergence.
    InProgress,
    /// The application's value.
    Done(TermId),
}

/// A suspended step of the evaluation.
enum Task {
    /// Evaluate a term, pushing its value.
    Eval(TermId),
    /// All arguments of a recursive application are on the value stack: build
    /// the application key and start its body.
    OpenApp {
        /// The definition being applied.
        index: usize,
        /// How many argument values to pop.
        arity: usize,
    },
    /// The body of the application keyed `key` has produced its value.
    CloseApp {
        /// The key to memoize the value under.
        key: TermId,
    },
    /// An `ite` condition is on the value stack; take one branch.
    Branch {
        /// The `then` term.
        then_term: TermId,
        /// The `else` term.
        else_term: TermId,
    },
    /// One operand of an `and` / `or` is on the value stack.
    Connective {
        /// `true` for `and`, `false` for `or`.
        is_and: bool,
        /// The remaining operands.
        operands: Vec<TermId>,
        /// The index of the operand just evaluated.
        done: usize,
    },
    /// Every child of a generic node is on the value stack; rebuild it and read
    /// off a value.
    Rebuild {
        /// The original node.
        node: TermId,
        /// Its children, in order (their values are on the stack).
        children: Vec<TermId>,
    },
}

/// Whether `term` is a literal value rather than merely a ground expression.
///
/// This distinction is the whole point: comparing what the model says against
/// what the definition says is only evidence of agreement when both sides
/// reduced to an actual value. If the model leaves an application unevaluated
/// and the evaluator returns the same unevaluated term, the two `TermId`s are
/// *equal* and would falsely certify. (`utils::stats::is_ground` is not this
/// check — `(+ 1 2)` is ground and is not a value.)
fn is_value(manager: &TermManager, term: TermId) -> bool {
    matches!(
        manager.get(term).map(|t| &t.kind),
        Some(
            TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_)
        )
    )
}

impl Context {
    /// Certificate C2 for the applications in `apps`.
    ///
    /// Returns `true` only when every application both re-computes to a value
    /// from the definitions and matches the value the model gives it.
    pub(super) fn certify_recfun_sat(&mut self, apps: &[TermId]) -> Certification {
        if self.last_check_produced_no_model() {
            return Certification::Inconclusive;
        }
        let Some(model) = self.solver.model().cloned() else {
            // No model to certify. An empty assertion stack cannot mention a
            // recursive application in the first place, so there is nothing
            // this could certify vacuously.
            return if apps.is_empty() {
                Certification::Certified
            } else {
                Certification::Inconclusive
            };
        };
        // Complete the model the way `(get-value ..)` does: a declared constant
        // with no entry takes its sort default, otherwise a perfectly good
        // model leaves the evaluator with a free variable and no value.
        let completion = self.model_completion(&model);
        let defs = self.recfun.defs.clone();
        let index_of: FxHashMap<String, usize> = self
            .recfun
            .name_to_index
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        let mut certifier = Certifier {
            terms: &mut self.terms,
            model: &model,
            completion: &completion,
            defs: &defs,
            index_of: &index_of,
            app_memo: FxHashMap::default(),
            node_memo: FxHashMap::default(),
            budget: EVAL_BUDGET,
            apps_evaluated: 0,
        };

        for &app in apps {
            let (Some(computed), Some(claimed)) = (certifier.eval(app), certifier.value_of(app))
            else {
                // The re-computation did not finish, so the model was neither
                // confirmed nor refuted, and the partial values the evaluator
                // did reach came from a chain it abandoned. Reporting them as
                // *learned* would be sound but useless — they are the divergent
                // tail, not the disagreement.
                return Certification::Inconclusive;
            };
            if computed != claimed {
                return Certification::Refuted {
                    applications: certifier.resolved_applications(),
                };
            }
        }
        Certification::Certified
    }

    /// Whether the last solver check left no usable model.
    fn last_check_produced_no_model(&self) -> bool {
        self.solver.model().is_none() && !self.assertions.is_empty()
    }

    /// Sort defaults for every declared constant the model left unassigned.
    fn model_completion(&mut self, model: &Model) -> FxHashMap<TermId, TermId> {
        let unassigned: Vec<(TermId, oxiz_core::sort::SortId)> = self
            .declared_consts
            .iter()
            .filter(|d| model.get(d.term).is_none())
            .map(|d| (d.term, d.sort))
            .collect();
        let mut completion: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (term, sort) in unassigned {
            if let Some(value) =
                crate::solver::model_builder::ground_default_term(&mut self.terms, sort)
            {
                completion.insert(term, value);
            }
        }
        completion
    }
}

/// The evaluation state, borrowed apart from [`Context`] so the term manager can
/// be mutated while the definitions are read.
struct Certifier<'a> {
    terms: &'a mut TermManager,
    model: &'a Model,
    completion: &'a FxHashMap<TermId, TermId>,
    defs: &'a [RecDef],
    index_of: &'a FxHashMap<String, usize>,
    /// Application key -> its value (or "being computed").
    app_memo: FxHashMap<TermId, Slot>,
    /// Term -> its value. Sound to share across the whole certification: the
    /// model is fixed and every term reaching the evaluator is closed, so a
    /// term's value does not depend on where it was met.
    node_memo: FxHashMap<TermId, TermId>,
    budget: u64,
    apps_evaluated: usize,
}

impl Certifier<'_> {
    /// The concrete applications this run pinned to a value, straight from the
    /// definitions.
    ///
    /// Each one's defining instance is a consequence of the axiom, so the
    /// driver can assert them to refute exactly the model that was rejected.
    fn resolved_applications(&self) -> Vec<TermId> {
        self.app_memo
            .iter()
            .filter(|(_, slot)| matches!(slot, Slot::Done(_)))
            .map(|(&key, _)| key)
            .collect()
    }

    /// Evaluate `root` to a literal value, or `None` if it diverges, exhausts a
    /// budget, or reaches something with no value under the model.
    fn eval(&mut self, root: TermId) -> Option<TermId> {
        let mut tasks: Vec<Task> = vec![Task::Eval(root)];
        let mut values: Vec<TermId> = Vec::new();

        while let Some(task) = tasks.pop() {
            self.budget = self.budget.checked_sub(1)?;
            match task {
                Task::Eval(term) => self.open(term, &mut tasks, &mut values)?,
                Task::OpenApp { index, arity } => {
                    let split = values.len().checked_sub(arity)?;
                    let args: Vec<TermId> = values.split_off(split);
                    self.open_app(index, args, &mut tasks, &mut values)?;
                }
                Task::CloseApp { key } => {
                    let value = *values.last()?;
                    self.app_memo.insert(key, Slot::Done(value));
                }
                Task::Branch {
                    then_term,
                    else_term,
                } => {
                    let cond = values.pop()?;
                    let taken = match self.terms.get(cond).map(|t| &t.kind) {
                        Some(TermKind::True) => then_term,
                        Some(TermKind::False) => else_term,
                        // A condition that did not reduce to a Boolean value
                        // leaves the branch undetermined: no certificate.
                        _ => return None,
                    };
                    tasks.push(Task::Eval(taken));
                }
                Task::Connective {
                    is_and,
                    operands,
                    done,
                } => {
                    let value = values.pop()?;
                    let short_circuit = match self.terms.get(value).map(|t| &t.kind) {
                        Some(TermKind::True) => !is_and,
                        Some(TermKind::False) => is_and,
                        _ => return None,
                    };
                    let next = done.checked_add(1)?;
                    if short_circuit {
                        values.push(value);
                    } else if next < operands.len() {
                        let operand = *operands.get(next)?;
                        tasks.push(Task::Connective {
                            is_and,
                            operands,
                            done: next,
                        });
                        tasks.push(Task::Eval(operand));
                    } else {
                        // Every operand agreed with the neutral value.
                        let neutral = if is_and {
                            self.terms.mk_true()
                        } else {
                            self.terms.mk_false()
                        };
                        values.push(neutral);
                    }
                }
                Task::Rebuild { node, children } => {
                    let split = values.len().checked_sub(children.len())?;
                    let child_values: Vec<TermId> = values.split_off(split);
                    let mut subst: FxHashMap<TermId, TermId> = FxHashMap::default();
                    for (&child, &value) in children.iter().zip(child_values.iter()) {
                        if child != value {
                            subst.insert(child, value);
                        }
                    }
                    let rebuilt = if subst.is_empty() {
                        node
                    } else {
                        self.terms.substitute(node, &subst)
                    };
                    let value = self.value_of(rebuilt)?;
                    self.node_memo.insert(node, value);
                    values.push(value);
                }
            }
        }
        values.pop()
    }

    /// Classify `term` and schedule the work it needs.
    fn open(
        &mut self,
        term: TermId,
        tasks: &mut Vec<Task>,
        values: &mut Vec<TermId>,
    ) -> Option<()> {
        if let Some(&value) = self.node_memo.get(&term) {
            values.push(value);
            return Some(());
        }
        // A recursive application: evaluate the arguments, then the body.
        if let Some((index, args)) = self.recfun_parts(term) {
            tasks.push(Task::OpenApp {
                index,
                arity: args.len(),
            });
            for &arg in args.iter().rev() {
                tasks.push(Task::Eval(arg));
            }
            return Some(());
        }
        let kind = self.terms.get(term).map(|t| t.kind.clone())?;
        match kind {
            TermKind::Ite(cond, then_term, else_term) => {
                tasks.push(Task::Branch {
                    then_term,
                    else_term,
                });
                tasks.push(Task::Eval(cond));
            }
            TermKind::And(ref operands) | TermKind::Or(ref operands) => {
                let is_and = matches!(kind, TermKind::And(_));
                let operands: Vec<TermId> = operands.iter().copied().collect();
                match operands.first() {
                    Some(&first) => {
                        tasks.push(Task::Connective {
                            is_and,
                            operands,
                            done: 0,
                        });
                        tasks.push(Task::Eval(first));
                    }
                    None => {
                        let neutral = if is_and {
                            self.terms.mk_true()
                        } else {
                            self.terms.mk_false()
                        };
                        values.push(neutral);
                    }
                }
            }
            // Binders own a scope this evaluator does not model, so they are
            // opaque: whatever the model makes of them, taken whole. A binder
            // still containing an unevaluated recursive application simply has
            // no value, and the certificate honestly fails.
            TermKind::Forall { .. }
            | TermKind::Exists { .. }
            | TermKind::Let { .. }
            | TermKind::Match { .. } => {
                let value = self.value_of(term)?;
                self.node_memo.insert(term, value);
                values.push(value);
            }
            other => {
                let children = get_children(&other);
                if children.is_empty() {
                    let value = self.value_of(term)?;
                    self.node_memo.insert(term, value);
                    values.push(value);
                } else {
                    let children: Vec<TermId> = children.into_iter().collect();
                    // `Rebuild` goes on first so it is popped *last*, after
                    // every child has pushed its value; the children go on in
                    // reverse so they are popped left to right.
                    tasks.push(Task::Rebuild {
                        node: term,
                        children: children.clone(),
                    });
                    for &child in children.iter().rev() {
                        tasks.push(Task::Eval(child));
                    }
                }
            }
        }
        Some(())
    }

    /// Start evaluating an application whose arguments are already values.
    fn open_app(
        &mut self,
        index: usize,
        args: Vec<TermId>,
        tasks: &mut Vec<Task>,
        values: &mut Vec<TermId>,
    ) -> Option<()> {
        let def = self.defs.get(index)?;
        let key = if args.is_empty() {
            // A nullary definition is its own key: the `Var` the parser and the
            // context both mint for the name.
            self.terms.mk_var(&def.name, def.ret_sort)
        } else {
            self.terms
                .mk_apply(&def.name, args.iter().copied(), def.ret_sort)
        };
        match self.app_memo.get(&key) {
            Some(Slot::Done(value)) => {
                values.push(*value);
                return Some(());
            }
            // Re-entry: the definition unfolds into itself with no progress.
            Some(Slot::InProgress) => return None,
            None => {}
        }
        self.apps_evaluated = self.apps_evaluated.checked_add(1)?;
        if self.apps_evaluated > MAX_EVAL_APPS {
            return None;
        }
        let def = self.defs.get(index)?;
        let formal_vars = def.formal_vars.clone();
        let body = def.body;
        if formal_vars.len() != args.len() {
            return None;
        }
        let mut subst: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (&formal, &actual) in formal_vars.iter().zip(args.iter()) {
            subst.insert(formal, actual);
        }
        let instantiated = if subst.is_empty() {
            body
        } else {
            self.terms.substitute(body, &subst)
        };
        self.app_memo.insert(key, Slot::InProgress);
        tasks.push(Task::CloseApp { key });
        tasks.push(Task::Eval(instantiated));
        Some(())
    }

    /// Decompose a recursive application, mirroring `Context::recfun_app_parts`
    /// against the borrowed definition tables.
    fn recfun_parts(&self, term: TermId) -> Option<(usize, Vec<TermId>)> {
        let t = self.terms.get(term)?;
        let (name, args): (&str, Vec<TermId>) = match &t.kind {
            TermKind::Apply { func, args } => (
                self.terms.resolve_str(*func),
                args.iter().copied().collect(),
            ),
            TermKind::Var(name) => (self.terms.resolve_str(*name), Vec::new()),
            _ => return None,
        };
        let index = *self.index_of.get(name)?;
        let def = self.defs.get(index)?;
        if def.formal_vars.len() != args.len() {
            return None;
        }
        Some((index, args))
    }

    /// The model's value for `term`, or `None` when the model does not pin it
    /// down to a literal.
    ///
    /// The two fast paths matter for more than speed: most nodes reaching here
    /// have already had their children replaced by literals, so plain rewriting
    /// finishes them, and only genuine leaves (a variable, an uninterpreted
    /// application) need to consult the model at all.
    fn value_of(&mut self, term: TermId) -> Option<TermId> {
        if is_value(self.terms, term) {
            return Some(term);
        }
        let rewritten = self.terms.simplify(term);
        if is_value(self.terms, rewritten) {
            return Some(rewritten);
        }
        let completed = if self.completion.is_empty() {
            term
        } else {
            self.terms.substitute(term, self.completion)
        };
        let value = self.model.eval(completed, self.terms);
        let value = self.terms.simplify(value);
        if is_value(self.terms, value) {
            Some(value)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_value_rejects_ground_non_values() {
        let mut m = TermManager::new();
        let one = m.mk_int(1);
        let truth = m.mk_true();
        let int_sort = m.sorts.int_sort;
        let unevaluated = m.mk_apply("f", vec![one], int_sort);
        assert!(is_value(&m, one));
        assert!(is_value(&m, truth));
        // An unevaluated application is not a value. Accepting one here would
        // compare the model's unevaluated term against the evaluator's
        // identical unevaluated term and wrongly report agreement.
        assert!(!is_value(&m, unevaluated));
    }
}
