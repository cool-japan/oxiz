//! The Skolemization tactic.
//!
//! Split out of the former single-file `tactic/quantifier.rs`; see
//! [`super`] for the module layout. Pure code motion.

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use smallvec::SmallVec;

use super::subst::substitute_single_var;
use crate::tactic::{Goal, TacticResult};
use std::rc::Rc;

/// Which node a [`SkFrame::Rebuild`] reconstructs from its children.
#[derive(Debug, Clone, Copy)]
enum SkOp {
    Not,
    And,
    Or,
    Implies,
    /// The untouched (mixed-polarity) condition; the two pending results are
    /// the Skolemized branches.
    Ite(TermId),
}

/// One pending step of [`SkolemizationTactic::skolemize_polar`]'s
/// explicit-stack walk.
#[derive(Debug)]
enum SkFrame {
    /// Skolemize `term_id` at `positive` polarity under `governing`.
    Eval {
        term_id: TermId,
        positive: bool,
        /// Effectively-universal variables in scope, shared by reference
        /// count so a deep binder nest does not clone the vector into every
        /// pending frame.
        governing: Rc<Vec<(Spur, SortId)>>,
    },
    /// All children of a Boolean node are done; rebuild it (keeping the
    /// original `TermId` when nothing changed, as the recursive version did).
    Rebuild {
        term_id: TermId,
        op: SkOp,
        orig: SmallVec<[TermId; 4]>,
    },
    /// A kept (effectively-universal) binder whose body is done.
    Binder {
        term_id: TermId,
        vars: Vec<(Spur, SortId)>,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
        is_forall: bool,
    },
}

/// Skolemization tactic
///
/// Eliminates existential quantifiers by replacing them with fresh Skolem
/// functions/constants, preserving equisatisfiability.
///
/// Correctness requirements handled here (each was previously violated):
///
/// 1. **Fresh names per goal.**  A single monotone counter is threaded through
///    *all* assertions of the goal, so distinct existentials always receive
///    distinct Skolem symbols.  (Resetting the counter per assertion made
///    {∃x.P(x), ∃x.¬P(x)} collapse to {P(sk_0), ¬P(sk_0)} — SAT → UNSAT.)
///
/// 2. **Polarity.**  Only *effectively existential* quantifiers are Skolemized:
///    a `Exists` under positive polarity, or a `Forall` under negative
///    polarity.  Effectively *universal* quantifiers are kept and their bound
///    variables become the arguments of inner Skolem functions.  (Ignoring
///    polarity let ¬(∃x.P(x)) become ¬P(sk_0) — UNSAT → SAT.)
///
/// 3. **Real argument sorts.**  Skolem function arguments use the *actual*
///    sorts of the governing universal variables, not a hard-coded `Bool`.
///
/// Skolemization only descends through Boolean structure (`Not`, `And`, `Or`,
/// `Implies`, `Ite` branches, and quantifiers).  Sub-formulas at genuinely
/// mixed polarity (an `Ite` condition, a Boolean equality) are left untouched
/// rather than Skolemized unsoundly.
#[derive(Debug)]
pub struct SkolemizationTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> SkolemizationTactic<'a> {
    /// Create a new Skolemization tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply the tactic to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut changed = false;
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());
        // One monotone counter shared across every assertion so that Skolem
        // names never collide between distinct existentials.
        let mut counter: usize = 0;

        for &assertion in &goal.assertions {
            let governing: Vec<(Spur, SortId)> = Vec::new();
            let skolemized = self.skolemize_polar(assertion, true, &governing, &mut counter);
            if skolemized != assertion {
                changed = true;
            }
            new_assertions.push(skolemized);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Polarity-aware Skolemization.
    ///
    /// `positive` is the polarity of `term_id` in the enclosing assertion
    /// (top-level assertions start positive).  `governing` lists the
    /// effectively-universal variables currently in scope, with their real
    /// sorts, used as Skolem-function arguments.
    /// # Explicit work stack
    ///
    /// This and its `skolemize_existential`/`skolemize_universal` helpers
    /// were a three-way mutual recursion over the assertion's Boolean and
    /// quantifier nesting, with no depth guard of any kind. The return type
    /// is `TermId` — there is no error channel — and a cap could not be added
    /// honestly: stopping partway would leave an existential un-Skolemized
    /// inside a formula the tactic reports as Skolemized, or (worse, on the
    /// `skolemize_existential` path, which drops the binder *before*
    /// recursing) leave a dropped binder's variable free. The walk is
    /// therefore driven by an explicit heap [`Vec`] of [`SkFrame`]s. Every
    /// frame that consumes child results was pushed together with exactly the
    /// evaluations that produce them.
    ///
    /// No memoization: the Skolemization of a subterm depends on its
    /// polarity, on the governing universals in scope, *and* on the mutable
    /// fresh-name counter — two occurrences of one shared subterm must
    /// receive **different** Skolem symbols, so reusing a cached result would
    /// reintroduce exactly the name collision this tactic's point 1 exists to
    /// prevent.
    fn skolemize_polar(
        &mut self,
        term_id: TermId,
        positive: bool,
        governing: &[(Spur, SortId)],
        counter: &mut usize,
    ) -> TermId {
        let mut frames: Vec<SkFrame> = vec![SkFrame::Eval {
            term_id,
            positive,
            governing: Rc::new(governing.to_vec()),
        }];
        let mut results: Vec<TermId> = Vec::new();

        while let Some(frame) = frames.pop() {
            match frame {
                SkFrame::Eval {
                    term_id,
                    positive,
                    governing,
                } => {
                    self.sk_eval(
                        term_id,
                        positive,
                        &governing,
                        counter,
                        &mut frames,
                        &mut results,
                    );
                }

                SkFrame::Rebuild { term_id, op, orig } => {
                    let start = results.len().saturating_sub(orig.len());
                    let mut new_args: SmallVec<[TermId; 4]> = results.split_off(start).into();
                    while new_args.len() < orig.len() {
                        new_args.push(orig[new_args.len()]);
                    }

                    let rebuilt = if new_args == orig {
                        term_id
                    } else {
                        match op {
                            SkOp::Not => self.manager.mk_not(new_args[0]),
                            SkOp::And => self.manager.mk_and(new_args),
                            SkOp::Or => self.manager.mk_or(new_args),
                            SkOp::Implies => self.manager.mk_implies(new_args[0], new_args[1]),
                            SkOp::Ite(cond) => self.manager.mk_ite(cond, new_args[0], new_args[1]),
                        }
                    };
                    results.push(rebuilt);
                }

                SkFrame::Binder {
                    term_id,
                    vars,
                    patterns,
                    is_forall,
                } => {
                    let sk_body = results.pop().unwrap_or(term_id);
                    let rebuilt = self.rebuild_binder(&vars, sk_body, &patterns, is_forall);
                    results.push(rebuilt);
                }
            }
        }

        results.pop().unwrap_or(term_id)
    }

    /// Expand one [`SkFrame::Eval`].
    fn sk_eval(
        &mut self,
        term_id: TermId,
        positive: bool,
        governing: &Rc<Vec<(Spur, SortId)>>,
        counter: &mut usize,
        frames: &mut Vec<SkFrame>,
        results: &mut Vec<TermId>,
    ) {
        let kind = match self.manager.get(term_id) {
            Some(t) => t.kind.clone(),
            None => {
                results.push(term_id);
                return;
            }
        };

        match kind {
            TermKind::Not(arg) => {
                // Negation flips the polarity of its argument.
                frames.push(SkFrame::Rebuild {
                    term_id,
                    op: SkOp::Not,
                    orig: SmallVec::from_slice(&[arg]),
                });
                frames.push(SkFrame::Eval {
                    term_id: arg,
                    positive: !positive,
                    governing: Rc::clone(governing),
                });
            }
            TermKind::And(args) => {
                Self::push_uniform(frames, term_id, SkOp::And, args, positive, governing);
            }
            TermKind::Or(args) => {
                Self::push_uniform(frames, term_id, SkOp::Or, args, positive, governing);
            }
            TermKind::Implies(lhs, rhs) => {
                // Antecedent is at flipped polarity, consequent keeps polarity.
                frames.push(SkFrame::Rebuild {
                    term_id,
                    op: SkOp::Implies,
                    orig: SmallVec::from_slice(&[lhs, rhs]),
                });
                frames.push(SkFrame::Eval {
                    term_id: rhs,
                    positive,
                    governing: Rc::clone(governing),
                });
                frames.push(SkFrame::Eval {
                    term_id: lhs,
                    positive: !positive,
                    governing: Rc::clone(governing),
                });
            }
            TermKind::Ite(cond, then_br, else_br) => {
                // `cond` occurs at mixed polarity (both c and ¬c); leave it
                // untouched.  Both branches preserve the ambient polarity.
                frames.push(SkFrame::Rebuild {
                    term_id,
                    op: SkOp::Ite(cond),
                    orig: SmallVec::from_slice(&[then_br, else_br]),
                });
                Self::push_uniform_evals(frames, &[then_br, else_br], positive, governing);
            }
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                if positive {
                    // Effectively universal: keep binder, extend governing set.
                    Self::push_universal(
                        frames, term_id, &vars, body, patterns, true, positive, governing,
                    );
                } else {
                    // ¬∀x.φ ≡ ∃x.¬φ: effectively existential, Skolemize it.
                    self.push_existential(frames, &vars, body, positive, governing, counter);
                }
            }
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                if positive {
                    // Effectively existential: Skolemize.
                    self.push_existential(frames, &vars, body, positive, governing, counter);
                } else {
                    // ¬∃x.φ ≡ ∀x.¬φ: effectively universal, keep binder.
                    Self::push_universal(
                        frames, term_id, &vars, body, patterns, false, positive, governing,
                    );
                }
            }
            // Atoms and mixed-polarity contexts (Boolean equalities, arithmetic,
            // uninterpreted applications, …) are left unchanged: they cannot be
            // Skolemized soundly without polarity information we do not have
            // here, and leaving them intact keeps the result equisatisfiable.
            _ => results.push(term_id),
        }
    }

    /// Queue a rebuild frame plus one evaluation per argument, all at the
    /// ambient polarity and governing set.
    fn push_uniform(
        frames: &mut Vec<SkFrame>,
        term_id: TermId,
        op: SkOp,
        args: SmallVec<[TermId; 4]>,
        positive: bool,
        governing: &Rc<Vec<(Spur, SortId)>>,
    ) {
        frames.push(SkFrame::Rebuild {
            term_id,
            op,
            orig: args.clone(),
        });
        Self::push_uniform_evals(frames, &args, positive, governing);
    }

    /// Queue one evaluation per element of `args`, ordered so results land in
    /// argument order.
    fn push_uniform_evals(
        frames: &mut Vec<SkFrame>,
        args: &[TermId],
        positive: bool,
        governing: &Rc<Vec<(Spur, SortId)>>,
    ) {
        for &arg in args.iter().rev() {
            frames.push(SkFrame::Eval {
                term_id: arg,
                positive,
                governing: Rc::clone(governing),
            });
        }
    }

    /// Handle an effectively-universal quantifier: keep the binder, add its
    /// variables to the governing set, and queue the body.
    #[allow(clippy::too_many_arguments)]
    fn push_universal(
        frames: &mut Vec<SkFrame>,
        term_id: TermId,
        vars: &[(Spur, SortId)],
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
        is_forall: bool,
        positive: bool,
        governing: &Rc<Vec<(Spur, SortId)>>,
    ) {
        let mut gov = (**governing).clone();
        gov.extend(vars.iter().copied());

        frames.push(SkFrame::Binder {
            term_id,
            vars: vars.to_vec(),
            patterns,
            is_forall,
        });
        frames.push(SkFrame::Eval {
            term_id: body,
            positive,
            governing: Rc::new(gov),
        });
    }

    /// Skolemize an effectively-existential quantifier: replace each bound
    /// variable with a fresh Skolem term over the governing universals, drop
    /// the binder, and queue the substituted body. The substituted body's
    /// value *is* this node's value, so no rebuild frame is needed.
    fn push_existential(
        &mut self,
        frames: &mut Vec<SkFrame>,
        vars: &[(Spur, SortId)],
        body: TermId,
        positive: bool,
        governing: &Rc<Vec<(Spur, SortId)>>,
        counter: &mut usize,
    ) {
        let mut substituted = body;
        for &(var_name, var_sort) in vars {
            let sk_term = self.make_skolem_term(var_sort, governing, counter);
            substituted = substitute_single_var(self.manager, substituted, var_name, sk_term);
        }
        frames.push(SkFrame::Eval {
            term_id: substituted,
            positive,
            governing: Rc::clone(governing),
        });
    }

    /// Rebuild a kept binder around its Skolemized body.
    fn rebuild_binder(
        &mut self,
        vars: &[(Spur, SortId)],
        sk_body: TermId,
        patterns: &[SmallVec<[TermId; 2]>],
        is_forall: bool,
    ) -> TermId {
        let var_names: Vec<_> = vars
            .iter()
            .map(|(n, s)| (self.manager.resolve_str(*n).to_string(), *s))
            .collect();
        let var_strs: Vec<_> = var_names
            .iter()
            .map(|(name, sort)| (name.as_str(), *sort))
            .collect();
        let patterns_owned: SmallVec<[SmallVec<[TermId; 2]>; 2]> =
            patterns.iter().cloned().collect();
        if is_forall {
            self.manager
                .mk_forall_with_patterns(var_strs, sk_body, patterns_owned)
        } else {
            self.manager
                .mk_exists_with_patterns(var_strs, sk_body, patterns_owned)
        }
    }

    /// Build the Skolem term for a variable of sort `var_sort`: a fresh
    /// constant when no universals govern it, otherwise a fresh function
    /// applied to the governing universal variables (using their real sorts).
    fn make_skolem_term(
        &mut self,
        var_sort: SortId,
        governing: &[(Spur, SortId)],
        counter: &mut usize,
    ) -> TermId {
        let skolem_name = crate::smtlib::reserved_name("sk", &counter.to_string());
        *counter += 1;

        if governing.is_empty() {
            self.manager.mk_var(&skolem_name, var_sort)
        } else {
            let gov_names: Vec<_> = governing
                .iter()
                .map(|(n, s)| (self.manager.resolve_str(*n).to_string(), *s))
                .collect();
            let arg_ids: SmallVec<[TermId; 4]> = gov_names
                .iter()
                .map(|(name, sort)| self.manager.mk_var(name, *sort))
                .collect();
            self.manager.mk_apply(&skolem_name, arg_ids, var_sort)
        }
    }
}
