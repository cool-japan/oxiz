//! Skolemization and universal-quantifier elimination.
//!
//! Split out of `ast/normal_forms.rs`; see that module's doc comment for the
//! general iterative-conversion rationale, and its "Capture avoidance"
//! section for why calling `TermManager::substitute` here needed no changes
//! of its own.

use super::super::{TermId, TermKind, TermManager};
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use smallvec::SmallVec;

// ===========================================================================
// Skolemization
// ===========================================================================

/// Skolemize a formula by eliminating existential quantifiers.
///
/// Skolemization replaces existentially quantified variables with fresh
/// function symbols (Skolem functions) closed over the enclosing universally
/// quantified variables, preserving equisatisfiability. For example:
/// - `∃x. P(x)` becomes `P(sk!0)`
/// - `∀y. ∃x. P(x, y)` becomes `∀y. P(sk!0(y), y)`
///
/// This mirrors the polarity-aware Skolemization tactic in
/// `crate::tactic::quantifier::SkolemizationTactic`. Two correctness
/// requirements are handled:
///
/// 1. **Polarity.** Only *effectively existential* quantifiers are
///    Skolemized: an `Exists` at positive polarity, or a `Forall` at
///    negative polarity (since `¬∀x.φ ≡ ∃x.¬φ`). Effectively *universal*
///    quantifiers keep their binder, and their bound variables become
///    arguments of any inner Skolem functions. Ignoring polarity would turn
///    `¬(∃x.P(x))` into `¬P(sk!0)`, flipping UNSAT into SAT.
/// 2. **Real argument sorts.** Skolem function arguments use the actual
///    sorts of the governing universal variables, not a fixed sort.
///
/// Skolemization only descends through Boolean structure (`Not`, `And`,
/// `Or`, `Implies`, `Ite` branches, and quantifiers). Sub-formulas at
/// genuinely mixed polarity (an `Ite` condition, a Boolean equality) are
/// left untouched rather than Skolemized unsoundly.
///
/// For a single formula this is sufficient. To Skolemize *several*
/// assertions belonging to the same goal, use [`skolemize_with_counter`]
/// with one shared counter across all calls: resetting the counter per
/// assertion would let distinct existentials collide on the same Skolem
/// symbol (e.g. `{∃x.P(x), ∃x.¬P(x)}` collapsing to `{P(sk!0), ¬P(sk!0)}`,
/// flipping SAT into UNSAT).
pub fn skolemize(term_id: TermId, manager: &mut TermManager) -> TermId {
    let mut counter = 0;
    skolemize_with_counter(term_id, manager, &mut counter)
}

/// Skolemize a formula, threading an external fresh-name counter.
///
/// See [`skolemize`] for the algorithm. Use this entry point instead of
/// [`skolemize`] when Skolemizing multiple assertions of the same goal:
/// pass the same `counter` to every call so that Skolem symbols never
/// collide across assertions.
pub fn skolemize_with_counter(
    term_id: TermId,
    manager: &mut TermManager,
    counter: &mut usize,
) -> TermId {
    let governing: SmallVec<[(Spur, SortId); 4]> = SmallVec::new();
    skolemize_polar(term_id, manager, true, &governing, counter)
}

/// A pending sub-result of the iterative Skolemization walk: the original
/// (pre-Skolemization) `TermId` -- used as a defensive fallback, matching
/// this workspace's established "structurally unreachable, but no panic"
/// style elsewhere -- paired with the index into `results` where its
/// resolved value is written once known.
type SkolemPending = (TermId, usize);

fn skolem_read(results: &[Option<TermId>], pending: SkolemPending) -> TermId {
    results[pending.1].unwrap_or(pending.0)
}

fn skolem_alloc(results: &mut Vec<Option<TermId>>) -> usize {
    let slot = results.len();
    results.push(None);
    slot
}

/// One pending step of the iterative Skolemization walk, using an explicit
/// heap stack rather than native recursion (see the parent module's doc
/// comment). `governing` is passed by value, cloned once per
/// universal-quantifier scope entered -- mirroring the original recursive
/// `skolemize_universal`, which built its own extended `Vec` per call rather
/// than mutating a shared one (unlike `TermManager::free_vars`'s `bound`,
/// there is nothing to "restore" here: each scope's governing set is simply
/// longer than its parent's, and siblings never see each other's
/// extensions).
///
/// There is deliberately no cross-call memoization (the original had none
/// either): the same `(TermId, positive, governing)` triple reached via two
/// different DAG parents is Skolemized independently both times, each
/// consuming its own fresh Skolem-function names from `counter`. Adding
/// memoization here would change *which* Skolem symbols get shared between
/// two structurally-identical occurrences -- an observable behavior change,
/// not merely a performance one, so (unlike `cnf::distribute_or_over_and`)
/// this is intentionally left unmemoized to preserve the exact original
/// output.
enum SkolemStep {
    Expand {
        id: TermId,
        positive: bool,
        governing: SmallVec<[(Spur, SortId); 4]>,
        out: usize,
    },
    CombineNot {
        out: usize,
        arg: SkolemPending,
    },
    CombineAnd {
        out: usize,
        args: SmallVec<[SkolemPending; 4]>,
    },
    CombineOr {
        out: usize,
        args: SmallVec<[SkolemPending; 4]>,
    },
    CombineImplies {
        out: usize,
        lhs: SkolemPending,
        rhs: SkolemPending,
    },
    CombineIte {
        out: usize,
        cond: TermId,
        then_branch: SkolemPending,
        else_branch: SkolemPending,
    },
    /// Rebuild a kept (effectively-universal) quantifier from its
    /// Skolemized body -- `is_forall` picks which binder shape to rebuild
    /// as (see `skolemize_polar`'s doc comment: this need not match which
    /// binder the *input* term was, since a negated `Exists` is rebuilt as
    /// an `Exists`, only its body changes).
    CombineUniversal {
        out: usize,
        is_forall: bool,
        vars: SmallVec<[(Spur, SortId); 2]>,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
        body: SkolemPending,
    },
    /// After substituting an effectively-existential quantifier's fresh
    /// Skolem terms into its body (a direct, non-recursive call to
    /// `TermManager::substitute`), the substituted body must still be
    /// Skolemized recursively (nested existentials/universals inside it)
    /// at the same polarity and governing set. `resume` is that follow-up
    /// `Expand`'s own pending result, which becomes this step's answer too.
    CombineExistential {
        out: usize,
        resume: SkolemPending,
    },
}

/// Polarity-aware Skolemization.
///
/// `positive` is the polarity of `term_id` in the enclosing formula
/// (top-level formulas start positive). `governing` lists the
/// effectively-universal variables currently in scope, with their real
/// sorts, used as Skolem-function arguments.
fn skolemize_polar(
    term_id: TermId,
    manager: &mut TermManager,
    positive: bool,
    governing: &[(Spur, SortId)],
    counter: &mut usize,
) -> TermId {
    let mut results: Vec<Option<TermId>> = vec![None];
    let mut work: Vec<SkolemStep> = vec![SkolemStep::Expand {
        id: term_id,
        positive,
        governing: governing.iter().copied().collect(),
        out: 0,
    }];
    while let Some(step) = work.pop() {
        run_skolem_step(step, manager, counter, &mut results, &mut work);
    }
    results.first().copied().flatten().unwrap_or(term_id)
}

#[allow(clippy::too_many_lines)]
fn run_skolem_step(
    step: SkolemStep,
    manager: &mut TermManager,
    counter: &mut usize,
    results: &mut Vec<Option<TermId>>,
    work: &mut Vec<SkolemStep>,
) {
    match step {
        SkolemStep::Expand {
            id,
            positive,
            governing,
            out,
        } => {
            let kind = match manager.get(id) {
                Some(t) => t.kind.clone(),
                None => {
                    results[out] = Some(id);
                    return;
                }
            };

            match kind {
                TermKind::Not(arg) => {
                    let arg_slot = skolem_alloc(results);
                    work.push(SkolemStep::CombineNot {
                        out,
                        arg: (arg, arg_slot),
                    });
                    work.push(SkolemStep::Expand {
                        id: arg,
                        positive: !positive,
                        governing,
                        out: arg_slot,
                    });
                }
                TermKind::And(args) => {
                    let pending: SmallVec<[SkolemPending; 4]> =
                        args.iter().map(|&a| (a, skolem_alloc(results))).collect();
                    work.push(SkolemStep::CombineAnd {
                        out,
                        args: pending.clone(),
                    });
                    for &(a, slot) in pending.iter().rev() {
                        work.push(SkolemStep::Expand {
                            id: a,
                            positive,
                            governing: governing.clone(),
                            out: slot,
                        });
                    }
                }
                TermKind::Or(args) => {
                    let pending: SmallVec<[SkolemPending; 4]> =
                        args.iter().map(|&a| (a, skolem_alloc(results))).collect();
                    work.push(SkolemStep::CombineOr {
                        out,
                        args: pending.clone(),
                    });
                    for &(a, slot) in pending.iter().rev() {
                        work.push(SkolemStep::Expand {
                            id: a,
                            positive,
                            governing: governing.clone(),
                            out: slot,
                        });
                    }
                }
                TermKind::Implies(lhs, rhs) => {
                    let lhs_slot = skolem_alloc(results);
                    let rhs_slot = skolem_alloc(results);
                    work.push(SkolemStep::CombineImplies {
                        out,
                        lhs: (lhs, lhs_slot),
                        rhs: (rhs, rhs_slot),
                    });
                    work.push(SkolemStep::Expand {
                        id: rhs,
                        positive,
                        governing: governing.clone(),
                        out: rhs_slot,
                    });
                    work.push(SkolemStep::Expand {
                        id: lhs,
                        positive: !positive,
                        governing,
                        out: lhs_slot,
                    });
                }
                TermKind::Ite(cond, then_br, else_br) => {
                    // `cond` occurs at mixed polarity (both c and not c);
                    // left untouched. Both branches preserve the ambient
                    // polarity.
                    let then_slot = skolem_alloc(results);
                    let else_slot = skolem_alloc(results);
                    work.push(SkolemStep::CombineIte {
                        out,
                        cond,
                        then_branch: (then_br, then_slot),
                        else_branch: (else_br, else_slot),
                    });
                    work.push(SkolemStep::Expand {
                        id: else_br,
                        positive,
                        governing: governing.clone(),
                        out: else_slot,
                    });
                    work.push(SkolemStep::Expand {
                        id: then_br,
                        positive,
                        governing,
                        out: then_slot,
                    });
                }
                TermKind::Forall {
                    vars,
                    body,
                    patterns,
                } => {
                    if positive {
                        // Effectively universal: keep binder, extend
                        // governing set.
                        let mut gov = governing;
                        gov.extend(vars.iter().copied());
                        let body_slot = skolem_alloc(results);
                        work.push(SkolemStep::CombineUniversal {
                            out,
                            is_forall: true,
                            vars,
                            patterns,
                            body: (body, body_slot),
                        });
                        work.push(SkolemStep::Expand {
                            id: body,
                            positive,
                            governing: gov,
                            out: body_slot,
                        });
                    } else {
                        // not(forall x.phi) = exists x.not(phi): effectively
                        // existential, Skolemize it.
                        expand_existential(
                            &vars, body, positive, &governing, counter, manager, out, results, work,
                        );
                    }
                }
                TermKind::Exists {
                    vars,
                    body,
                    patterns,
                } => {
                    if positive {
                        // Effectively existential: Skolemize.
                        expand_existential(
                            &vars, body, positive, &governing, counter, manager, out, results, work,
                        );
                    } else {
                        // not(exists x.phi) = forall x.not(phi): effectively
                        // universal, keep binder.
                        let mut gov = governing;
                        gov.extend(vars.iter().copied());
                        let body_slot = skolem_alloc(results);
                        work.push(SkolemStep::CombineUniversal {
                            out,
                            is_forall: false,
                            vars,
                            patterns,
                            body: (body, body_slot),
                        });
                        work.push(SkolemStep::Expand {
                            id: body,
                            positive,
                            governing: gov,
                            out: body_slot,
                        });
                    }
                }
                // Atoms and mixed-polarity contexts (Boolean equalities,
                // arithmetic, uninterpreted applications, ...) are left
                // unchanged: they cannot be Skolemized soundly without
                // polarity information not available here, and leaving them
                // intact keeps the result equisatisfiable.
                _ => {
                    results[out] = Some(id);
                }
            }
        }

        SkolemStep::CombineNot { out, arg } => {
            let sk = skolem_read(results, arg);
            results[out] = Some(manager.mk_not(sk));
        }
        SkolemStep::CombineAnd { out, args } => {
            let sk_args: SmallVec<[TermId; 4]> =
                args.iter().map(|&p| skolem_read(results, p)).collect();
            results[out] = Some(manager.mk_and(sk_args));
        }
        SkolemStep::CombineOr { out, args } => {
            let sk_args: SmallVec<[TermId; 4]> =
                args.iter().map(|&p| skolem_read(results, p)).collect();
            results[out] = Some(manager.mk_or(sk_args));
        }
        SkolemStep::CombineImplies { out, lhs, rhs } => {
            let sk_lhs = skolem_read(results, lhs);
            let sk_rhs = skolem_read(results, rhs);
            results[out] = Some(manager.mk_implies(sk_lhs, sk_rhs));
        }
        SkolemStep::CombineIte {
            out,
            cond,
            then_branch,
            else_branch,
        } => {
            let sk_then = skolem_read(results, then_branch);
            let sk_else = skolem_read(results, else_branch);
            results[out] = Some(manager.mk_ite(cond, sk_then, sk_else));
        }
        SkolemStep::CombineUniversal {
            out,
            is_forall,
            vars,
            patterns,
            body,
        } => {
            let sk_body = skolem_read(results, body);
            let var_names: Vec<(String, SortId)> = vars
                .iter()
                .map(|(n, s)| (manager.resolve_str(*n).to_string(), *s))
                .collect();
            let var_strs: Vec<(&str, SortId)> = var_names
                .iter()
                .map(|(name, sort)| (name.as_str(), *sort))
                .collect();
            let result = if is_forall {
                manager.mk_forall_with_patterns(var_strs, sk_body, patterns)
            } else {
                manager.mk_exists_with_patterns(var_strs, sk_body, patterns)
            };
            results[out] = Some(result);
        }
        SkolemStep::CombineExistential { out, resume } => {
            results[out] = Some(skolem_read(results, resume));
        }
    }
}

/// Schedule Skolemization of an effectively-existential quantifier: build
/// its fresh-Skolem-term substitution, substitute it into the body (a
/// direct, non-recursive call -- `TermManager::substitute` is not itself
/// recursed into here), then schedule the substituted body for further
/// Skolemization at the same polarity/governing set (mirrors the original
/// recursive `skolemize_existential`'s own final call into
/// `skolemize_polar`).
#[allow(clippy::too_many_arguments)]
fn expand_existential(
    vars: &[(Spur, SortId)],
    body: TermId,
    positive: bool,
    governing: &SmallVec<[(Spur, SortId); 4]>,
    counter: &mut usize,
    manager: &mut TermManager,
    out: usize,
    results: &mut Vec<Option<TermId>>,
    work: &mut Vec<SkolemStep>,
) {
    let mut subst = FxHashMap::default();
    for &(var_name, var_sort) in vars {
        let var_name_str = manager.resolve_str(var_name).to_string();
        let var_id = manager.mk_var(&var_name_str, var_sort);
        let skolem_term = make_skolem_term(var_sort, manager, governing, counter);
        subst.insert(var_id, skolem_term);
    }

    // Substitution is capture-avoiding, so Skolem terms (closed over the
    // governing universals) cannot be captured by inner binders.
    let substituted = manager.substitute(body, &subst);

    let resume_slot = skolem_alloc(results);
    work.push(SkolemStep::CombineExistential {
        out,
        resume: (substituted, resume_slot),
    });
    work.push(SkolemStep::Expand {
        id: substituted,
        positive,
        governing: governing.clone(),
        out: resume_slot,
    });
}

/// Build the Skolem term for a variable of sort `var_sort`: a fresh
/// constant when no universals govern it, otherwise a fresh function
/// applied to the governing universal variables (using their real sorts).
fn make_skolem_term(
    var_sort: SortId,
    manager: &mut TermManager,
    governing: &[(Spur, SortId)],
    counter: &mut usize,
) -> TermId {
    let skolem_name = crate::smtlib::reserved_name("sk", &counter.to_string());
    *counter += 1;

    if governing.is_empty() {
        manager.mk_var(&skolem_name, var_sort)
    } else {
        let gov_names: Vec<_> = governing
            .iter()
            .map(|(n, s)| (manager.resolve_str(*n).to_string(), *s))
            .collect();
        let arg_ids: SmallVec<[TermId; 4]> = gov_names
            .iter()
            .map(|(name, sort)| manager.mk_var(name, *sort))
            .collect();
        manager.mk_apply(&skolem_name, arg_ids, var_sort)
    }
}

// ===========================================================================
// Universal-quantifier elimination
// ===========================================================================

/// Eliminate universal quantifiers by replacing them with fresh variables
///
/// This is useful for converting formulas to quantifier-free form when
/// combined with Skolemization. The formula should have existential
/// quantifiers eliminated first (via Skolemization).
pub fn eliminate_universal_quantifiers(term_id: TermId, manager: &mut TermManager) -> TermId {
    let mut counter = 0;
    eliminate_universal_impl(term_id, manager, &mut counter)
}

/// One pending step of the iterative universal-elimination walk, using an
/// explicit heap stack rather than native recursion (see the parent
/// module's doc comment). Shaped like [`SkolemStep`]'s
/// `Expand`/`CombineExistential` pair for the `Forall` case (build a
/// fresh-variable substitution, substitute non-recursively, then schedule
/// the result for further elimination), but simpler: there is no polarity
/// or governing set to thread, since this pass eliminates every `Forall`
/// unconditionally regardless of polarity (see this function's own doc
/// comment: it is only sound to use *after* Skolemization has already
/// removed every existential).
enum EliminateStep {
    Expand {
        id: TermId,
        out: usize,
    },
    CombineNot {
        out: usize,
        arg: (TermId, usize),
    },
    CombineAnd {
        out: usize,
        args: SmallVec<[(TermId, usize); 4]>,
    },
    CombineOr {
        out: usize,
        args: SmallVec<[(TermId, usize); 4]>,
    },
    CombineForall {
        out: usize,
        resume: (TermId, usize),
    },
}

fn eliminate_universal_impl(
    term_id: TermId,
    manager: &mut TermManager,
    counter: &mut usize,
) -> TermId {
    let mut results: Vec<Option<TermId>> = vec![None];
    let mut work: Vec<EliminateStep> = vec![EliminateStep::Expand {
        id: term_id,
        out: 0,
    }];
    while let Some(step) = work.pop() {
        run_eliminate_step(step, manager, counter, &mut results, &mut work);
    }
    results.first().copied().flatten().unwrap_or(term_id)
}

fn run_eliminate_step(
    step: EliminateStep,
    manager: &mut TermManager,
    counter: &mut usize,
    results: &mut Vec<Option<TermId>>,
    work: &mut Vec<EliminateStep>,
) {
    match step {
        EliminateStep::Expand { id, out } => {
            match manager.get(id).map(|t| t.kind.clone()) {
                None
                | Some(
                    TermKind::True
                    | TermKind::False
                    | TermKind::Var(_)
                    | TermKind::IntConst(_)
                    | TermKind::RealConst(_)
                    | TermKind::BitVecConst { .. },
                ) => {
                    results[out] = Some(id);
                }
                Some(TermKind::Not(arg)) => {
                    let arg_slot = skolem_alloc(results);
                    work.push(EliminateStep::CombineNot {
                        out,
                        arg: (arg, arg_slot),
                    });
                    work.push(EliminateStep::Expand {
                        id: arg,
                        out: arg_slot,
                    });
                }
                Some(TermKind::And(args)) => {
                    let pending: SmallVec<[(TermId, usize); 4]> =
                        args.iter().map(|&a| (a, skolem_alloc(results))).collect();
                    work.push(EliminateStep::CombineAnd {
                        out,
                        args: pending.clone(),
                    });
                    for &(a, slot) in pending.iter().rev() {
                        work.push(EliminateStep::Expand { id: a, out: slot });
                    }
                }
                Some(TermKind::Or(args)) => {
                    let pending: SmallVec<[(TermId, usize); 4]> =
                        args.iter().map(|&a| (a, skolem_alloc(results))).collect();
                    work.push(EliminateStep::CombineOr {
                        out,
                        args: pending.clone(),
                    });
                    for &(a, slot) in pending.iter().rev() {
                        work.push(EliminateStep::Expand { id: a, out: slot });
                    }
                }
                Some(TermKind::Forall { vars, body, .. }) => {
                    // Replace quantified vars with fresh constants.
                    let mut subst = FxHashMap::default();
                    for (var_name, var_sort) in &vars {
                        let fresh_name = format!("u_{counter}");
                        *counter += 1;
                        let var_name_str = manager.resolve_str(*var_name).to_string();
                        let var_id = manager.mk_var(&var_name_str, *var_sort);
                        let fresh_var = manager.mk_var(&fresh_name, *var_sort);
                        subst.insert(var_id, fresh_var);
                    }
                    let substituted = manager.substitute(body, &subst);
                    let resume_slot = skolem_alloc(results);
                    work.push(EliminateStep::CombineForall {
                        out,
                        resume: (substituted, resume_slot),
                    });
                    work.push(EliminateStep::Expand {
                        id: substituted,
                        out: resume_slot,
                    });
                }
                Some(_) => {
                    results[out] = Some(id);
                }
            }
        }
        EliminateStep::CombineNot { out, arg } => {
            let (fallback, slot) = arg;
            let sk = results[slot].unwrap_or(fallback);
            results[out] = Some(manager.mk_not(sk));
        }
        EliminateStep::CombineAnd { out, args } => {
            let new_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&(fallback, slot)| results[slot].unwrap_or(fallback))
                .collect();
            results[out] = Some(manager.mk_and(new_args));
        }
        EliminateStep::CombineOr { out, args } => {
            let new_args: SmallVec<[TermId; 4]> = args
                .iter()
                .map(|&(fallback, slot)| results[slot].unwrap_or(fallback))
                .collect();
            results[out] = Some(manager.mk_or(new_args));
        }
        EliminateStep::CombineForall { out, resume } => {
            let (fallback, slot) = resume;
            results[out] = Some(results[slot].unwrap_or(fallback));
        }
    }
}
