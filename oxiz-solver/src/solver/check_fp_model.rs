//! Concrete floating-point model construction and verification.
//!
//! The FP conflict checks in [`super::check_fp`] recognise a fixed catalogue of
//! *unsatisfiable* patterns; everything else falls through to the honesty gate
//! ([`Solver::fp_atoms_need_theory`]) and is reported `Unknown`, because the
//! CDCL(T) core has no complete FP theory wired in.
//!
//! This module closes that gap on the *satisfiable* side without ever
//! sacrificing soundness. It attempts to build a fully concrete assignment to
//! every floating-point term in the current assertion set and then evaluates
//! every assertion against it using the bit-exact [`Ieee754Engine`]. A `Sat`
//! verdict is returned **only** when a genuine, verified model witness exists:
//! every FP-sorted variable is pinned to a concrete IEEE-754 datum and every
//! assertion evaluates to `true`. There is no guessing — if any term cannot be
//! pinned, or any assertion cannot be evaluated, or the constructed model
//! fails to satisfy some assertion, the routine gives up (returns `false`) and
//! the caller falls back to the honest `Unknown`.
//!
//! The construction handles the common concrete-evaluation shapes that dominate
//! `QF_FP` benchmarks: variables pinned by `(= x <fp-expr>)`, arithmetic and
//! conversion operations computed through the engine, literal conversions from
//! `Real`/`Int` constants, and free variables whose only constraints are
//! special-value predicates (`fp.isNaN`, `fp.isInfinite`, …) for which a witness
//! special value is synthesised.
//!
//! # Recursion and memoization
//!
//! The three evaluators (`eval_real_core`, `eval_fp_core`, `eval_bool_core`)
//! used to be plain native recursion with no depth guard and no memo: a term
//! built through the `TermManager` builder API can nest arbitrarily deep, so
//! a sufficiently deep formula overflowed the native stack — a fatal,
//! `catch_unwind`-proof process abort — and a shared sub-DAG of the
//! hash-consed term graph was re-expanded once per path (`2^n` work for an
//! `n`-level doubling DAG).  Each evaluator is now an explicit-worklist walk
//! ([`Task`]) over a per-call `TermId`-keyed memo table.  A depth cap was
//! never an option: these functions return `Option` where `None` means "give
//! up on `Sat`", so a cap would be survivable — but the explicit stack makes
//! the walk total on every input, which is strictly better than refusing
//! deep-but-legitimate models.
//!
//! Memoizing on bare `TermId` is exact here because every evaluation is
//! context-free: no binder is ever entered (quantified formulas fall out of
//! the concrete fragment as `None` leaves), `values` is never mutated during
//! a single evaluation (only [`FpModelFinder::try_define`] writes it, after
//! the evaluator returned), and the engine's rounding mode cannot leak
//! between sub-evaluations — every rounding-sensitive operation
//! (`add`/`sub`/`mul`/`div`/`sqrt`/`fma`/`convert_format`) sets its mode from
//! the term's own `RoundingMode` immediately before executing, and the
//! remaining operations (`abs`/`neg`/`rem`/`min`/`max`/`classify` and the
//! comparisons) are exact and mode-independent per IEEE-754.  The memos are
//! created fresh per top-level call precisely because `values` *does* change
//! between calls (each `try_define` fixpoint round pins new variables).
//!
//! The worklist evaluates every operand of a node before combining, whereas
//! the recursive original short-circuited (`?`, and `And`/`Or` early
//! returns).  The produced values are identical — a definite `false`/`true`
//! still dominates an unknown sibling, and skipped operands were pure — only
//! the work profile differs, and the memo keeps that linear in DAG size.

#[allow(unused_imports)]
use crate::prelude::*;
use num_traits::ToPrimitive;
use oxiz_core::ast::{RoundingMode, TermId, TermKind, TermManager};
use oxiz_theories::fp::ieee754_full::{Ieee754Engine, convert_format};
use oxiz_theories::{FpFormat, FpRoundingMode, FpValue};

use super::Solver;

/// Positive/negative special-value predicate constraints gathered for a free
/// FP variable, used to synthesise a witness value.
#[derive(Default, Clone, Copy)]
struct PredicateFlags {
    want_nan: bool,
    want_inf: bool,
    want_zero: bool,
    want_normal: bool,
    want_subnormal: bool,
    want_positive: bool,
    want_negative: bool,
}

impl PredicateFlags {
    /// `true` when at least one positive class/sign constraint was recorded, so
    /// a meaningful witness can be synthesised (as opposed to leaving the
    /// variable to be defined by propagation).
    fn has_positive_constraint(&self) -> bool {
        self.want_nan
            || self.want_inf
            || self.want_zero
            || self.want_normal
            || self.want_subnormal
            || self.want_positive
            || self.want_negative
    }
}

/// One step of an explicit-worklist evaluation (see the module doc's
/// "Recursion and memoization" section): either visit a term — resolving it
/// immediately when it is a leaf of the walk, or scheduling its operands —
/// or apply a deferred operator whose operands have all been evaluated into
/// the memo table.
enum Task<Op> {
    /// Evaluate this term (memo-checked; may schedule an `Apply`).
    Visit(TermId),
    /// All operands are in the memo; combine them into this term's value.
    Apply(TermId, Op),
}

/// A deferred `Real`-arithmetic operator awaiting its operand values
/// (`eval_real_core`).  N-ary operand lists are copied out of the term so the
/// combine step never has to re-fetch and re-match the term kind.
enum RealOp {
    Neg(TermId),
    Sub(TermId, TermId),
    Div(TermId, TermId),
    Add(Vec<TermId>),
    Mul(Vec<TermId>),
}

/// A deferred FP operator awaiting its operand values (`eval_fp_core`).
/// Rounding-sensitive operators carry the term's own [`RoundingMode`], which
/// is applied to the engine immediately before the operation — exactly where
/// the recursive original applied it.
enum FpOp {
    Abs(TermId),
    Neg(TermId),
    Sqrt(RoundingMode, TermId),
    Add(RoundingMode, TermId, TermId),
    Sub(RoundingMode, TermId, TermId),
    Mul(RoundingMode, TermId, TermId),
    Div(RoundingMode, TermId, TermId),
    Rem(TermId, TermId),
    Fma(RoundingMode, TermId, TermId, TermId),
    Min(TermId, TermId),
    Max(TermId, TermId),
    /// `fp.to_fp` format conversion to `(eb, sb)`.
    Convert(RoundingMode, TermId, u32, u32),
    /// The value of an `ite` is the value of the branch its condition
    /// selected. Only that branch is scheduled — evaluating both and choosing
    /// afterwards would return `unknown` whenever the *un*taken branch happens
    /// to be unevaluable, which is exactly the common case for a symbolic
    /// rounding mode: only one of the five arms has its operands pinned.
    IteBranch(TermId),
}

/// A deferred Boolean connective awaiting its operand values
/// (`eval_bool_core`).
enum BoolOp {
    Not(TermId),
    And(Vec<TermId>),
    Or(Vec<TermId>),
    /// The Boolean-equality fallback of `Eq`, entered only after the FP
    /// interpretation of both sides failed.
    EqBool(TermId, TermId),
}

/// The rounding mode a term denotes under a candidate model, or `None` when
/// the term is not a rounding mode this finder has a value for.
///
/// The five modes are nullary `Var`s at the reserved `RoundingMode` sort,
/// interned under their canonical long names, so a mode *constant* resolves
/// from its name alone; any other `RoundingMode`-sorted variable resolves
/// from the finder's `rm_values` assignment.
fn canonical_rounding_mode(name: &str) -> Option<RoundingMode> {
    RoundingMode::ALL
        .into_iter()
        .find(|&rm| TermManager::rounding_mode_name(rm) == name)
}

/// Committed value of `term` in a memo table: `None` both for "evaluated to
/// unknown" and for "absent".  Operands of a scheduled `Apply` are always
/// present (their `Visit` completed first), so the collapse is unobservable
/// there; at the root it returns the honest `None`.
/// Cap on how deeply FP evaluation may re-enter Boolean evaluation through
/// an `ite` condition.
///
/// Each crossing costs one native stack frame (see
/// `FpModelFinder::ite_condition_depth`), so this is the one place in this
/// module where native depth is input-controlled. Formulas that arise in
/// practice have depth 1 — the rounding-mode case split puts no `ite` inside
/// its conditions — so the cap is generous and never fires on real input;
/// past it the evaluator answers the honest `unknown` and the solve reports
/// `Unknown` rather than risking an unrecoverable stack overflow.
const MAX_ITE_CONDITION_ALTERNATION: u32 = 64;

fn memoed<T: Copy>(memo: &FxHashMap<TermId, Option<T>>, term: TermId) -> Option<T> {
    memo.get(&term).copied().flatten()
}

/// Concrete FP model finder: assigns a bit-exact [`FpValue`] to every relevant
/// FP term and verifies the assertion set against the assignment.
struct FpModelFinder<'a> {
    manager: &'a TermManager,
    engine: Ieee754Engine,
    values: FxHashMap<TermId, FpValue>,
    /// Candidate assignment for the free `RoundingMode` variables — the
    /// symbolic-mode counterpart of `values`.
    rm_values: FxHashMap<TermId, RoundingMode>,
    /// How many times the walk has crossed from FP evaluation back into
    /// Boolean evaluation without returning.
    ///
    /// `eval_bool_core` calls `eval_fp_core` for its FP atoms, and the `Ite`
    /// arm of `eval_fp_core` calls `eval_bool_core` back for the branch
    /// condition. Each pair of crossings costs one native frame, so a formula
    /// that nests an FP-sorted `ite` inside another one's *condition*, over
    /// and over, would grow the native stack — the one thing this module's
    /// explicit worklists exist to prevent. The counter caps that alternation
    /// at [`MAX_ITE_CONDITION_ALTERNATION`] and answers the honest `unknown`
    /// beyond it. Ordinary formulas never come close: the rounding-mode case
    /// split puts no `ite` in its conditions at all, so its depth is 1.
    ite_condition_depth: u32,
}

impl<'a> FpModelFinder<'a> {
    fn new(manager: &'a TermManager) -> Self {
        Self {
            manager,
            engine: Ieee754Engine::new(),
            values: FxHashMap::default(),
            rm_values: FxHashMap::default(),
            ite_condition_depth: 0,
        }
    }

    /// Map the AST rounding mode to the engine's rounding-mode enum.
    fn engine_rm(rm: RoundingMode) -> FpRoundingMode {
        match rm {
            RoundingMode::RNE => FpRoundingMode::RoundNearestTiesToEven,
            RoundingMode::RNA => FpRoundingMode::RoundNearestTiesToAway,
            RoundingMode::RTP => FpRoundingMode::RoundTowardPositive,
            RoundingMode::RTN => FpRoundingMode::RoundTowardNegative,
            RoundingMode::RTZ => FpRoundingMode::RoundTowardZero,
        }
    }

    /// Return the IEEE-754 format of `term` from its sort, if it is FP-sorted.
    fn fp_format_of(&self, term: TermId) -> Option<FpFormat> {
        let td = self.manager.get(term)?;
        let sort = self.manager.sorts.get(td.sort)?;
        let (eb, sb) = sort.float_format()?;
        Some(FpFormat::new(eb, sb))
    }

    /// `true` iff `term` is a plain FP-sorted variable (an assignment target).
    fn is_fp_var(&self, term: TermId) -> bool {
        let Some(td) = self.manager.get(term) else {
            return false;
        };
        matches!(td.kind, TermKind::Var(_))
            && self
                .manager
                .sorts
                .get(td.sort)
                .is_some_and(|s| s.is_float())
    }

    /// The rounding mode `term` denotes, if any.
    ///
    /// A mode *constant* resolves from its canonical name; any other
    /// `RoundingMode`-sorted variable resolves from the candidate assignment
    /// in `rm_values`. Anything else — a differently-sorted term, or a mode
    /// variable this finder never pinned — is `None`, and the caller reports
    /// the honest unknown.
    fn resolve_rm(&self, term: TermId) -> Option<RoundingMode> {
        let td = self.manager.get(term)?;
        if td.sort != self.manager.sorts.rounding_mode_sort {
            return None;
        }
        let TermKind::Var(spur) = td.kind else {
            return None;
        };
        canonical_rounding_mode(self.manager.resolve_str(spur))
            .or_else(|| self.rm_values.get(&term).copied())
    }

    /// `true` iff `term` is a *free* rounding-mode variable: one at the
    /// `RoundingMode` sort that is not one of the five mode constants, and so
    /// is an assignment target rather than a value.
    fn is_free_rm_var(&self, term: TermId) -> bool {
        let Some(td) = self.manager.get(term) else {
            return false;
        };
        if td.sort != self.manager.sorts.rounding_mode_sort {
            return false;
        }
        match td.kind {
            TermKind::Var(spur) => {
                canonical_rounding_mode(self.manager.resolve_str(spur)).is_none()
            }
            _ => false,
        }
    }

    /// Evaluate a `Real`/`Int`-sorted term to an `f64`, following the small
    /// arithmetic shapes that appear as `(_ to_fp …)` operands.
    ///
    /// Explicit-worklist walk over a per-call memo — see the module doc's
    /// "Recursion and memoization" section.  Purely a function of the term
    /// (no engine, no `values`), so `TermId`-keyed memoization is trivially
    /// exact.
    fn eval_real_core(
        &self,
        root: TermId,
        memo: &mut FxHashMap<TermId, Option<f64>>,
    ) -> Option<f64> {
        let mut stack: Vec<Task<RealOp>> = vec![Task::Visit(root)];
        while let Some(task) = stack.pop() {
            match task {
                Task::Visit(term) => {
                    if memo.contains_key(&term) {
                        continue;
                    }
                    let Some(td) = self.manager.get(term) else {
                        memo.insert(term, None);
                        continue;
                    };
                    match &td.kind {
                        TermKind::RealConst(r) => {
                            memo.insert(term, r.to_f64());
                        }
                        TermKind::IntConst(n) => {
                            memo.insert(term, n.to_f64());
                        }
                        TermKind::Neg(a) => {
                            stack.push(Task::Apply(term, RealOp::Neg(*a)));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::Sub(a, b) => {
                            stack.push(Task::Apply(term, RealOp::Sub(*a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::Div(a, b) => {
                            stack.push(Task::Apply(term, RealOp::Div(*a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::Add(args) => {
                            stack.push(Task::Apply(term, RealOp::Add(args.to_vec())));
                            for &a in args.iter().rev() {
                                stack.push(Task::Visit(a));
                            }
                        }
                        TermKind::Mul(args) => {
                            stack.push(Task::Apply(term, RealOp::Mul(args.to_vec())));
                            for &a in args.iter().rev() {
                                stack.push(Task::Visit(a));
                            }
                        }
                        _ => {
                            memo.insert(term, None);
                        }
                    }
                }
                Task::Apply(term, op) => {
                    let value = match op {
                        RealOp::Neg(a) => memoed(memo, a).map(|v| -v),
                        RealOp::Sub(a, b) => match (memoed(memo, a), memoed(memo, b)) {
                            (Some(x), Some(y)) => Some(x - y),
                            _ => None,
                        },
                        // The original evaluated the denominator first and gave
                        // up on zero; combining after both operands are known
                        // produces the same value in every case (`-0.0 == 0.0`
                        // included).
                        RealOp::Div(a, b) => match (memoed(memo, a), memoed(memo, b)) {
                            (Some(x), Some(y)) if y != 0.0 => Some(x / y),
                            _ => None,
                        },
                        // Left-to-right accumulation preserved: float addition
                        // and multiplication are order-sensitive.
                        RealOp::Add(args) => {
                            let mut acc = Some(0.0);
                            for a in args {
                                acc = match (acc, memoed(memo, a)) {
                                    (Some(s), Some(v)) => Some(s + v),
                                    _ => None,
                                };
                                if acc.is_none() {
                                    break;
                                }
                            }
                            acc
                        }
                        RealOp::Mul(args) => {
                            let mut acc = Some(1.0);
                            for a in args {
                                acc = match (acc, memoed(memo, a)) {
                                    (Some(s), Some(v)) => Some(s * v),
                                    _ => None,
                                };
                                if acc.is_none() {
                                    break;
                                }
                            }
                            acc
                        }
                    };
                    memo.insert(term, value);
                }
            }
        }
        memoed(memo, root)
    }

    /// Round an `f64` value to `format` under rounding mode `rm`, producing a
    /// concrete [`FpValue`]. The `f64` is treated as an exact dyadic rational,
    /// so the conversion is a single correctly-rounded step for `Float64` (and
    /// nearest-then-round for narrower targets, matching the RNE conversions
    /// used by these benchmarks).
    fn real_to_fp(&mut self, value: f64, format: FpFormat, rm: FpRoundingMode) -> FpValue {
        self.engine.set_rounding_mode(rm);
        let as_f64 = FpValue::from_f64(value);
        convert_format(&mut self.engine, &as_f64, format)
    }

    /// Evaluate an FP-sorted term to a concrete [`FpValue`], if all of its
    /// leaves are already pinned. Returns `None` when any input is unknown or
    /// the operation is not (yet) supported by concrete evaluation.
    ///
    /// Thin entry point over [`Self::eval_fp_core`] with fresh per-call memos
    /// (see the module doc for why the memos must not outlive a call).
    /// Evaluate an `ite` condition reached from inside FP evaluation.
    ///
    /// Deliberately re-enters [`Self::eval_bool`] with fresh memos rather than
    /// threading the FP walk's memo tables through: evaluation is a pure
    /// function of the term and the current assignment, so a fresh memo can
    /// only cost work, never change an answer, and the conditions this is
    /// called on are tiny (`(= m RNE)`). What it *does* cost is one native
    /// frame per FP→Bool crossing, which is why the crossing is counted and
    /// capped — see `FpModelFinder::ite_condition_depth`.
    fn eval_ite_condition(&mut self, cond: TermId) -> Option<bool> {
        if self.ite_condition_depth >= MAX_ITE_CONDITION_ALTERNATION {
            return None;
        }
        self.ite_condition_depth += 1;
        let value = self.eval_bool(cond);
        self.ite_condition_depth -= 1;
        value
    }

    fn eval_fp(&mut self, term: TermId) -> Option<FpValue> {
        let mut fp_memo: FxHashMap<TermId, Option<FpValue>> = FxHashMap::default();
        let mut real_memo: FxHashMap<TermId, Option<f64>> = FxHashMap::default();
        self.eval_fp_core(term, &mut fp_memo, &mut real_memo)
    }

    /// Explicit-worklist body of [`Self::eval_fp`]; also driven by
    /// [`Self::eval_bool_core`] with the memos of the enclosing Boolean
    /// evaluation, so shared FP sub-DAGs are evaluated once per assertion
    /// rather than once per reference.
    fn eval_fp_core(
        &mut self,
        root: TermId,
        fp_memo: &mut FxHashMap<TermId, Option<FpValue>>,
        real_memo: &mut FxHashMap<TermId, Option<f64>>,
    ) -> Option<FpValue> {
        let mut stack: Vec<Task<FpOp>> = vec![Task::Visit(root)];
        while let Some(task) = stack.pop() {
            match task {
                Task::Visit(term) => {
                    if fp_memo.contains_key(&term) {
                        continue;
                    }
                    let Some(td) = self.manager.get(term) else {
                        fp_memo.insert(term, None);
                        continue;
                    };
                    match &td.kind {
                        TermKind::Var(_) => {
                            let value = self.values.get(&term).copied();
                            fp_memo.insert(term, value);
                        }
                        TermKind::FpLit {
                            sign,
                            exp,
                            sig,
                            eb,
                            sb,
                        } => {
                            let value = match (exp.to_u64(), sig.to_u64()) {
                                (Some(exponent), Some(significand)) => Some(FpValue {
                                    sign: *sign,
                                    exponent,
                                    significand,
                                    format: FpFormat::new(*eb, *sb),
                                }),
                                _ => None,
                            };
                            fp_memo.insert(term, value);
                        }
                        TermKind::FpPlusInfinity { eb, sb } => {
                            let value = Some(FpValue::pos_infinity(FpFormat::new(*eb, *sb)));
                            fp_memo.insert(term, value);
                        }
                        TermKind::FpMinusInfinity { eb, sb } => {
                            let value = Some(FpValue::neg_infinity(FpFormat::new(*eb, *sb)));
                            fp_memo.insert(term, value);
                        }
                        TermKind::FpPlusZero { eb, sb } => {
                            let value = Some(FpValue::pos_zero(FpFormat::new(*eb, *sb)));
                            fp_memo.insert(term, value);
                        }
                        TermKind::FpMinusZero { eb, sb } => {
                            let value = Some(FpValue::neg_zero(FpFormat::new(*eb, *sb)));
                            fp_memo.insert(term, value);
                        }
                        TermKind::FpNaN { eb, sb } => {
                            let value = Some(FpValue::nan(FpFormat::new(*eb, *sb)));
                            fp_memo.insert(term, value);
                        }
                        TermKind::FpAbs(a) => {
                            stack.push(Task::Apply(term, FpOp::Abs(*a)));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpNeg(a) => {
                            stack.push(Task::Apply(term, FpOp::Neg(*a)));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpSqrt(rm, a) => {
                            stack.push(Task::Apply(term, FpOp::Sqrt(*rm, *a)));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpAdd(rm, a, b) => {
                            stack.push(Task::Apply(term, FpOp::Add(*rm, *a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpSub(rm, a, b) => {
                            stack.push(Task::Apply(term, FpOp::Sub(*rm, *a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpMul(rm, a, b) => {
                            stack.push(Task::Apply(term, FpOp::Mul(*rm, *a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpDiv(rm, a, b) => {
                            stack.push(Task::Apply(term, FpOp::Div(*rm, *a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpRem(a, b) => {
                            stack.push(Task::Apply(term, FpOp::Rem(*a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpFma(rm, a, b, c) => {
                            stack.push(Task::Apply(term, FpOp::Fma(*rm, *a, *b, *c)));
                            stack.push(Task::Visit(*c));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpMin(a, b) => {
                            stack.push(Task::Apply(term, FpOp::Min(*a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpMax(a, b) => {
                            stack.push(Task::Apply(term, FpOp::Max(*a, *b)));
                            stack.push(Task::Visit(*b));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::FpToFp { rm, arg, eb, sb } => {
                            stack.push(Task::Apply(term, FpOp::Convert(*rm, *arg, *eb, *sb)));
                            stack.push(Task::Visit(*arg));
                        }
                        TermKind::RealToFp { rm, arg, eb, sb } => {
                            let (rm, arg, eb, sb) = (*rm, *arg, *eb, *sb);
                            let value = self.eval_real_core(arg, real_memo).map(|v| {
                                self.real_to_fp(v, FpFormat::new(eb, sb), Self::engine_rm(rm))
                            });
                            fp_memo.insert(term, value);
                        }
                        // An FP-sorted `ite`. `needs_ite_elimination` leaves
                        // these in place for the FP path precisely so this
                        // evaluator can take them, and taking them is what
                        // makes a *symbolic* rounding mode decidable: the
                        // parser compiles `(fp.add m x y)` into a five-way
                        // `ite` over `(= m RNE)` … `(= m RTZ)` whose leaves are
                        // ordinary `FpAdd` nodes.
                        //
                        // Only the selected branch is scheduled. In the
                        // rounding-mode split the four unselected arms are
                        // perfectly evaluable too, but in general an `ite` is
                        // exactly the construct whose untaken branch may be
                        // nonsense, and its value must not contaminate the
                        // result.
                        TermKind::Ite(cond, then_branch, else_branch) => {
                            let (cond, then_branch, else_branch) =
                                (*cond, *then_branch, *else_branch);
                            match self.eval_ite_condition(cond) {
                                Some(taken) => {
                                    let branch = if taken { then_branch } else { else_branch };
                                    stack.push(Task::Apply(term, FpOp::IteBranch(branch)));
                                    stack.push(Task::Visit(branch));
                                }
                                None => {
                                    fp_memo.insert(term, None);
                                }
                            }
                        }
                        _ => {
                            fp_memo.insert(term, None);
                        }
                    }
                }
                Task::Apply(term, op) => {
                    let value = self.apply_fp_op(op, fp_memo);
                    fp_memo.insert(term, value);
                }
            }
        }
        memoed(fp_memo, root)
    }

    /// Execute one deferred FP operator against operand values already in the
    /// memo, setting the engine rounding mode exactly where the recursive
    /// original did: immediately before each rounding-sensitive operation.
    /// The remaining operations (`abs`/`neg`/`rem`/`min`/`max`) are exact and
    /// mode-independent, so no mode is set for them — same as before.
    fn apply_fp_op(
        &mut self,
        op: FpOp,
        memo: &FxHashMap<TermId, Option<FpValue>>,
    ) -> Option<FpValue> {
        match op {
            FpOp::Abs(a) => memoed(memo, a).map(|v| self.engine.abs(&v)),
            FpOp::Neg(a) => memoed(memo, a).map(|v| self.engine.neg(&v)),
            FpOp::Sqrt(rm, a) => {
                let v = memoed(memo, a)?;
                self.engine.set_rounding_mode(Self::engine_rm(rm));
                Some(self.engine.sqrt(&v))
            }
            FpOp::Add(rm, a, b) => {
                let (va, vb) = (memoed(memo, a)?, memoed(memo, b)?);
                self.engine.set_rounding_mode(Self::engine_rm(rm));
                Some(self.engine.add(&va, &vb))
            }
            FpOp::Sub(rm, a, b) => {
                let (va, vb) = (memoed(memo, a)?, memoed(memo, b)?);
                self.engine.set_rounding_mode(Self::engine_rm(rm));
                Some(self.engine.sub(&va, &vb))
            }
            FpOp::Mul(rm, a, b) => {
                let (va, vb) = (memoed(memo, a)?, memoed(memo, b)?);
                self.engine.set_rounding_mode(Self::engine_rm(rm));
                Some(self.engine.mul(&va, &vb))
            }
            FpOp::Div(rm, a, b) => {
                let (va, vb) = (memoed(memo, a)?, memoed(memo, b)?);
                self.engine.set_rounding_mode(Self::engine_rm(rm));
                Some(self.engine.div(&va, &vb))
            }
            FpOp::Rem(a, b) => {
                let (va, vb) = (memoed(memo, a)?, memoed(memo, b)?);
                Some(self.engine.rem(&va, &vb))
            }
            FpOp::Fma(rm, a, b, c) => {
                let (va, vb, vc) = (memoed(memo, a)?, memoed(memo, b)?, memoed(memo, c)?);
                self.engine.set_rounding_mode(Self::engine_rm(rm));
                Some(self.engine.fma(&va, &vb, &vc))
            }
            FpOp::Min(a, b) => {
                let (va, vb) = (memoed(memo, a)?, memoed(memo, b)?);
                Some(self.engine.min(&va, &vb))
            }
            FpOp::Max(a, b) => {
                let (va, vb) = (memoed(memo, a)?, memoed(memo, b)?);
                Some(self.engine.max(&va, &vb))
            }
            FpOp::Convert(rm, arg, eb, sb) => {
                let v = memoed(memo, arg)?;
                self.engine.set_rounding_mode(Self::engine_rm(rm));
                Some(convert_format(&mut self.engine, &v, FpFormat::new(eb, sb)))
            }
            // The branch was chosen before it was scheduled, so its value
            // *is* the `ite`'s value.
            FpOp::IteBranch(branch) => memoed(memo, branch),
        }
    }

    /// Structural (SMT-LIB `=`) equality on two concrete FP data: all NaNs are
    /// equal to one another; otherwise the encodings must match bit-for-bit, so
    /// `+0` and `-0` are distinct.
    fn fp_structural_eq(&self, a: &FpValue, b: &FpValue) -> bool {
        let ca = self.engine.classify(a);
        let cb = self.engine.classify(b);
        if ca.is_nan() || cb.is_nan() {
            return ca.is_nan() && cb.is_nan();
        }
        a.sign == b.sign && a.exponent == b.exponent && a.significand == b.significand
    }

    /// `fp.isPositive`: not NaN and sign bit clear (`+0` counts as positive,
    /// matching Z3).
    fn is_positive(&self, v: &FpValue) -> bool {
        let c = self.engine.classify(v);
        !c.is_nan() && !c.sign()
    }

    /// `fp.isNegative`: not NaN and sign bit set (`-0` counts as negative,
    /// matching Z3).
    fn is_negative(&self, v: &FpValue) -> bool {
        let c = self.engine.classify(v);
        !c.is_nan() && c.sign()
    }

    /// Evaluate a Boolean-sorted term against the concrete FP assignment.
    /// Returns `None` when the term contains anything the concrete evaluator
    /// cannot decide (a non-FP atom, an unassigned variable, an unsupported
    /// operation, …), which forces the caller to give up on `Sat`.
    ///
    /// Thin entry point over [`Self::eval_bool_core`] with fresh per-call
    /// memos (see the module doc for why the memos must not outlive a call).
    fn eval_bool(&mut self, term: TermId) -> Option<bool> {
        let mut bool_memo: FxHashMap<TermId, Option<bool>> = FxHashMap::default();
        let mut fp_memo: FxHashMap<TermId, Option<FpValue>> = FxHashMap::default();
        let mut real_memo: FxHashMap<TermId, Option<f64>> = FxHashMap::default();
        self.eval_bool_core(term, &mut bool_memo, &mut fp_memo, &mut real_memo)
    }

    /// Explicit-worklist body of [`Self::eval_bool`].  FP atoms are resolved
    /// inline through [`Self::eval_fp_core`] (itself iterative, so the native
    /// call depth stays constant); only the Boolean connectives are deferred
    /// onto the worklist.
    fn eval_bool_core(
        &mut self,
        root: TermId,
        bool_memo: &mut FxHashMap<TermId, Option<bool>>,
        fp_memo: &mut FxHashMap<TermId, Option<FpValue>>,
        real_memo: &mut FxHashMap<TermId, Option<f64>>,
    ) -> Option<bool> {
        let mut stack: Vec<Task<BoolOp>> = vec![Task::Visit(root)];
        while let Some(task) = stack.pop() {
            match task {
                Task::Visit(term) => {
                    if bool_memo.contains_key(&term) {
                        continue;
                    }
                    let Some(td) = self.manager.get(term) else {
                        bool_memo.insert(term, None);
                        continue;
                    };
                    match &td.kind {
                        TermKind::True => {
                            bool_memo.insert(term, Some(true));
                        }
                        TermKind::False => {
                            bool_memo.insert(term, Some(false));
                        }
                        TermKind::Not(a) => {
                            stack.push(Task::Apply(term, BoolOp::Not(*a)));
                            stack.push(Task::Visit(*a));
                        }
                        TermKind::And(args) => {
                            stack.push(Task::Apply(term, BoolOp::And(args.to_vec())));
                            for &a in args.iter().rev() {
                                stack.push(Task::Visit(a));
                            }
                        }
                        TermKind::Or(args) => {
                            stack.push(Task::Apply(term, BoolOp::Or(args.to_vec())));
                            for &a in args.iter().rev() {
                                stack.push(Task::Visit(a));
                            }
                        }
                        TermKind::Eq(a, b) => {
                            let (a, b) = (*a, *b);
                            // Rounding modes first: they are nullary `Var`s, so
                            // neither the FP interpretation below nor the
                            // Boolean fallback can read them, and `(= m RNE)` —
                            // the condition of every arm of a symbolic
                            // rounding-mode case split — would evaluate to
                            // `unknown` and take the whole formula with it.
                            if let (Some(ra), Some(rb)) = (self.resolve_rm(a), self.resolve_rm(b)) {
                                bool_memo.insert(term, Some(ra == rb));
                                continue;
                            }
                            let va = self.eval_fp_core(a, fp_memo, real_memo);
                            let vb = self.eval_fp_core(b, fp_memo, real_memo);
                            if let (Some(va), Some(vb)) = (va, vb) {
                                let value = Some(self.fp_structural_eq(&va, &vb));
                                bool_memo.insert(term, value);
                            } else {
                                // Fall back to Boolean equality (e.g.
                                // `(= (fp.isNaN x) true)`).
                                stack.push(Task::Apply(term, BoolOp::EqBool(a, b)));
                                stack.push(Task::Visit(b));
                                stack.push(Task::Visit(a));
                            }
                        }
                        // `(distinct t1 .. tn)`, over rounding modes or over
                        // FP terms. The rounding-mode case is not optional
                        // decoration: the solver asserts
                        // `(distinct RNE RNA RTP RTN RTZ)` on every solve that
                        // mentions a mode, and `find` requires *every*
                        // assertion to evaluate to `true` — so without this arm
                        // no symbolic-rounding-mode formula could ever produce
                        // a verified model.
                        //
                        // SMT-LIB `distinct` is pairwise disequality under `=`,
                        // which on both sorts is structural identity (for FP:
                        // all NaNs equal, and `+0` differs from `-0` — the
                        // `fp.eq` numeric comparison is a different predicate
                        // and is handled by `TermKind::FpEq`).
                        TermKind::Distinct(args) => {
                            let args = args.to_vec();
                            let value = self.eval_distinct(&args, fp_memo, real_memo);
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpEq(a, b) => {
                            let value = self
                                .eval_fp_pair(*a, *b, fp_memo, real_memo)
                                .map(|(va, vb)| self.engine.eq(&va, &vb));
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpLt(a, b) => {
                            let value = self
                                .eval_fp_pair(*a, *b, fp_memo, real_memo)
                                .map(|(va, vb)| self.engine.lt(&va, &vb));
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpGt(a, b) => {
                            let value = self
                                .eval_fp_pair(*a, *b, fp_memo, real_memo)
                                .map(|(va, vb)| self.engine.gt(&va, &vb));
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpLeq(a, b) => {
                            let value = self
                                .eval_fp_pair(*a, *b, fp_memo, real_memo)
                                .map(|(va, vb)| self.engine.le(&va, &vb));
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpGeq(a, b) => {
                            let value = self
                                .eval_fp_pair(*a, *b, fp_memo, real_memo)
                                .map(|(va, vb)| self.engine.ge(&va, &vb));
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpIsNaN(a) => {
                            let value = self
                                .eval_fp_core(*a, fp_memo, real_memo)
                                .map(|v| self.engine.classify(&v).is_nan());
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpIsInfinite(a) => {
                            let value = self
                                .eval_fp_core(*a, fp_memo, real_memo)
                                .map(|v| self.engine.classify(&v).is_infinite());
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpIsZero(a) => {
                            let value = self
                                .eval_fp_core(*a, fp_memo, real_memo)
                                .map(|v| self.engine.classify(&v).is_zero());
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpIsNormal(a) => {
                            let value = self
                                .eval_fp_core(*a, fp_memo, real_memo)
                                .map(|v| self.engine.classify(&v).is_normal());
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpIsSubnormal(a) => {
                            let value = self
                                .eval_fp_core(*a, fp_memo, real_memo)
                                .map(|v| self.engine.classify(&v).is_subnormal());
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpIsPositive(a) => {
                            let value = self
                                .eval_fp_core(*a, fp_memo, real_memo)
                                .map(|v| self.is_positive(&v));
                            bool_memo.insert(term, value);
                        }
                        TermKind::FpIsNegative(a) => {
                            let value = self
                                .eval_fp_core(*a, fp_memo, real_memo)
                                .map(|v| self.is_negative(&v));
                            bool_memo.insert(term, value);
                        }
                        _ => {
                            bool_memo.insert(term, None);
                        }
                    }
                }
                Task::Apply(term, op) => {
                    let value = match op {
                        BoolOp::Not(a) => memoed(bool_memo, a).map(|b| !b),
                        // The recursive original returned early on a definite
                        // `false`; scanning the memoized children reproduces
                        // the same dominance (a definite `false` wins even
                        // when an earlier sibling was unknown).
                        BoolOp::And(args) => {
                            let mut result = Some(true);
                            for a in args {
                                match memoed(bool_memo, a) {
                                    Some(false) => {
                                        result = Some(false);
                                        break;
                                    }
                                    Some(true) => {}
                                    None => result = None,
                                }
                            }
                            result
                        }
                        BoolOp::Or(args) => {
                            let mut result = Some(false);
                            for a in args {
                                match memoed(bool_memo, a) {
                                    Some(true) => {
                                        result = Some(true);
                                        break;
                                    }
                                    Some(false) => {}
                                    None => result = None,
                                }
                            }
                            result
                        }
                        BoolOp::EqBool(a, b) => {
                            match (memoed(bool_memo, a), memoed(bool_memo, b)) {
                                (Some(ba), Some(bb)) => Some(ba == bb),
                                _ => None,
                            }
                        }
                    };
                    bool_memo.insert(term, value);
                }
            }
        }
        memoed(bool_memo, root)
    }

    /// Evaluate `(distinct t1 .. tn)`.
    ///
    /// Answers only when *every* operand resolves in one interpretation —
    /// all rounding modes, or all concrete FP values. A mixed or partly
    /// unresolved list is an honest `None`: a `false` could be justified by a
    /// single colliding pair, but a `true` requires knowing all of them, and
    /// reporting one without the other would make the answer depend on
    /// operand order.
    fn eval_distinct(
        &mut self,
        args: &[TermId],
        fp_memo: &mut FxHashMap<TermId, Option<FpValue>>,
        real_memo: &mut FxHashMap<TermId, Option<f64>>,
    ) -> Option<bool> {
        // Fewer than two operands are vacuously distinct.
        if args.len() < 2 {
            return Some(true);
        }
        if let Some(modes) = args
            .iter()
            .map(|&arg| self.resolve_rm(arg))
            .collect::<Option<Vec<_>>>()
        {
            return Some(
                modes
                    .iter()
                    .enumerate()
                    .all(|(i, a)| modes[i + 1..].iter().all(|b| a != b)),
            );
        }
        let mut values = Vec::with_capacity(args.len());
        for &arg in args {
            values.push(self.eval_fp_core(arg, fp_memo, real_memo)?);
        }
        Some(
            values
                .iter()
                .enumerate()
                .all(|(i, a)| values[i + 1..].iter().all(|b| !self.fp_structural_eq(a, b))),
        )
    }

    /// Evaluate both operands of a binary FP predicate; `None` when either
    /// side is unknown.  Mirrors the recursive original's `?`-chain: the
    /// second operand is not evaluated when the first fails.
    fn eval_fp_pair(
        &mut self,
        a: TermId,
        b: TermId,
        fp_memo: &mut FxHashMap<TermId, Option<FpValue>>,
        real_memo: &mut FxHashMap<TermId, Option<f64>>,
    ) -> Option<(FpValue, FpValue)> {
        let va = self.eval_fp_core(a, fp_memo, real_memo)?;
        let vb = self.eval_fp_core(b, fp_memo, real_memo)?;
        Some((va, vb))
    }

    /// Propagate definitional equalities `(= var <fp-expr>)` to a fixpoint,
    /// pinning each variable whose defining expression becomes evaluable.
    fn propagate(&mut self, assertions: &[TermId]) {
        loop {
            let mut changed = false;
            for &assertion in assertions {
                let Some(td) = self.manager.get(assertion) else {
                    continue;
                };
                if let TermKind::Eq(l, r) = &td.kind {
                    changed |= self.try_define(*l, *r);
                    changed |= self.try_define(*r, *l);
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// If `var` is an unpinned FP variable and `expr` is fully evaluable, pin
    /// `var` to the value of `expr`. Returns `true` when a new binding is made.
    fn try_define(&mut self, var: TermId, expr: TermId) -> bool {
        if self.values.contains_key(&var) || !self.is_fp_var(var) {
            return false;
        }
        if let Some(v) = self.eval_fp(expr) {
            self.values.insert(var, v);
            return true;
        }
        false
    }

    /// Collect the positive/negative special-value predicate constraints that
    /// the assertion set imposes directly on `var`, tracking Boolean polarity
    /// through `not`/`and`/`or`.
    fn collect_predicates(&self, var: TermId, assertions: &[TermId]) -> PredicateFlags {
        let mut flags = PredicateFlags::default();
        let mut stack: Vec<(TermId, bool)> = assertions.iter().map(|&a| (a, true)).collect();
        let mut visited: FxHashSet<(TermId, bool)> = FxHashSet::default();
        while let Some((term, positive)) = stack.pop() {
            if !visited.insert((term, positive)) {
                continue;
            }
            let Some(td) = self.manager.get(term) else {
                continue;
            };
            match &td.kind {
                TermKind::Not(a) => stack.push((*a, !positive)),
                TermKind::And(args) | TermKind::Or(args) => {
                    for &a in args {
                        stack.push((a, positive));
                    }
                }
                TermKind::FpIsNaN(a) if *a == var && positive => flags.want_nan = true,
                TermKind::FpIsInfinite(a) if *a == var && positive => flags.want_inf = true,
                TermKind::FpIsZero(a) if *a == var && positive => flags.want_zero = true,
                TermKind::FpIsNormal(a) if *a == var && positive => flags.want_normal = true,
                TermKind::FpIsSubnormal(a) if *a == var && positive => flags.want_subnormal = true,
                TermKind::FpIsPositive(a) if *a == var && positive => flags.want_positive = true,
                TermKind::FpIsNegative(a) if *a == var && positive => flags.want_negative = true,
                _ => {}
            }
        }
        flags
    }

    /// Synthesise a witness value for a free FP variable from its special-value
    /// predicate constraints. Returns `None` when no positive class/sign
    /// constraint applies (leaving the variable for propagation or the honest
    /// `Unknown` fallback). The verification pass is the ultimate soundness
    /// guard: a witness that fails to satisfy every assertion never yields
    /// `Sat`.
    fn synthesize_witness(&self, var: TermId, assertions: &[TermId]) -> Option<FpValue> {
        let format = self.fp_format_of(var)?;
        let flags = self.collect_predicates(var, assertions);
        if !flags.has_positive_constraint() {
            return None;
        }
        let sign = flags.want_negative;
        if flags.want_nan {
            return Some(FpValue::nan(format));
        }
        if flags.want_inf {
            return Some(if sign {
                FpValue::neg_infinity(format)
            } else {
                FpValue::pos_infinity(format)
            });
        }
        if flags.want_zero {
            return Some(if sign {
                FpValue::neg_zero(format)
            } else {
                FpValue::pos_zero(format)
            });
        }
        if flags.want_subnormal {
            // Smallest-magnitude subnormal: exponent field 0, significand 1.
            return Some(FpValue {
                sign,
                exponent: 0,
                significand: 1,
                format,
            });
        }
        // Normal / bare sign constraint: pick +/-1.0, which is `1.<zeros>` with
        // the biased exponent equal to the format bias.
        Some(FpValue {
            sign,
            exponent: format.bias() as u64,
            significand: 0,
            format,
        })
    }

    /// Collect every FP-sorted variable that appears in the assertion set.
    fn collect_fp_vars(&self, assertions: &[TermId]) -> Vec<TermId> {
        let mut vars = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = assertions.to_vec();
        while let Some(term) = stack.pop() {
            if !visited.insert(term) {
                continue;
            }
            if self.is_fp_var(term) && seen.insert(term) {
                vars.push(term);
            }
            if let Some(td) = self.manager.get(term) {
                super::term_walk::collect_structural_children(&td.kind, &mut stack);
            }
        }
        vars
    }

    /// Assign a rounding mode to every free `RoundingMode` variable in the
    /// assertion set.
    ///
    /// A variable that a top-level equality pins to a specific mode
    /// (`(assert (= m RTZ))`, or the same equality written the other way
    /// round) gets that mode; every other one gets `RNE`, IEEE 754's default.
    ///
    /// The default is a *guess*, and deliberately not a search. `find` ends by
    /// verifying every assertion against the completed assignment, so a wrong
    /// guess yields `false` — reported as `Unknown`, never as a wrong verdict.
    /// Enumerating all `5^k` assignments would buy completeness on formulas
    /// that pin a mode only indirectly, at a cost paid by every `QF_FP` solve;
    /// the honest partial answer is the better trade, and matches how this
    /// finder already treats FP variables it cannot witness.
    fn assign_rounding_modes(&mut self, assertions: &[TermId]) {
        let mut free: Vec<TermId> = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = assertions.to_vec();
        while let Some(term) = stack.pop() {
            if !visited.insert(term) {
                continue;
            }
            if self.is_free_rm_var(term) && seen.insert(term) {
                free.push(term);
            }
            if let Some(td) = self.manager.get(term) {
                super::term_walk::collect_structural_children(&td.kind, &mut stack);
            }
        }
        if free.is_empty() {
            return;
        }

        // Definitional pass: `(= m <mode>)` at the top level pins `m`.
        for &assertion in assertions {
            let Some(td) = self.manager.get(assertion) else {
                continue;
            };
            let TermKind::Eq(l, r) = td.kind else {
                continue;
            };
            for (var, value) in [(l, r), (r, l)] {
                if self.is_free_rm_var(var)
                    && !self.rm_values.contains_key(&var)
                    && let Some(mode) = self.resolve_rm(value)
                {
                    self.rm_values.insert(var, mode);
                }
            }
        }

        for var in free {
            self.rm_values.entry(var).or_insert(RoundingMode::RNE);
        }
    }

    /// Drive the full construct-and-verify pipeline, returning `true` only when
    /// a verified satisfying model exists.
    fn find(&mut self, assertions: &[TermId]) -> bool {
        // Rounding modes are assigned first: an FP-sorted `ite` over
        // `(= m RNE)` cannot be evaluated — and so `z = (fp.add m x y)` cannot
        // define `z` — until `m` has a value.
        self.assign_rounding_modes(assertions);
        let fp_vars = self.collect_fp_vars(assertions);
        // Definitional propagation, then witness the still-free predicate-
        // constrained variables, then propagate again (a witness can unlock
        // further definitions, e.g. `z = x + y` once `x` becomes a NaN).
        self.propagate(assertions);
        for &var in &fp_vars {
            if !self.values.contains_key(&var)
                && let Some(witness) = self.synthesize_witness(var, assertions)
            {
                self.values.insert(var, witness);
            }
        }
        self.propagate(assertions);
        // Verify: every assertion must evaluate to a concrete `true`.
        for &assertion in assertions {
            if self.eval_bool(assertion) != Some(true) {
                return false;
            }
        }
        true
    }
}

impl Solver {
    /// Attempt to prove the current assertion set satisfiable by constructing
    /// and verifying a concrete floating-point model.
    ///
    /// Returns `true` **only** when a genuine model witness is found: every
    /// FP-sorted variable is pinned to a concrete IEEE-754 value and every
    /// assertion evaluates to `true` under the bit-exact engine. This is sound
    /// — it never reports a satisfiable verdict for an unsatisfiable formula,
    /// and it declines (returns `false`) whenever any assertion falls outside
    /// the concrete-evaluation fragment, letting the caller answer `Unknown`.
    pub(super) fn try_fp_model_sat(&self, manager: &TermManager) -> bool {
        if self.assertions.is_empty() {
            return false;
        }
        let mut finder = FpModelFinder::new(manager);
        finder.find(&self.assertions)
    }
}

/// Regression tests for the explicit-worklist conversion of the three
/// concrete-model evaluators (`eval_real_core` / `eval_fp_core` /
/// `eval_bool_core`) — see the module doc's "Recursion and memoization"
/// section for the rationale.  Deep-nesting tests run on a deliberately small
/// (128 KiB) thread stack: a native stack overflow is a fatal abort that
/// `catch_unwind` cannot intercept, so returning at all is the proof, and the
/// pinned values additionally prove the walk was complete and correct.
#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::Rational64;

    /// `eval_real_core` with a fresh memo, mirroring how `eval_fp_core`
    /// drives it.
    fn eval_real(finder: &FpModelFinder<'_>, term: TermId) -> Option<f64> {
        let mut memo: FxHashMap<TermId, Option<f64>> = FxHashMap::default();
        finder.eval_real_core(term, &mut memo)
    }

    // -----------------------------------------------------------------------
    // Semantic pins: small inputs with known-exact answers.
    // -----------------------------------------------------------------------

    #[test]
    fn eval_real_pins_the_arithmetic_shapes() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let three_halves = manager.mk_real(Rational64::new(3, 2));
        let five_halves = manager.mk_real(Rational64::new(5, 2));
        let two = manager.mk_int(2);
        let one = manager.mk_int(1);
        let sum = manager.mk_add(vec![three_halves, five_halves]); // 4.0
        let product = manager.mk_mul(vec![sum, two]); // 8.0
        let diff = manager.mk_sub(product, one); // 7.0
        let neg = manager.mk_neg(diff); // -7.0
        let quotient = manager.mk_div(neg, two); // -3.5
        let zero = manager.mk_int(0);
        let div_by_zero = manager.mk_div(three_halves, zero);
        let opaque = manager.mk_var("p", bool_sort);
        let sum_with_opaque = manager.mk_add(vec![three_halves, opaque]);

        let finder = FpModelFinder::new(&manager);
        assert_eq!(eval_real(&finder, quotient), Some(-3.5));
        assert_eq!(
            eval_real(&finder, div_by_zero),
            None,
            "division by zero must stay an honest unknown"
        );
        assert_eq!(
            eval_real(&finder, sum_with_opaque),
            None,
            "an unsupported operand makes the whole sum unknown"
        );
    }

    #[test]
    fn eval_fp_pins_arithmetic_over_pinned_variables() {
        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let x = manager.mk_var("x", fp_sort);
        let y = manager.mk_var("y", fp_sort);
        let add = manager.mk_fp_add(RoundingMode::RNE, x, y);
        let neg = manager.mk_fp_neg(add);
        let abs = manager.mk_fp_abs(neg);
        let free = manager.mk_var("free", fp_sort);
        let add_free = manager.mk_fp_add(RoundingMode::RNE, x, free);

        let mut finder = FpModelFinder::new(&manager);
        finder.values.insert(x, FpValue::from_f64(1.5));
        finder.values.insert(y, FpValue::from_f64(2.25));

        assert_eq!(finder.eval_fp(add), Some(FpValue::from_f64(3.75)));
        assert_eq!(finder.eval_fp(neg), Some(FpValue::from_f64(-3.75)));
        assert_eq!(finder.eval_fp(abs), Some(FpValue::from_f64(3.75)));
        assert_eq!(
            finder.eval_fp(free),
            None,
            "an unpinned variable stays unknown"
        );
        assert_eq!(finder.eval_fp(add_free), None, "unknown operands propagate");
    }

    /// Each conversion applies the rounding mode of its *own* term: `1/3` is
    /// not representable in float32, so rounding toward positive and toward
    /// zero must land on adjacent, strictly ordered values.  This pins that
    /// the worklist applies modes immediately before each operation instead
    /// of letting one leak across scheduled operations.
    #[test]
    fn eval_fp_applies_each_conversions_own_rounding_mode() {
        let mut manager = TermManager::new();
        let third = manager.mk_real(Rational64::new(1, 3));
        let rtp = manager.mk_real_to_fp(RoundingMode::RTP, third, 8, 24);
        let rtz = manager.mk_real_to_fp(RoundingMode::RTZ, third, 8, 24);

        let mut finder = FpModelFinder::new(&manager);
        let vp = finder.eval_fp(rtp).and_then(|v| v.to_f32());
        let vz = finder.eval_fp(rtz).and_then(|v| v.to_f32());
        assert!(
            vp.zip(vz).is_some_and(|(p, z)| p > z),
            "RTP must round 1/3 up and RTZ down; got {vp:?} vs {vz:?}"
        );
    }

    #[test]
    fn eval_bool_pins_connective_semantics_including_unknowns() {
        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let bool_sort = manager.sorts.bool_sort;
        let x = manager.mk_var("x", fp_sort);
        let p = manager.mk_var("p", bool_sort);
        let is_zero = manager.mk_fp_is_zero(x);
        let is_neg = manager.mk_fp_is_negative(x);
        let not_zero = manager.mk_not(is_zero);
        let and_unknown = manager.mk_and(vec![is_zero, p]);
        let and_false = manager.mk_and(vec![p, is_neg]);
        let or_unknown = manager.mk_or(vec![is_neg, p]);
        let or_true = manager.mk_or(vec![p, is_zero]);

        let mut finder = FpModelFinder::new(&manager);
        finder
            .values
            .insert(x, FpValue::pos_zero(FpFormat::new(11, 53)));

        assert_eq!(finder.eval_bool(is_zero), Some(true));
        assert_eq!(finder.eval_bool(is_neg), Some(false), "+0 is not negative");
        assert_eq!(finder.eval_bool(not_zero), Some(false));
        assert_eq!(finder.eval_bool(and_unknown), None);
        assert_eq!(
            finder.eval_bool(and_false),
            Some(false),
            "a definite false dominates an unknown sibling"
        );
        assert_eq!(finder.eval_bool(or_unknown), None);
        assert_eq!(
            finder.eval_bool(or_true),
            Some(true),
            "a definite true dominates an unknown sibling"
        );
    }

    #[test]
    fn eval_bool_eq_uses_fp_first_then_boolean_fallback() {
        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let x = manager.mk_var("x", fp_sort);
        let y = manager.mk_var("y", fp_sort);
        let n1 = manager.mk_var("n1", fp_sort);
        let n2 = manager.mk_var("n2", fp_sort);
        let eq_fp = manager.mk_eq(x, y);
        let eq_nan = manager.mk_eq(n1, n2);
        let is_zero_x = manager.mk_fp_is_zero(x);
        let t = manager.mk_true();
        let eq_bool = manager.mk_eq(t, is_zero_x);

        let mut finder = FpModelFinder::new(&manager);
        finder.values.insert(x, FpValue::from_f64(2.5));
        finder.values.insert(y, FpValue::from_f64(2.5));
        let format = FpFormat::new(11, 53);
        finder.values.insert(n1, FpValue::nan(format));
        finder.values.insert(n2, FpValue::nan(format));

        assert_eq!(
            finder.eval_bool(eq_fp),
            Some(true),
            "FP structural-equality path"
        );
        assert_eq!(
            finder.eval_bool(eq_nan),
            Some(true),
            "all NaNs are structurally equal under SMT-LIB `=`"
        );
        assert_eq!(
            finder.eval_bool(eq_bool),
            Some(false),
            "Boolean fallback: (= true (fp.isZero 2.5)) is false"
        );
    }

    /// End-to-end pin of the construct-and-verify pipeline over the new
    /// evaluators: a witness is found and verified for a satisfiable set, and
    /// the finder declines (never guesses) when its witness fails to verify.
    #[test]
    fn find_verifies_a_witness_and_declines_a_contradiction() {
        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let x = manager.mk_var("x", fp_sort);
        let is_zero = manager.mk_fp_is_zero(x);
        let is_neg = manager.mk_fp_is_negative(x);
        let is_normal = manager.mk_fp_is_normal(x);

        let mut finder = FpModelFinder::new(&manager);
        assert!(
            finder.find(&[is_zero, is_neg]),
            "-0 witnesses isZero ∧ isNegative"
        );

        let mut finder = FpModelFinder::new(&manager);
        assert!(
            !finder.find(&[is_zero, is_normal]),
            "the zero witness cannot satisfy isNormal; the finder must decline"
        );
    }

    // -----------------------------------------------------------------------
    // Shared-DAG regressions: doubling DAGs that were exponential without the
    // per-call memo must now be linear.
    // -----------------------------------------------------------------------

    #[test]
    fn eval_real_shared_add_dag_is_linear_not_exponential() {
        const LEVELS: i32 = 60;
        let mut manager = TermManager::new();
        let mut term = manager.mk_real(Rational64::new(3, 2));
        for _ in 0..LEVELS {
            term = manager.mk_add(vec![term, term]);
        }

        let finder = FpModelFinder::new(&manager);
        assert_eq!(
            eval_real(&finder, term),
            Some(1.5 * (2.0f64).powi(LEVELS)),
            "60 exact doublings of 1.5"
        );
    }

    #[test]
    fn eval_fp_shared_add_dag_is_linear_not_exponential() {
        const LEVELS: i32 = 60;
        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let x = manager.mk_var("x", fp_sort);
        let mut term = x;
        for _ in 0..LEVELS {
            term = manager.mk_fp_add(RoundingMode::RNE, term, term);
        }

        let mut finder = FpModelFinder::new(&manager);
        finder.values.insert(x, FpValue::from_f64(1.5));
        assert_eq!(
            finder.eval_fp(term),
            Some(FpValue::from_f64(1.5 * (2.0f64).powi(LEVELS))),
            "60 exact doublings of 1.5 through the IEEE engine"
        );
    }

    #[test]
    fn eval_bool_shared_and_dag_is_linear_not_exponential() {
        const LEVELS: usize = 60;
        let mut manager = TermManager::new();
        let fp_sort = manager.sorts.float_sort(11, 53);
        let bool_sort = manager.sorts.bool_sort;
        let x = manager.mk_var("x", fp_sort);
        let mut term = manager.mk_fp_is_zero(x);
        for _ in 0..LEVELS {
            // Raw intern: `mk_and` flattens nested `And`s, which would defeat
            // the sharing this test exists to exercise.
            term =
                manager.intern_term(TermKind::And([term, term].into_iter().collect()), bool_sort);
        }

        let mut finder = FpModelFinder::new(&manager);
        finder
            .values
            .insert(x, FpValue::pos_zero(FpFormat::new(11, 53)));
        assert_eq!(finder.eval_bool(term), Some(true));
    }

    // -----------------------------------------------------------------------
    // Deep-nesting regressions on a 128 KiB stack.
    //
    // Each `(STACK_SIZE, DEPTH)` pair below was scaled down from
    // (1 MiB, 100 000) by a factor of 8 on both sides.  What these tests pin
    // is the ~10 bytes of stack available per nesting level — no native frame
    // fits in that, so a recursive evaluator still dies — not the absolute
    // depth, and the smaller pair costs a 64th of the construction work.
    // Never raise one of the two without the other.
    // -----------------------------------------------------------------------

    #[test]
    fn eval_real_survives_a_deep_neg_chain_on_a_small_stack() {
        const STACK_SIZE: usize = 1 << 17; // 128 KiB
        const DEPTH: usize = 12_500;

        let handle = std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut manager = TermManager::new();
                let mut term = manager.mk_real(Rational64::new(3, 2));
                for _ in 0..DEPTH {
                    term = manager.mk_neg(term);
                }
                let finder = FpModelFinder::new(&manager);
                assert_eq!(
                    eval_real(&finder, term),
                    Some(1.5),
                    "an even number of negations returns the original value"
                );
            })
            .expect("spawning a 128 KiB-stack thread should succeed");
        handle
            .join()
            .expect("eval_real must return on a 128 KiB stack instead of overflowing");
    }

    #[test]
    fn eval_fp_survives_a_deep_conversion_chain_on_a_small_stack() {
        const STACK_SIZE: usize = 1 << 17; // 128 KiB
        const DEPTH: usize = 12_500;

        let handle = std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut manager = TermManager::new();
                let three_halves = manager.mk_real(Rational64::new(3, 2));
                let mut term = manager.mk_real_to_fp(RoundingMode::RNE, three_halves, 11, 53);
                for _ in 0..DEPTH {
                    term = manager.mk_fp_to_fp(RoundingMode::RNE, term, 11, 53);
                }
                let mut finder = FpModelFinder::new(&manager);
                assert_eq!(
                    finder.eval_fp(term),
                    Some(FpValue::from_f64(1.5)),
                    "identity conversions must preserve 1.5 through every level"
                );
            })
            .expect("spawning a 128 KiB-stack thread should succeed");
        handle
            .join()
            .expect("eval_fp must return on a 128 KiB stack instead of overflowing");
    }

    #[test]
    fn eval_bool_survives_a_deep_eq_chain_on_a_small_stack() {
        const STACK_SIZE: usize = 1 << 17; // 128 KiB
        const DEPTH: usize = 12_500;

        let handle = std::thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(|| {
                let mut manager = TermManager::new();
                let fp_sort = manager.sorts.float_sort(11, 53);
                let x = manager.mk_var("x", fp_sort);
                let t = manager.mk_true();
                let mut term = manager.mk_fp_is_zero(x);
                for _ in 0..DEPTH {
                    // `mk_eq` folds only constant pairs, so `(= true prev)`
                    // nests one level per iteration.
                    term = manager.mk_eq(t, term);
                }
                let mut finder = FpModelFinder::new(&manager);
                finder
                    .values
                    .insert(x, FpValue::pos_zero(FpFormat::new(11, 53)));
                assert_eq!(
                    finder.eval_bool(term),
                    Some(true),
                    "every level of the (= true …) chain evaluates to true"
                );
            })
            .expect("spawning a 128 KiB-stack thread should succeed");
        handle
            .join()
            .expect("eval_bool must return on a 128 KiB stack instead of overflowing");
    }
}
