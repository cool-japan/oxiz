//! The evaluator behind the model-verification soundness gate.
//!
//! [`Solver::model_refutes_assertions`] is the last thing standing between a
//! candidate model and a reported `Sat`: it re-evaluates every top-level
//! assertion under the freshly built model and refuses the verdict when the
//! model does not hold up.  [`Solver::eval_in_model_outcome`] is the evaluator
//! it runs.
//!
//! Two properties of that evaluator are load-bearing and are the reason it
//! lives in its own module rather than inline in the solver:
//!
//! * **It is iterative.**  Terms are evaluated by an explicit frame stack held
//!   on the heap, so a term nested a million levels deep costs a million `Vec`
//!   entries and a constant number of native stack frames.  The recursive
//!   version spent one native frame per nesting level of the assertion, and a
//!   library cannot know how much stack its caller has: an embedder's worker
//!   thread typically gets ~1 MiB, and a `fatal runtime error: stack overflow`
//!   there is a process abort, not a verdict the caller can handle.  Nothing
//!   bounds that depth from the outside either — the SMT-LIB parser's nesting
//!   limit does not apply to terms built through [`TermManager`]'s builder API,
//!   nor to the lemmas the array-axiom refinement loop synthesises.
//! * **Its arithmetic is checked.**  [`EvalVal::Num`] is a fixed-width
//!   `Rational64`, so a sum, difference, product or negation of two
//!   representable model values need not itself be representable.  Every
//!   arithmetic step therefore goes through `checked_*` and reports
//!   [`EvalOutcome::Unrepresentable`] rather than overflowing — see that
//!   variant's documentation for why the unchecked version was a soundness bug
//!   and not merely a robustness one.
//! * **It folds the DAG, not the tree.**  Terms are hash-consed, so a formula
//!   that mentions a shared subterm twice is one node with two parents; a
//!   per-call memo table in [`Solver::eval_in_model_outcome`] keeps the cost
//!   proportional to the number of distinct subterms.  Without it a chain of
//!   `n` shared doublings costs `2^n` visits, which is 22.8 s at `n = 28` and
//!   unbounded thereafter — see that table's comment for the measurements.
//!
//! Reference: Z3's `smt_model_checker.cpp` plays the same role — re-checking a
//! candidate model against the assertions before the verdict is trusted.

use super::model_eval_bv::{self, BvBinaryOp, BvCompareOp};
use super::types::Model;
use super::{ENCODE_DEPTH_LIMIT, EvalVal, Solver};
#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use num_rational::Rational64;
use num_traits::{CheckedAdd, CheckedMul, CheckedSub, ToPrimitive};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::hash_map::Entry;

/// A definite model value in hashable form, for the congruence half of
/// [`Solver::quantified_model_refutes_ground_assertions`].
///
/// [`EvalVal`] cannot serve directly: it holds a [`Rational64`] and derives
/// only `PartialEq`, while grouping applications by their arguments needs
/// `Eq + Hash`.  A `Rational64` is always kept in canonical form (reduced,
/// positive denominator), so its `(numer, denom)` pair is a faithful key —
/// equal pairs mean equal rationals and vice versa.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ArgKey {
    Bool(bool),
    /// Numerator and denominator of a canonical [`Rational64`].
    Num(i64, i64),
}

impl ArgKey {
    /// The key for a definite outcome; `None` for anything the model did not
    /// pin, which makes the caller skip the application entirely rather than
    /// key on an incomplete argument tuple.
    fn from_outcome(outcome: EvalOutcome) -> Option<Self> {
        match outcome {
            EvalOutcome::Value(EvalVal::Bool(b)) => Some(Self::Bool(b)),
            EvalOutcome::Value(EvalVal::Num(n)) => Some(Self::Num(*n.numer(), *n.denom())),
            // A bit-vector application value is deliberately not keyed.  This
            // is the *congruence* half of the quantified gate, which fires
            // when one function has two different values at the same argument
            // tuple; skipping bit-vector values keeps its behaviour exactly
            // what it was before bit-vectors became evaluable (a `BitVecConst`
            // witness read back `Undetermined` and produced `None` here), so
            // no quantified problem changes verdict because of this module.
            EvalOutcome::Value(EvalVal::Bv { .. }) => None,
            EvalOutcome::Undetermined | EvalOutcome::Unrepresentable => None,
        }
    }
}

/// What evaluating a term under a candidate model produced.
///
/// The two non-value answers are deliberately *not* the same thing, and the
/// gate treats them differently — see [`Solver::model_refutes_assertions`].
///
/// Not `Copy`: [`EvalVal::Bv`] owns a [`BigInt`] (see that variant for why
/// truncating it to keep `Copy` would be a soundness bug rather than a
/// convenience).  The two non-value variants carry nothing, so cloning an
/// `Undetermined` or an `Unrepresentable` — which is what the driver's carried
/// outcomes always are — allocates nothing.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum EvalOutcome {
    /// A concrete value the model determines.
    Value(EvalVal),
    /// The model does not determine this term's value: an unconstrained
    /// arithmetic variable, an opaque application the model does not pin, an
    /// operator the gate deliberately declines to decide (`distinct`, a
    /// numeric equality collision, a strict comparison at its boundary), an
    /// ill-typed operand, or a term past the evaluator's depth budget.
    ///
    /// This is the ordinary, expected answer for large parts of any real
    /// formula, and it must never by itself downgrade a `Sat`: the arithmetic
    /// solver represents disequalities by case splitting and strict bounds by
    /// a symbolic delta, so `Undetermined` is what a *perfectly good* model
    /// looks like through this evaluator.
    Undetermined,
    /// The model pinned every operand, but the evaluator's own fixed-width
    /// arithmetic could not represent an intermediate result.
    ///
    /// This is a statement about the evaluator, not about the model, and it is
    /// kept apart from [`EvalOutcome::Undetermined`] because the gate cannot
    /// afford to shrug it off.  Before the arithmetic here was checked, an
    /// overflowing assertion aborted a debug build outright and, in release,
    /// wrapped: `(< (+ 2^62 2^62) 0)` — false under the model, i.e. a genuine
    /// violation — wrapped to `i64::MIN < 0` and reported `true`, so the gate
    /// waved the bad model through as `Sat`.  The mirror image
    /// `(>= (+ 2^62 2^62) 0)` — true under the model — wrapped to `false` and
    /// refuted a perfectly good one.
    Unrepresentable,
}

impl EvalOutcome {
    /// The outcome for a term whose value the model does not determine.
    pub(super) const UNDETERMINED: EvalOutcome = EvalOutcome::Undetermined;

    /// A Boolean outcome.
    pub(super) fn boolean(value: bool) -> Self {
        EvalOutcome::Value(EvalVal::Bool(value))
    }

    /// A numeric outcome.
    fn number(value: Rational64) -> Self {
        EvalOutcome::Value(EvalVal::Num(value))
    }

    /// A bit-vector outcome.  `value` must already be reduced into
    /// `[0, 2^width)` — see [`EvalVal::Bv`]'s invariant; every producer in
    /// [`super::model_eval_bv`] establishes it.
    pub(super) fn bits(value: BigInt, width: u32) -> Self {
        EvalOutcome::Value(EvalVal::Bv { value, width })
    }

    /// The value this outcome carries, if any.
    fn value(self) -> Option<EvalVal> {
        match self {
            EvalOutcome::Value(v) => Some(v),
            _ => None,
        }
    }

    /// This outcome demoted to its non-value form: a `Value` the caller could
    /// not use (wrong type, or a sibling operand already failed) is no better
    /// than `Undetermined`, while `Unrepresentable` stays `Unrepresentable`.
    fn demote(self) -> Self {
        match self {
            EvalOutcome::Unrepresentable => EvalOutcome::Unrepresentable,
            _ => EvalOutcome::UNDETERMINED,
        }
    }

    /// The more cautious of two non-value outcomes: `Unrepresentable` wins,
    /// because it is the one the gate must act on.
    fn worse(self, other: Self) -> Self {
        if matches!(self, EvalOutcome::Unrepresentable)
            || matches!(other, EvalOutcome::Unrepresentable)
        {
            EvalOutcome::Unrepresentable
        } else {
            EvalOutcome::UNDETERMINED
        }
    }
}

/// A fixed-arity operator whose operands are evaluated left to right, stopping
/// at the first operand that does not produce a value.
///
/// The recursive version wrote each of these as `match (rec(a)?, rec(b)?)`, and
/// `?` on the left operand short-circuits before the right one is touched.
/// Evaluation here is pure — no cache, no model mutation — so the only thing
/// that short-circuit ever changed was the cost, and it is kept for that.
#[derive(Debug, Clone, Copy)]
enum EagerKind {
    /// `not`
    Not,
    /// `=`
    Eq,
    /// binary `-`
    Sub,
    /// unary `-`
    Neg,
    /// `<` (`strict_less`) or `>`; both soften at the boundary — see
    /// [`cmp_strict`].
    CmpStrict {
        /// `true` for `<`, `false` for `>`.
        less: bool,
    },
    /// `<=` (`or_equal_less`) or `>=`.
    CmpWeak {
        /// `true` for `<=`, `false` for `>=`.
        less: bool,
    },
    /// Pass the single operand's value straight through (`let` → its body).
    Identity,
    /// `bvnot`.
    BvNot,
    /// A binary bit-vector operator producing a bit-vector.
    BvBinary(BvBinaryOp),
    /// `(_ extract high low)`.
    BvExtract {
        /// High bit index, inclusive.
        high: u32,
        /// Low bit index, inclusive.
        low: u32,
    },
    /// One of the four bit-vector comparisons, producing a truth value.
    BvCompare(BvCompareOp),
}

/// How far an `ite` has got.
#[derive(Debug, Clone, Copy)]
enum IteState {
    /// The condition has not produced a value yet.
    Cond,
    /// The condition selected this branch; the other one is never evaluated.
    Branch(TermId),
}

/// How far an `=>` has got.
///
/// Not `Copy`: [`ImpliesState::ConsequentMayRescue`] carries an
/// [`EvalOutcome`], which stopped being `Copy` when [`EvalVal`] gained its
/// bit-vector variant.
#[derive(Debug, Clone)]
enum ImpliesState {
    /// The antecedent has not produced a value yet.
    Antecedent,
    /// The antecedent is `true`, so the implication *is* its consequent.
    ConsequentDecides,
    /// The antecedent produced no usable truth value.  Only a `true`
    /// consequent can still decide the implication; anything else leaves the
    /// carried non-value outcome.
    ConsequentMayRescue(EvalOutcome),
}

/// A pending operator, plus whatever state distinguishes "part-way through"
/// from "ready to combine".
#[derive(Debug)]
enum Op {
    /// A fixed-arity operator; only `operands[..arity]` is meaningful.
    Eager {
        /// Operand term ids in evaluation order.
        operands: [TermId; 2],
        /// How many of `operands` this operator takes.
        arity: u8,
        /// What to compute from the operand values.
        kind: EagerKind,
    },
    /// `and` (`conjunction = true`) or `or`, which stop at the first operand
    /// that decides the result but keep scanning the rest otherwise.
    Connective {
        /// Operand term ids in evaluation order.
        operands: SmallVec<[TermId; 4]>,
        /// `true` for `and`, `false` for `or`.
        conjunction: bool,
    },
    /// n-ary `+` (`product = false`) or `*`, folded into `acc` as each operand
    /// arrives so the fold order matches the recursive version exactly.
    Arith {
        /// Operand term ids in evaluation order.
        operands: SmallVec<[TermId; 4]>,
        /// `true` for `*`, `false` for `+`.
        product: bool,
        /// The running sum or product.
        acc: Rational64,
    },
    /// `ite`, which evaluates the condition and then only the taken branch.
    Ite {
        /// The condition.
        cond: TermId,
        /// Branch taken when the condition is `true`.
        then_branch: TermId,
        /// Branch taken when the condition is `false`.
        else_branch: TermId,
        /// How far the `ite` has got.
        state: IteState,
    },
    /// `distinct`, which this gate decides only when every operand folds to a
    /// bit-vector value of one width; see [`Frame::accept`]'s arm for why the
    /// other cases end the frame the moment they are seen.
    Distinct {
        /// Operand term ids in evaluation order.
        operands: SmallVec<[TermId; 4]>,
    },
    /// `=>`, whose antecedent decides whether the consequent is consulted at
    /// all and how its answer is used.
    Implies {
        /// The antecedent.
        antecedent: TermId,
        /// The consequent.
        consequent: TermId,
        /// How far the implication has got.
        state: ImpliesState,
    },
}

/// One entry of the driver's explicit stack.
#[derive(Debug)]
struct Frame {
    /// The operator and its per-operator progress.
    op: Op,
    /// How many operands have been consumed so far.
    filled: usize,
    /// Where this frame's operand values start in the driver's value stack.
    base: usize,
    /// The nesting depth of this frame's term, charged against
    /// [`ENCODE_DEPTH_LIMIT`].
    depth: u32,
    /// The most cautious non-value outcome any operand has produced, for the
    /// operators that keep scanning after one.
    carried: Option<EvalOutcome>,
}

/// What reading a term produced.
enum Opened {
    /// A leaf, an operator the gate declines to decide, or an unknown term id.
    Done(EvalOutcome),
    /// A compound term that now needs operands.
    Frame(Frame),
}

/// What the driver must do next for the frame on top of its stack.
enum Step {
    /// Evaluate this operand and hand the result back on the next turn.
    Need(TermId),
    /// The frame is finished; this is its result.
    Done(EvalOutcome),
}

impl Frame {
    /// A frame around an arbitrary [`Op`].  `base` is filled in by the driver
    /// when the frame is pushed, once the value stack's height is known.
    fn new(op: Op, depth: u32) -> Self {
        Self {
            op,
            filled: 0,
            base: 0,
            depth,
            carried: None,
        }
    }

    /// A frame for a one-operand operator.
    fn unary(a: TermId, kind: EagerKind, depth: u32) -> Self {
        Self::new(
            Op::Eager {
                operands: [a, a],
                arity: 1,
                kind,
            },
            depth,
        )
    }

    /// A frame for a two-operand operator, evaluated `a` then `b`.
    fn binary(a: TermId, b: TermId, kind: EagerKind, depth: u32) -> Self {
        Self::new(
            Op::Eager {
                operands: [a, b],
                arity: 2,
                kind,
            },
            depth,
        )
    }

    /// Fold `incoming` (when there is one) into the frame, then say what the
    /// frame needs next.
    fn advance(&mut self, values: &mut Vec<EvalVal>, incoming: Option<EvalOutcome>) -> Step {
        if let Some(result) = incoming
            && let Some(finished) = self.accept(values, result)
        {
            return Step::Done(finished);
        }
        self.request(values)
    }

    /// Fold one operand outcome into the frame.
    ///
    /// Returns `Some` when that operand ends the frame there and then — the
    /// short-circuiting cases, where the remaining operands must not be
    /// evaluated.
    fn accept(&mut self, values: &mut Vec<EvalVal>, result: EvalOutcome) -> Option<EvalOutcome> {
        match &mut self.op {
            // A fixed-arity operator gives up at the first operand it cannot
            // use, exactly as the recursive version's `rec(a)?` did.
            Op::Eager { .. } => match result {
                EvalOutcome::Value(value) => {
                    values.push(value);
                    self.filled += 1;
                    None
                }
                other => Some(other.demote()),
            },
            Op::Connective { conjunction, .. } => {
                let conjunction = *conjunction;
                match result {
                    // The deciding truth value ends the connective outright and
                    // outranks anything an earlier operand carried: `false ∧ ?`
                    // really is `false`.
                    EvalOutcome::Value(EvalVal::Bool(b)) if b != conjunction => {
                        Some(EvalOutcome::boolean(b))
                    }
                    EvalOutcome::Value(EvalVal::Bool(_)) => {
                        self.filled += 1;
                        None
                    }
                    // A non-Boolean operand, or one with no value: remember the
                    // most cautious answer seen and keep scanning, because a
                    // later operand may still decide the connective.
                    other => {
                        let demoted = other.demote();
                        let carried = match self.carried.take() {
                            Some(existing) => existing.worse(demoted),
                            None => demoted,
                        };
                        self.carried = Some(carried);
                        self.filled += 1;
                        None
                    }
                }
            }
            // `distinct` is decided ONLY when every operand is a definite
            // bit-vector value of one width (the width check is
            // `model_eval_bv::all_distinct`'s).  Anything else — a Boolean, a
            // number, an unpinned leaf, an arithmetic `Unrepresentable` — ends
            // the frame as `Undetermined` here and now, which is *exactly* the
            // answer this gate gave for every `distinct` before bit-vectors
            // became evaluable.  So no arithmetic `distinct` changes verdict,
            // and in particular an overflowing operand cannot turn a `distinct`
            // into an `Unrepresentable` refutation it never used to be.
            //
            // The arithmetic case must stay inconclusive on its own merits
            // too: the linear-arithmetic solver enforces a disequality by case
            // splitting rather than by pinning distinct witnesses, so
            // colliding values in its LP model are not evidence of anything.
            // A bit-vector model witness names every bit and carries no such
            // caveat.
            Op::Distinct { .. } => match result {
                EvalOutcome::Value(value @ EvalVal::Bv { .. }) => {
                    values.push(value);
                    self.filled += 1;
                    None
                }
                _ => Some(EvalOutcome::UNDETERMINED),
            },
            Op::Arith { product, acc, .. } => {
                let product = *product;
                let EvalOutcome::Value(EvalVal::Num(operand)) = result else {
                    return Some(result.demote());
                };
                // Checked, because `Rational64` is fixed width.  See
                // [`EvalOutcome::Unrepresentable`] for what the unchecked fold
                // did to the gate in each build profile.
                let folded = if product {
                    acc.checked_mul(&operand)
                } else {
                    acc.checked_add(&operand)
                };
                match folded {
                    Some(value) => *acc = value,
                    None => return Some(EvalOutcome::Unrepresentable),
                }
                self.filled += 1;
                None
            }
            Op::Ite {
                then_branch,
                else_branch,
                state,
                ..
            } => match state {
                IteState::Cond => match result {
                    EvalOutcome::Value(EvalVal::Bool(true)) => {
                        *state = IteState::Branch(*then_branch);
                        None
                    }
                    EvalOutcome::Value(EvalVal::Bool(false)) => {
                        *state = IteState::Branch(*else_branch);
                        None
                    }
                    other => Some(other.demote()),
                },
                // The taken branch's outcome *is* the `ite`'s outcome.
                IteState::Branch(_) => Some(result),
            },
            Op::Implies { state, .. } => match state {
                ImpliesState::Antecedent => match result {
                    // `false => _` is `true`, with no look at the consequent.
                    EvalOutcome::Value(EvalVal::Bool(false)) => Some(EvalOutcome::boolean(true)),
                    EvalOutcome::Value(EvalVal::Bool(true)) => {
                        *state = ImpliesState::ConsequentDecides;
                        None
                    }
                    other => {
                        *state = ImpliesState::ConsequentMayRescue(other.demote());
                        None
                    }
                },
                ImpliesState::ConsequentDecides => Some(match result {
                    EvalOutcome::Value(EvalVal::Bool(b)) => EvalOutcome::boolean(b),
                    other => other.demote(),
                }),
                // `_ => true` is `true` whatever the antecedent was.
                ImpliesState::ConsequentMayRescue(carried) => Some(match result {
                    EvalOutcome::Value(EvalVal::Bool(true)) => EvalOutcome::boolean(true),
                    // `carried` is always a demoted outcome, so it carries no
                    // payload and the clone allocates nothing.
                    other => carried.clone().worse(other.demote()),
                }),
            },
        }
    }

    /// The next operand to evaluate, or the frame's finished outcome.
    fn request(&mut self, values: &[EvalVal]) -> Step {
        match &self.op {
            Op::Eager {
                operands,
                arity,
                kind,
            } => {
                let arity = usize::from(*arity);
                if self.filled < arity {
                    return Step::Need(operands[self.filled]);
                }
                Step::Done(combine_eager(*kind, &values[self.base..]))
            }
            Op::Connective {
                operands,
                conjunction,
            } => {
                if self.filled < operands.len() {
                    Step::Need(operands[self.filled])
                } else {
                    // No operand decided the connective.  If every one agreed
                    // with it the connective holds; otherwise the most cautious
                    // outcome seen stands.
                    Step::Done(match self.carried.clone() {
                        Some(carried) => carried,
                        None => EvalOutcome::boolean(*conjunction),
                    })
                }
            }
            Op::Distinct { operands } => {
                if self.filled < operands.len() {
                    Step::Need(operands[self.filled])
                } else {
                    Step::Done(model_eval_bv::all_distinct(&values[self.base..]))
                }
            }
            Op::Arith { operands, acc, .. } => {
                if self.filled < operands.len() {
                    Step::Need(operands[self.filled])
                } else {
                    Step::Done(EvalOutcome::number(*acc))
                }
            }
            Op::Ite { cond, state, .. } => match state {
                IteState::Cond => Step::Need(*cond),
                IteState::Branch(branch) => Step::Need(*branch),
            },
            Op::Implies {
                antecedent,
                consequent,
                state,
            } => match state {
                ImpliesState::Antecedent => Step::Need(*antecedent),
                _ => Step::Need(*consequent),
            },
        }
    }
}

/// Combine the operand values of a fixed-arity operator.
///
/// `values` holds exactly the operands the frame collected, in order; the
/// driver only reaches here once every one of them produced a value.
fn combine_eager(kind: EagerKind, values: &[EvalVal]) -> EvalOutcome {
    match (kind, values) {
        (EagerKind::Not, [EvalVal::Bool(b)]) => EvalOutcome::boolean(!b),
        (EagerKind::Identity, [v]) => EvalOutcome::Value(v.clone()),
        (EagerKind::Eq, [a, b]) => combine_eq(a, b),
        (EagerKind::Sub, [EvalVal::Num(x), EvalVal::Num(y)]) => match x.checked_sub(y) {
            Some(d) => EvalOutcome::number(d),
            None => EvalOutcome::Unrepresentable,
        },
        // Negation is the one arithmetic operator that overflows on a *single*
        // operand: `-i64::MIN` has no `i64`.  Negating the numerator of an
        // already-reduced ratio keeps it reduced with a positive denominator,
        // so no re-reduction is needed.
        (EagerKind::Neg, [EvalVal::Num(n)]) => match n.numer().checked_neg() {
            Some(numer) => EvalOutcome::number(Rational64::new_raw(numer, *n.denom())),
            None => EvalOutcome::Unrepresentable,
        },
        (EagerKind::CmpStrict { less }, [EvalVal::Num(x), EvalVal::Num(y)]) => {
            cmp_strict(*x, *y, less)
        }
        (EagerKind::CmpWeak { less }, [EvalVal::Num(x), EvalVal::Num(y)]) => {
            EvalOutcome::boolean(if less { x <= y } else { x >= y })
        }
        // ---- bit-vectors ------------------------------------------------
        // No arithmetic is written here: `model_eval_bv` adapts to
        // `oxiz_core::ast::bv_fold`, the workspace's single definition of the
        // SMT-LIB folding rules.
        (EagerKind::BvNot, [EvalVal::Bv { value, width }]) => {
            model_eval_bv::complement(value, *width)
        }
        (
            EagerKind::BvBinary(op),
            [
                EvalVal::Bv {
                    value: left,
                    width: left_width,
                },
                EvalVal::Bv {
                    value: right,
                    width: right_width,
                },
            ],
        ) => model_eval_bv::binary(op, left, *left_width, right, *right_width),
        (EagerKind::BvExtract { high, low }, [EvalVal::Bv { value, width }]) => {
            model_eval_bv::extract(high, low, value, *width)
        }
        (
            EagerKind::BvCompare(op),
            [
                EvalVal::Bv {
                    value: left,
                    width: left_width,
                },
                EvalVal::Bv {
                    value: right,
                    width: right_width,
                },
            ],
        ) => model_eval_bv::compare(op, left, *left_width, right, *right_width),
        // Ill-typed operands (a Bool where a number was wanted, a number where
        // a bit-vector was wanted, or the other way round).  The gate has
        // nothing to say about such a term.
        _ => EvalOutcome::UNDETERMINED,
    }
}

/// Combine the operand values of `=`.
///
/// Booleans come straight from the SAT assignment and are reliable in both
/// directions.  Numeric equality is trustworthy only in the NEGATIVE direction:
/// distinct arithmetic values genuinely falsify the equality, but a *collision*
/// is not evidence — the LP model can assign two variables the same value even
/// when they were never asserted equal.  Reporting a collision as
/// `Undetermined` also keeps a negated equality (`distinct` / `not (= ..)`)
/// inconclusive there instead of a false violation.
fn combine_eq(a: &EvalVal, b: &EvalVal) -> EvalOutcome {
    match (a, b) {
        (EvalVal::Bool(x), EvalVal::Bool(y)) => EvalOutcome::boolean(x == y),
        (EvalVal::Num(x), EvalVal::Num(y)) => {
            if x == y {
                EvalOutcome::UNDETERMINED
            } else {
                EvalOutcome::boolean(false)
            }
        }
        // Bit-vectors are exact in BOTH directions, unlike the numeric arm
        // above: a bit-vector model witness is a `BitVecConst` naming every
        // bit, produced by a bit-blasted decision procedure, so a collision is
        // evidence where an LP collision is not.  Making `=` weaker than
        // `bvule` would also be incoherent, since `(= a b)` is
        // `(and (bvule a b) (bvule b a))`.  Unequal widths are an ill-sorted
        // term — the parser's problem, not the gate's.
        (
            EvalVal::Bv {
                value: x,
                width: x_width,
            },
            EvalVal::Bv {
                value: y,
                width: y_width,
            },
        ) => model_eval_bv::equal(x, *x_width, y, *y_width),
        _ => EvalOutcome::UNDETERMINED,
    }
}

/// Evaluate a STRICT comparison (`less` selects `<` over `>`).
///
/// STRICT comparisons are softened AT THE BOUNDARY: the arithmetic solver
/// represents `x > c` internally with a delta above `c` but `value()` reports
/// the boundary `c` itself, so a model value equal to the bound cannot
/// distinguish `x > c` (satisfiable) from a real violation.  Reporting
/// `Undetermined` there keeps the gate from falsely refuting a genuine
/// strict-inequality model; away from the boundary the comparison is concrete
/// and trustworthy.  Non-strict `<=` / `>=` have no such ambiguity.
fn cmp_strict(x: Rational64, y: Rational64, less: bool) -> EvalOutcome {
    if x == y {
        EvalOutcome::UNDETERMINED
    } else if less {
        EvalOutcome::boolean(x < y)
    } else {
        EvalOutcome::boolean(x > y)
    }
}

impl Solver {
    /// Soundness gate: may the freshly built model be reported as `Sat`?
    ///
    /// Returns `true` when it may not, for either of two reasons:
    ///
    /// * a top-level assertion evaluates to a concrete `false` under the model
    ///   — the model provably violates it (this is the case the name refers
    ///   to); or
    /// * a top-level assertion could not be evaluated *at all* because the
    ///   evaluator's fixed-width arithmetic could not represent an intermediate
    ///   result ([`EvalOutcome::Unrepresentable`]).
    ///
    /// The second reason is the conservative direction, and which direction is
    /// conservative here is worth spelling out.  A `false` answer from this
    /// function is consumed as "go ahead and report `Sat`"; a `true` answer
    /// costs only precision, because the caller then answers `Unknown`, which
    /// is always a legal verdict.  So when the gate cannot evaluate an
    /// assertion it must not conclude that the model satisfies it.  Doing the
    /// opposite is precisely the bug the checked arithmetic removes: an
    /// overflowing `(< (+ 2^62 2^62) 0)` wrapped to `true` in release, and the
    /// gate waved through a model that genuinely falsifies the assertion.
    ///
    /// [`EvalOutcome::Undetermined`] is emphatically *not* treated that way.
    /// The key to the gate's usefulness is where leaf numeric values come from:
    /// an Int/Real variable is read from the *arithmetic solver*
    /// (`arith.value`), which reports `None` for a variable it does not
    /// actually constrain.  That `None` propagates to `Undetermined` and never
    /// triggers a downgrade — so a `distinct` / comparison over variables that
    /// `build_model` merely *defaulted* to 0 (a genuinely satisfiable formula)
    /// is never mistaken for a violation.  Combined with the strict-inequality
    /// boundary softening (see [`cmp_strict`]), the gate only fires on a
    /// witness the theory genuinely determined yet the assignment falsifies —
    /// the signature of the SAT core committing an inconsistent trail (e.g. a
    /// clause reported satisfied whose every disjunct is false).  In that case
    /// the reported `Sat` is spurious and the solver answers `Unknown` instead.
    pub(super) fn model_refutes_assertions(&self, manager: &TermManager) -> bool {
        let Some(model) = self.model.as_ref() else {
            return false;
        };
        for &assertion in &self.assertions {
            match self.eval_in_model_outcome(assertion, model, manager, 0) {
                EvalOutcome::Value(EvalVal::Bool(false)) | EvalOutcome::Unrepresentable => {
                    return true;
                }
                _ => {}
            }
        }
        self.model_violates_negated_equality(manager)
    }

    /// The other half of the `#P2b-27` repair, on the gate's side: an
    /// assertion the gate cannot evaluate because a **Boolean variable in it
    /// has no model entry** is judged on the model the user will actually
    /// see, not skipped.
    ///
    /// [`Self::model_refutes_assertions`] skips an `Undetermined` assertion,
    /// and its doc gives the reason that is right for numeric variables: a
    /// variable the tableau never constrained reads back `Undetermined`, and
    /// a satisfiable `distinct` over two such variables must not be mistaken
    /// for a violation.  A Boolean variable is different.  Its value is not a
    /// witness some theory declined to pin; it is a truth value the
    /// published model *prints* — `(get-value)` and `(get-model)` fall back
    /// to `false` for a variable with no entry — so an assertion that is
    /// `Undetermined` only because such a variable has no entry may be
    /// falsified by the very model that is printed, and the gate has no
    /// opinion about it.  That is exactly how the free-bit-vector model of
    /// the pre-`#P2b-24` fuzz campaign got past the gate: the selector `p0`
    /// occurred in no outer clause, the SAT core never assigned it,
    /// `build_model` recorded nothing, every assertion above it evaluated
    /// `Undetermined`, and `(bvsgt (ite (distinct t6 #xc3) t9 v1) v0)` was
    /// published `sat` under a model that falsifies it.
    ///
    /// `build_model` now publishes the circuit's value for every selector
    /// (`BvSolver::bool_value`), so the common case never reaches here.
    /// What is left is a Boolean variable no theory decided at all, and the
    /// question is then whether the *printed* model satisfies the assertion:
    /// the missing variables are completed with the printed default and the
    /// assertion is evaluated once more.  `true` is accepted — `(or a (not
    /// a))`, whose `a` the encoder folds away, holds under any default, and
    /// the model counter enumerates exactly such tautologies — while `false`
    /// or a still-`Undetermined` answer is refused, so the answer is
    /// `unknown` rather than a `sat` with a model nobody vouched for.
    /// Conservative in the safe direction: a `true` here costs precision
    /// (the caller answers `Unknown`), never soundness.
    ///
    /// Returns `true` when some assertion evaluates `Undetermined`, has a
    /// Bool-sorted free variable with no model entry, and does not evaluate
    /// to `true` once those variables take the printed default.
    pub(super) fn model_leaves_a_boolean_undetermined(&self, manager: &TermManager) -> bool {
        let Some(model) = self.model.as_ref() else {
            return false;
        };
        let bool_sort = manager.sorts.bool_sort;
        let printed_default = manager.mk_false();
        for &assertion in &self.assertions {
            if !matches!(
                self.eval_in_model_outcome(assertion, model, manager, 0),
                EvalOutcome::Undetermined
            ) {
                continue;
            }
            let unassigned: Vec<TermId> = manager
                .free_vars(assertion)
                .into_iter()
                .filter(|&var| {
                    manager
                        .get(var)
                        .is_some_and(|t| t.sort == bool_sort && matches!(t.kind, TermKind::Var(_)))
                        && model.get(var).is_none()
                })
                .collect();
            if unassigned.is_empty() {
                continue;
            }
            let mut completed = model.clone();
            for var in unassigned {
                completed.set(var, printed_default);
            }
            if !matches!(
                self.eval_in_model_outcome(assertion, &completed, manager, 0),
                EvalOutcome::Value(EvalVal::Bool(true))
            ) {
                return true;
            }
        }
        false
    }

    /// The half of the gate that [`combine_eq`] structurally cannot see: a
    /// numeric equality atom the SAT core assigned **false** whose two sides
    /// the arithmetic model gives the **same** value.
    ///
    /// # Why `combine_eq` cannot do this itself
    ///
    /// [`combine_eq`] answers `Undetermined` — deliberately, and it must keep
    /// doing so — when two numeric operands are equal.  Its input is a pair of
    /// values and nothing else, and a *collision* in the LP model is not by
    /// itself evidence of anything: the tableau enforces a disequality by case
    /// splitting, not by pinning distinct witnesses, so two variables that were
    /// never asserted equal routinely share a value in a perfectly good model.
    /// Returning `Bool(true)` there would turn every satisfiable `distinct`
    /// into a spurious `Unknown`.
    ///
    /// The missing information is not in the values — it is the **trail
    /// polarity**.  If the core committed `(= a b)` to *false* and the model it
    /// then produced makes `a` and `b` equal, the assignment and the model
    /// contradict each other outright.  That is a definite refutation, not a
    /// coincidence, and it is exactly the witness the false-`sat` family left
    /// behind: the disequality never reached the tableau, so the LP was free to
    /// collide the two sides while the Boolean level believed they differed.
    ///
    /// This gate has the trail (`sat.model_value`) even though `combine_eq`
    /// does not, so the check lives here and `combine_eq` is left untouched.
    ///
    /// # Conservatism
    ///
    /// Both sides must evaluate to a *definite* number.  `Solver::arith.value`
    /// returns `None` for a term the tableau does not constrain, which reads
    /// back `Undetermined` and is skipped — so a variable `build_model` merely
    /// defaulted can never trigger this.  Only atoms the core actually assigned
    /// `False` are considered; `LBool::Undef` and `True` are ignored.
    fn model_violates_negated_equality(&self, manager: &TermManager) -> bool {
        use super::types::Constraint;
        use oxiz_sat::LBool;

        let Some(model) = self.model.as_ref() else {
            return false;
        };
        for (&var, constraint) in &self.var_to_constraint {
            let Constraint::Eq(lhs, rhs) = *constraint else {
                continue;
            };
            if self.sat.model_value(var) != LBool::False {
                continue;
            }
            // Numeric operands only: a Bool/BV/EUF equality has its own
            // theory and no arithmetic value to compare.
            let is_numeric = manager.get(lhs).is_some_and(|t| {
                t.sort == manager.sorts.int_sort || t.sort == manager.sorts.real_sort
            });
            if !is_numeric {
                continue;
            }
            let (
                EvalOutcome::Value(EvalVal::Num(lhs_val)),
                EvalOutcome::Value(EvalVal::Num(rhs_val)),
            ) = (
                self.eval_in_model_outcome(lhs, model, manager, 0),
                self.eval_in_model_outcome(rhs, model, manager, 0),
            )
            else {
                continue;
            };
            if lhs_val == rhs_val {
                return true;
            }
        }
        false
    }

    /// The quantified-search counterpart of [`Self::model_refutes_assertions`]:
    /// `true` iff the candidate model definitely falsifies a **ground**
    /// assertion, *or* is definitely not a function on the ground
    /// uninterpreted applications those assertions contain.
    ///
    /// Both jobs are one question — "is the thing we are about to report
    /// actually a model of the input's ground part?" — and both are needed;
    /// see the two sections below.
    ///
    /// # Why the quantified `Sat` exits need a gate at all
    ///
    /// The ground search path has run [`Self::model_refutes_assertions`] before
    /// reporting `Sat` since it was introduced; the quantified exits
    /// (`certify_quantified_sat`, `MBQIResult::Satisfied`,
    /// `MBQIResult::NoQuantifiers`) had none, and reported whatever the MBQI
    /// loop concluded.  That is not merely a missing belt-and-braces check: the
    /// numeric UF-argument purification that closes the ground false-`sat`
    /// (see `encode::numeric_purification`) deliberately *skips* the arguments
    /// of any function that also occurs under a binder, so on a formula like
    ///
    /// ```smtlib
    /// (assert (forall ((z Int)) (>= (f z) 0)))
    /// (assert (= x 2)) (assert (= y (+ x 1)))
    /// (assert (not (= (f y) (f 3))))
    /// ```
    ///
    /// the ground disequality's `f(3)` is never purified, arithmetic's entailed
    /// `y = 3` has nothing on the EUF side to attach to, the congruence
    /// `f(y) = f(3)` is never derived — and MBQI, which is only asked about the
    /// `forall`, happily reports its fixpoint.  The result was a wrong `sat` on
    /// a formula whose ground part alone is unsatisfiable.
    ///
    /// # Why only ground assertions, and why only a definite `false`
    ///
    /// Two deliberate restrictions, both of which make this gate *weaker* than
    /// the ground path's:
    ///
    /// * **Ground assertions only.**  A quantified assertion has no value in a
    ///   candidate model — the evaluator has no way to range over an infinite
    ///   domain — so it comes back [`EvalOutcome::Undetermined`] at best and
    ///   [`EvalOutcome::Unrepresentable`] at worst.  Deciding quantified
    ///   assertions is MBQI's job, not this gate's, and including them would
    ///   downgrade essentially every quantified problem to `Unknown`.
    /// * **Only [`EvalOutcome::Value`]`(`[`EvalVal::Bool`]`(false))` counts.**
    ///   Unlike the ground path, `Unrepresentable` is *not* treated as a
    ///   refutation here.  On the quantified path `build_model` produces an
    ///   explicitly *partial* model (see its call site in `check_core`), so an
    ///   assertion the evaluator cannot fold is the normal case rather than the
    ///   danger sign it is once a total ground model has been built.  Treating
    ///   it as refutation would trade a rare wrong `sat` for a routine
    ///   `Unknown`.  The narrower trigger still catches the shape above, where
    ///   the ground contradiction is fully concrete.
    ///
    /// # Why a congruence check is needed on top of that
    ///
    /// Evaluating the assertions alone does **not** catch the repro above, and
    /// the reason is worth stating precisely.  This evaluator treats an
    /// uninterpreted application as an *opaque leaf*, looked up in the model by
    /// term identity (see the fallback arm of [`Self::open_in_model`]).  On the
    /// repro it returns the model's `f(y) = 1` and `f(3) = 0`, so
    /// `(not (= (f y) (f 3)))` evaluates to `true` and nothing is refuted —
    /// even though the same model says `y = 3`, which makes `f(y)` and `f(3)`
    /// the *same application* and those two values a contradiction.
    ///
    /// The model is therefore not a function, and that — not any individual
    /// assertion's truth value — is the witness that it is not a model.  So the
    /// second half of this gate groups every ground application by its function
    /// symbol together with the *evaluated* values of its arguments, and
    /// refuses the verdict when one group holds two different definite values.
    ///
    /// Deliberately not fixed in [`Self::open_in_model`] itself: making the
    /// evaluator congruence-aware would change the answer of every `Apply` on
    /// the ground path too, which is protected by purification and has no such
    /// gap — a large behavioural change for no extra coverage here.
    ///
    /// Both halves stay conservative in the same direction: a group contributes
    /// nothing unless *every* argument evaluated to a definite value (a partial
    /// key would group applications that are not in fact congruent), and two
    /// members conflict only when both application values are definite and
    /// unequal.
    ///
    /// A `true` answer makes the caller report `Unknown` rather than `Sat`:
    /// always a legal verdict, and the honest one when the model on the table
    /// contradicts the input's own ground part.
    pub(super) fn quantified_model_refutes_ground_assertions(&self, manager: &TermManager) -> bool {
        let Some(model) = self.model.as_ref() else {
            return false;
        };

        // Applications already seen, keyed by function symbol plus evaluated
        // argument values.  The stored value is the first definite value found
        // for that key; a later member disagreeing with it is the violation.
        let mut congruence: FxHashMap<(Spur, SmallVec<[ArgKey; 4]>), ArgKey> = FxHashMap::default();

        for &assertion in &self.assertions {
            if super::encode::finite_expand::contains_quantifier(assertion, manager) {
                continue;
            }
            if matches!(
                self.eval_in_model_outcome(assertion, model, manager, 0),
                EvalOutcome::Value(EvalVal::Bool(false))
            ) {
                return true;
            }

            // `collect_ground_subterms` is the shared iterative (heap-stack)
            // walk and never descends into a binder, so everything it yields is
            // ground by construction — no native recursion, and no risk of
            // keying on a term containing a bound variable.
            for sub in super::encode::bool_euf_encoding::collect_ground_subterms(assertion, manager)
            {
                let Some(node) = manager.get(sub) else {
                    continue;
                };
                let TermKind::Apply { func, args } = &node.kind else {
                    continue;
                };
                let Some(arg_key) = args
                    .iter()
                    .map(|&a| {
                        ArgKey::from_outcome(self.eval_in_model_outcome(a, model, manager, 0))
                    })
                    .collect::<Option<SmallVec<[ArgKey; 4]>>>()
                else {
                    continue; // at least one argument is not pinned
                };
                let Some(value) =
                    ArgKey::from_outcome(self.eval_in_model_outcome(sub, model, manager, 0))
                else {
                    continue; // the application itself is not pinned
                };
                match congruence.entry((*func, arg_key)) {
                    Entry::Vacant(slot) => {
                        slot.insert(value);
                    }
                    Entry::Occupied(slot) => {
                        if *slot.get() != value {
                            // Same function, same argument values, two
                            // different results: not a function.
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// The value `model` determines for `term`, or `None` when it determines
    /// none.
    ///
    /// This is the *value-only* view of [`Self::eval_in_model_outcome`], for
    /// callers that act on a concrete value and treat every other answer alike
    /// — currently the array-axiom instantiator, which asks "does the candidate
    /// model already satisfy this axiom instance?" and re-asserts the instance
    /// whenever the answer is not a definite `true`.  Collapsing
    /// [`EvalOutcome::Unrepresentable`] into `None` is right for that caller:
    /// an axiom instance the evaluator could not fold is one it has not
    /// verified, so instantiating it again is the safe move.  The
    /// model-verification gate must *not* collapse the two and uses the outcome
    /// form directly.
    pub(super) fn eval_in_model(
        &self,
        term: TermId,
        model: &Model,
        manager: &TermManager,
        depth: u32,
    ) -> Option<EvalVal> {
        self.eval_in_model_outcome(term, model, manager, depth)
            .value()
    }

    /// Evaluate `term` under `model`.
    ///
    /// Runs the explicit frame stack described in the module documentation: a
    /// single loop that alternates between asking the innermost pending
    /// operator for its next operand and handing finished operand outcomes back
    /// to it.  Native stack usage is constant in the nesting depth of `term`.
    ///
    /// `depth` is the nesting depth `term` itself sits at, charged against
    /// [`ENCODE_DEPTH_LIMIT`].  Now that the walk is iterative that limit is no
    /// longer a stack bound — it is a plain *work* bound, with the visible
    /// outcome [`EvalOutcome::Undetermined`] for anything past it, which is the
    /// same answer the recursive version gave and therefore leaves the gate's
    /// verdict on deep terms unchanged.
    ///
    /// IMPORTANT: `model.get` is consulted only for *leaf* / opaque terms (the
    /// `Var` and fallback arms of [`Self::open_in_model`]).  Operator terms
    /// (`and` / `or` / `=` / `+` / …) are ALWAYS recomputed structurally from
    /// their children — never read back from the model cache.  `build_model`
    /// records the SAT core's Boolean value for every atom and gate, and when
    /// that core commits an inconsistent trail those cached values are exactly
    /// what must not be trusted (e.g. an `or` gate cached `true` while both
    /// disjuncts are `false`).  Recomputing from leaves is what makes this gate
    /// sound.
    pub(super) fn eval_in_model_outcome(
        &self,
        term: TermId,
        model: &Model,
        manager: &TermManager,
        depth: u32,
    ) -> EvalOutcome {
        let mut frames: Vec<Frame> = Vec::new();
        // The term each frame on `frames` is evaluating, so a finished frame
        // can be memoised.  It lives beside the stack rather than inside
        // `Frame` because it is the driver's bookkeeping, exactly like
        // `Frame::base`.
        let mut frame_terms: Vec<TermId> = Vec::new();
        // Operand values of every frame on the stack, concatenated; a frame
        // owns `values[frame.base..]` while it is the innermost one.
        let mut values: Vec<EvalVal> = Vec::new();
        // A finished operand outcome travelling back to the frame that asked
        // for it.
        let mut carry: Option<EvalOutcome> = None;
        // Outcomes of the compound terms already finished on this call, so the
        // walk costs the term's DAG size rather than its TREE size.
        //
        // Terms are hash-consed, so a formula that mentions a shared subterm
        // twice really is one node with two parents — and without this table a
        // chain of `n` such nodes (`y1 = x+x`, `y2 = y1+y1`, ...) costs `2^n`
        // visits.  Measured on the bit-vector doubling chain
        // `dag<n>_shared_doubling`: 90 ms at n = 20, 1.4 s at n = 24, 22.8 s at
        // n = 28, i.e. a factor of two per level.  That cost was invisible
        // while bit-vector operators fell into `open_in_model`'s closing arm
        // and answered `Undetermined` at the first node without descending; it
        // became reachable the moment they gained arms.
        //
        // Sound because the walk is pure: it reads `model`, `self.arith` and
        // the term arena, mutates none of them, and the table lives exactly as
        // long as one call.  `depth` is the one input not in the key, and that
        // is deliberate — it is a *work* bound whose only effect is to answer
        // `Undetermined`, so re-using a value computed at a shallower depth can
        // only make a deep occurrence more precise, never wrong.
        let mut memo: FxHashMap<TermId, EvalOutcome> = FxHashMap::default();

        match self.open_in_model(term, model, manager, depth) {
            Opened::Done(outcome) => return outcome,
            Opened::Frame(frame) => {
                frames.push(frame);
                frame_terms.push(term);
            }
        }

        loop {
            let step = match frames.last_mut() {
                Some(top) => top.advance(&mut values, carry.take()),
                // Only the `Step::Done` arm below empties the stack, and it
                // returns; reaching here would mean the driver lost its root.
                None => return EvalOutcome::UNDETERMINED,
            };

            match step {
                Step::Need(child) => {
                    if let Some(cached) = memo.get(&child) {
                        carry = Some(cached.clone());
                        continue;
                    }
                    let child_depth = match frames.last() {
                        Some(top) => top.depth.saturating_add(1),
                        None => depth,
                    };
                    match self.open_in_model(child, model, manager, child_depth) {
                        Opened::Done(outcome) => carry = Some(outcome),
                        Opened::Frame(mut frame) => {
                            frame.base = values.len();
                            frames.push(frame);
                            frame_terms.push(child);
                        }
                    }
                }
                Step::Done(outcome) => {
                    let Some(frame) = frames.pop() else {
                        return EvalOutcome::UNDETERMINED;
                    };
                    let finished = frame_terms.pop();
                    values.truncate(frame.base);
                    if frames.is_empty() {
                        return outcome;
                    }
                    if let Some(finished) = finished {
                        memo.insert(finished, outcome.clone());
                    }
                    carry = Some(outcome);
                }
            }
        }
    }

    /// Read one term: either it has an outcome on its own, or it opens a frame.
    ///
    /// This is the former recursive `eval_in_model`'s dispatch, minus the
    /// recursion: an arm that used to call itself now describes its operands to
    /// the driver instead of evaluating them.
    fn open_in_model(
        &self,
        term: TermId,
        model: &Model,
        manager: &TermManager,
        depth: u32,
    ) -> Opened {
        if depth > ENCODE_DEPTH_LIMIT {
            return Opened::Done(EvalOutcome::UNDETERMINED);
        }
        let Some(t) = manager.get(term) else {
            return Opened::Done(EvalOutcome::UNDETERMINED);
        };
        let sort = t.sort;
        // The two bit-vector shapes that repeat twenty times below.  Both are
        // ordinary `Op::Eager` frames: bit-vector operators have fixed arity
        // and no short-circuit, so the driver's existing machinery carries
        // them unchanged and no new frame kind is needed.
        let bv_binary = |a: TermId, b: TermId, op: BvBinaryOp| {
            Opened::Frame(Frame::binary(a, b, EagerKind::BvBinary(op), depth))
        };
        let bv_compare = |a: TermId, b: TermId, op: BvCompareOp| {
            Opened::Frame(Frame::binary(a, b, EagerKind::BvCompare(op), depth))
        };
        match &t.kind {
            TermKind::True => Opened::Done(EvalOutcome::boolean(true)),
            TermKind::False => Opened::Done(EvalOutcome::boolean(false)),
            TermKind::IntConst(_) | TermKind::RealConst(_) | TermKind::BitVecConst { .. } => {
                Opened::Done(parse_value_term(term, manager))
            }
            TermKind::Var(_) => Opened::Done({
                // For a numeric variable, take the value from the ARITHMETIC
                // solver, not the built model.  `arith.value` returns `None` for
                // a variable the solver does not actually constrain, which makes
                // the whole evaluation inconclusive (never a false downgrade) —
                // exactly the variables `build_model` would have defaulted to 0.
                if sort == manager.sorts.int_sort || sort == manager.sorts.real_sort {
                    match self.arith.value(term) {
                        Some(n) => EvalOutcome::number(n),
                        None => EvalOutcome::UNDETERMINED,
                    }
                } else {
                    // Boolean / bit-vector / other: the model witness is fine
                    // (Booleans are exactly determined by the SAT assignment).
                    match model.get(term) {
                        Some(value_term) => parse_value_term(value_term, manager),
                        None => EvalOutcome::UNDETERMINED,
                    }
                }
            }),
            TermKind::Not(a) => Opened::Frame(Frame::unary(*a, EagerKind::Not, depth)),
            TermKind::And(args) => Opened::Frame(Frame::new(
                Op::Connective {
                    operands: args.clone(),
                    conjunction: true,
                },
                depth,
            )),
            TermKind::Or(args) => Opened::Frame(Frame::new(
                Op::Connective {
                    operands: args.clone(),
                    conjunction: false,
                },
                depth,
            )),
            TermKind::Implies(a, b) => Opened::Frame(Frame::new(
                Op::Implies {
                    antecedent: *a,
                    consequent: *b,
                    state: ImpliesState::Antecedent,
                },
                depth,
            )),
            TermKind::Ite(c, t, e) => Opened::Frame(Frame::new(
                Op::Ite {
                    cond: *c,
                    then_branch: *t,
                    else_branch: *e,
                    state: IteState::Cond,
                },
                depth,
            )),
            // ---- the structural pre-pass (`#P2b-22`) ------------------
            //
            // `x = x` is true and `(distinct … x … x …)` is false in *every*
            // interpretation, whatever sort `x` has and whether or not the
            // model pins a value for it.  Deciding them here, before any model
            // lookup, is what lets the gate refute a candidate model for an
            // assertion whose variables the encoder folded away.
            //
            // That was a measured hole, not a hypothetical one:
            // `(assert (distinct b b))` encodes to `¬true` because `mk_eq(b, b)`
            // folds to `true`, so `b` never reaches the bit-blaster and the
            // published model has no value for it.  The value-based evaluator
            // below can only answer `Undetermined` for such a term, and the
            // gate therefore approved the `sat` that `#P2b-20` produced —
            // leaving that defect with no second line of defence at all.  A
            // value-based evaluator cannot refute what has no value; this
            // needs no value.
            //
            // Sound in the only direction that matters: it manufactures a
            // definite answer *only* for a syntactic identity, which no
            // interpretation can disagree with.  Terms are hash-consed, so
            // operand identity is `TermId` equality — no traversal, no
            // normalisation.
            TermKind::Eq(a, b) if a == b => Opened::Done(EvalOutcome::boolean(true)),
            TermKind::Eq(a, b) => Opened::Frame(Frame::binary(*a, *b, EagerKind::Eq, depth)),
            TermKind::Distinct(args) if has_repeated_operand(args) => {
                Opened::Done(EvalOutcome::boolean(false))
            }
            // `distinct` is INCONCLUSIVE for the gate on everything except a
            // uniform-width bit-vector tuple.  A model in which two ARITHMETIC
            // operands share a value does NOT reliably indicate a real
            // violation: the linear-arithmetic solver enforces disequalities by
            // case-splitting, not by pinning distinct witnesses in its LP model,
            // so `arith.value` routinely reports colliding integer values for a
            // genuinely satisfiable `distinct`.  Downgrading on that would turn
            // correct `Sat`s into spurious `Unknown`s; the gate targets violated
            // POSITIVE structure (a falsified equality or an all-false clause)
            // instead, which the arithmetic model represents faithfully.
            //
            // A bit-vector witness carries no such caveat — it is a
            // `BitVecConst` naming every bit — so `Op::Distinct` decides the
            // all-bit-vector case and bails out to `Undetermined` on the first
            // operand that is anything else, which reproduces the old answer
            // for every arithmetic, Boolean or opaque `distinct` exactly.
            TermKind::Distinct(args) => Opened::Frame(Frame::new(
                Op::Distinct {
                    operands: args.clone(),
                },
                depth,
            )),
            TermKind::Add(args) => Opened::Frame(Frame::new(
                Op::Arith {
                    operands: args.clone(),
                    product: false,
                    acc: Rational64::from_integer(0),
                },
                depth,
            )),
            TermKind::Sub(a, b) => Opened::Frame(Frame::binary(*a, *b, EagerKind::Sub, depth)),
            TermKind::Mul(args) => Opened::Frame(Frame::new(
                Op::Arith {
                    operands: args.clone(),
                    product: true,
                    acc: Rational64::from_integer(1),
                },
                depth,
            )),
            TermKind::Neg(a) => Opened::Frame(Frame::unary(*a, EagerKind::Neg, depth)),
            TermKind::Lt(a, b) => Opened::Frame(Frame::binary(
                *a,
                *b,
                EagerKind::CmpStrict { less: true },
                depth,
            )),
            TermKind::Gt(a, b) => Opened::Frame(Frame::binary(
                *a,
                *b,
                EagerKind::CmpStrict { less: false },
                depth,
            )),
            TermKind::Le(a, b) => Opened::Frame(Frame::binary(
                *a,
                *b,
                EagerKind::CmpWeak { less: true },
                depth,
            )),
            TermKind::Ge(a, b) => Opened::Frame(Frame::binary(
                *a,
                *b,
                EagerKind::CmpWeak { less: false },
                depth,
            )),
            // ---- bit-vector operators ---------------------------------
            // Every one of these fell into the closing `_ =>` arm before this
            // module gained a bit-vector value: `model.get` found nothing (an
            // operator term is not a model leaf) and the gate answered
            // `Undetermined`, so it vouched for every candidate model of a
            // QF_BV formula whatever the model said.  Structural recomputation
            // from the leaves — never a read-back of a cached gate value — is
            // what makes the gate sound; see `eval_in_model_outcome`.
            TermKind::BvNot(a) => Opened::Frame(Frame::unary(*a, EagerKind::BvNot, depth)),
            TermKind::BvAnd(a, b) => bv_binary(*a, *b, BvBinaryOp::And),
            TermKind::BvOr(a, b) => bv_binary(*a, *b, BvBinaryOp::Or),
            TermKind::BvXor(a, b) => bv_binary(*a, *b, BvBinaryOp::Xor),
            TermKind::BvAdd(a, b) => bv_binary(*a, *b, BvBinaryOp::Add),
            TermKind::BvSub(a, b) => bv_binary(*a, *b, BvBinaryOp::Sub),
            TermKind::BvMul(a, b) => bv_binary(*a, *b, BvBinaryOp::Mul),
            TermKind::BvUdiv(a, b) => bv_binary(*a, *b, BvBinaryOp::Udiv),
            TermKind::BvSdiv(a, b) => bv_binary(*a, *b, BvBinaryOp::Sdiv),
            TermKind::BvUrem(a, b) => bv_binary(*a, *b, BvBinaryOp::Urem),
            TermKind::BvSrem(a, b) => bv_binary(*a, *b, BvBinaryOp::Srem),
            TermKind::BvShl(a, b) => bv_binary(*a, *b, BvBinaryOp::Shl),
            TermKind::BvLshr(a, b) => bv_binary(*a, *b, BvBinaryOp::Lshr),
            TermKind::BvAshr(a, b) => bv_binary(*a, *b, BvBinaryOp::Ashr),
            TermKind::BvConcat(a, b) => bv_binary(*a, *b, BvBinaryOp::Concat),
            TermKind::BvExtract { high, low, arg } => Opened::Frame(Frame::unary(
                *arg,
                EagerKind::BvExtract {
                    high: *high,
                    low: *low,
                },
                depth,
            )),
            TermKind::BvUlt(a, b) => bv_compare(*a, *b, BvCompareOp::Ult),
            TermKind::BvUle(a, b) => bv_compare(*a, *b, BvCompareOp::Ule),
            TermKind::BvSlt(a, b) => bv_compare(*a, *b, BvCompareOp::Slt),
            TermKind::BvSle(a, b) => bv_compare(*a, *b, BvCompareOp::Sle),
            // `TermKind` has no variant for `bvnand` / `bvnor` / `bvxnor` /
            // `bvcomp` / `bvsmod` / `bvneg` / `zero_extend` / `sign_extend` /
            // `rotate_left` / `rotate_right` / `repeat`: the term builder and
            // the parser's indexed-identifier path lower every one of them to
            // the primitives above (`TermManager::mk_bv_nand` and friends,
            // `smtlib::parser::indexed`), so they are covered here without an
            // arm of their own.
            //
            // A `let` evaluates to its body.
            //
            // The SMT-LIB parser substitutes bindings into the body and returns
            // it directly, so no `Let` reaches here on the parse path; this arm
            // exists for terms built programmatically through
            // `TermManager::mk_let`, and because falling into the opaque-leaf
            // arm below made a `Let`-rooted assertion `Undetermined` *before*
            // the gate ever looked at the formula underneath — the gate was
            // blind to exactly the assertions the vacuous wrapper covered.
            //
            // Evaluating the body alone is sound for the substituted shape (the
            // body is already the whole term) and stays *conservative* for a
            // real binder: the bound name appears as a `Var` the model does not
            // pin, which reads back `Undetermined` and propagates outward, so a
            // genuine binder yields no verdict rather than a wrong one. It can
            // never manufacture a definite `false` from a binding it ignored,
            // because a value it did not substitute cannot make a comparison
            // concrete.
            TermKind::Let { body, .. } => {
                Opened::Frame(Frame::unary(*body, EagerKind::Identity, depth))
            }
            // Opaque leaves (uninterpreted applications, selects, …): the model
            // may pin a concrete value; otherwise inconclusive.
            _ => Opened::Done(match model.get(term) {
                Some(value_term) => parse_value_term(value_term, manager),
                None => EvalOutcome::UNDETERMINED,
            }),
        }
    }
}

/// Does `args` name the same operand twice?
///
/// Terms are hash-consed, so two syntactically identical operands *are* the
/// same [`TermId`] and this is exact — no traversal and no normalisation.  A
/// `true` answer makes `(distinct …)` false in every interpretation, which is
/// the structural half of the model gate (`#P2b-22`); see the pre-pass arms in
/// [`Solver::open_in_model`].
///
/// The pairwise scan is the fast path for the two- to four-operand `distinct`
/// terms that make up essentially all real input (a `SmallVec<[TermId; 4]>` is
/// what the AST stores them in); the hash-set fallback keeps a pathologically
/// wide one linear rather than quadratic, since this runs once per `distinct`
/// node on every gate evaluation.
fn has_repeated_operand(args: &[TermId]) -> bool {
    /// Above this many operands the pairwise scan stops being the cheaper one.
    const PAIRWISE_LIMIT: usize = 16;

    if args.len() < 2 {
        return false;
    }
    if args.len() <= PAIRWISE_LIMIT {
        return args
            .iter()
            .enumerate()
            .any(|(i, a)| args[i + 1..].contains(a));
    }
    let mut seen = FxHashSet::with_capacity_and_hasher(args.len(), rustc_hash::FxBuildHasher);
    args.iter().any(|arg| !seen.insert(*arg))
}

/// Parse a constant value term (`IntConst` / `RealConst` / `True` / `False`)
/// into an outcome.
///
/// A non-constant term, or an integer constant outside `i64`, is
/// [`EvalOutcome::Undetermined`] rather than [`EvalOutcome::Unrepresentable`]:
/// a constant too wide for the evaluator's arithmetic is a property of the
/// *formula*, present before any model existed, and the recursive version has
/// always reported it inconclusive.  Treating it as a reason to downgrade would
/// make every formula that merely mentions a wide literal answer `Unknown`.
fn parse_value_term(term: TermId, manager: &TermManager) -> EvalOutcome {
    let Some(t) = manager.get(term) else {
        return EvalOutcome::UNDETERMINED;
    };
    match &t.kind {
        TermKind::True => EvalOutcome::boolean(true),
        TermKind::False => EvalOutcome::boolean(false),
        TermKind::IntConst(n) => match n.to_i64() {
            Some(v) => EvalOutcome::number(Rational64::from_integer(v)),
            None => EvalOutcome::UNDETERMINED,
        },
        TermKind::RealConst(r) => EvalOutcome::number(*r),
        // A bit-vector witness, either an interned literal or the value
        // `build_model` recorded for a variable or an opaque application
        // (`model_builder` writes them as real `BitVecConst` terms through
        // `mk_bitvec`).  Reading it is what lets the `Var` and fallback arms of
        // `open_in_model` see a bit-vector at all.
        TermKind::BitVecConst { value, width } => model_eval_bv::leaf(value, *width),
        _ => EvalOutcome::UNDETERMINED,
    }
}

#[cfg(test)]
mod tests;
