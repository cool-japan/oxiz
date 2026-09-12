//! Ground evaluation of bit-vector expressions for the array cross-theory
//! check.
//!
//! # Why the answer matters more than it looks
//!
//! `check_cross_theory_conflict` is not an optimisation in front of a complete
//! procedure.  It is the **only** component in the solver that closes the
//! EUF↔bit-vector gap for two array reads at congruent indices: the congruence
//! core merges `select(a, x)` with `select(a, #x05)` when `x = #x05`, but the
//! bit-vector solver receives the two reads as *independent* fresh variables and
//! there is no EUF→BV equality bridge to tell it otherwise (the only
//! Nelson–Oppen bridge is `propagate_euf_equalities_to_arith`, and bit-vector
//! equalities never enter `var_to_parsed_arith`).  Nor is there a gate that
//! downgrades to `Unknown` in this shape: `array_atoms_need_theory` requires a
//! positive `store = store` equality, `arith_atoms_need_theory` skips non-Int
//! atoms outright, and the model-verification gate cannot stand in for this
//! check: it *does* now read bit-vector values (`EvalVal::Bv` and
//! `solver/model_eval_bv.rs`), but it only inspects a **finished candidate
//! model**, after the search has already decided to report `Sat`.  This
//! evaluator runs *during* the search, on the partial assignment, which is
//! why it still exists: the gate can at best downgrade a wrong `Sat` to
//! `Unknown`, whereas the conflict found here is what lets the search reach
//! the correct `Unsat`.
//!
//! So when this evaluator answers "not evaluable" the entry is dropped, the
//! check finds nothing, and `check` returns **`Sat` for an unsatisfiable
//! formula**.  Every operator missing an arm here was a soundness hole, not an
//! incompleteness: with `bvadd` the conflict below is refuted, and with
//! `bvashr`, `bvsdiv` or `concat`/`extract` in its place it was reported `sat`
//! until each got an arm.
//!
//! ```text
//! (assert (= (select a x) (bvashr x #x01)))   ; x = 5, so the read is 2
//! (assert (= x #x05))
//! (assert (= (select a #x05) #x10))           ; the same read is 16 -> UNSAT
//! ```
//!
//! Two consequences for anyone editing this file.  Adding an arm can only turn
//! `Sat` into `Unsat`, so coverage is worth having; but getting an arm *wrong*
//! turns a satisfiable formula into `Unsat`, which is the worse direction.
//! Hence: no operator's semantics are written here.
//!
//! # Where the semantics live
//!
//! In [`oxiz_core::ast::bv_fold`], the workspace's single definition of the
//! SMT-LIB `FixedSizeBitVectors` folding rules, which the term builder, the
//! rewriter and the bit-blaster already route through.  This module contributes
//! only the parts `bv_fold` cannot know about, and adapts at that boundary
//! rather than reimplementing:
//!
//! * **Reducing operands.** `bv_fold` takes values already in `[0, 2^width)` as
//!   a precondition, and `TermManager::mk_bitvec` does not enforce it, so every
//!   leaf goes through [`bv_fold::bv_wrap_unsigned`] on the way in — the same
//!   normalisation the term builder's `bv_const_unsigned`, the SMT-LIB printer
//!   and the model builder apply.  A negative literal therefore means the same
//!   thing here as everywhere else.
//! * **Width agreement.** `bv_fold` takes one width per call and cannot tell
//!   that two operands disagreed.  Which operators require agreement, which take
//!   their width from one side only, and which produce a width neither operand
//!   has, is decided in [`apply_binary`].
//! * **"Not a value."** `bv_fold` is total on its domain; the `Option` that
//!   reports an unbound variable, an unimplemented kind or an ill-sorted term is
//!   this module's.
//!
//! This file used to carry its own copy of the folding rules, and that copy had
//! diverged three times: a shift distance read out of its low 64-bit limb (so
//! `bvshl x (_ bv2^64 65)` folded to `x` instead of zero — a manufacturable
//! spurious `unsat`), a negative distance shifted by its magnitude, and no arm
//! at all for five of the operators `bv_fold` folds.
//!
//! # Why the walk is iterative
//!
//! The terms folded here are the index and value sides of `(= (select a i) v)`
//! assertions, so their nesting depth is input-controlled, and this runs on
//! whatever stack `check_sat`'s caller has.  A stack overflow aborts the process
//! instead of returning an answer, and there is no error channel to report "too
//! deep" through — the only two answers are a value and "not evaluable", and
//! reporting "not evaluable" for a term that does have a value is exactly the
//! false `Sat` described above.  So the depth is removed rather than capped.
//!
//! The [`Solver::term_exceeds_encode_depth`](super::Solver::term_exceeds_encode_depth)
//! gate bounds *structural* depth at
//! [`ENCODE_DEPTH_LIMIT`](crate::solver::ENCODE_DEPTH_LIMIT) before `check`
//! reaches the array checks at all, but that bound was never a stack bound:
//! with `opt-level = 0` the recursive version's native frame measured about
//! 6.5 KiB, so a 1 MiB worker overflowed at roughly 160 levels — far below
//! the gate's historical value of 2000.  A depth-200 bit-vector term in a
//! select-value equality aborted the process with the gate reporting nothing
//! wrong.
//!
//! # Shape of the walk
//!
//! Following `model_eval.rs` and `check_fp.rs`.  The walk is *heterogeneous*:
//! most positions want a bit-vector value, but an `ite` condition wants a truth
//! value, so a [`Cursor`] carries the [`Position`] a term is being opened in and
//! [`open`] can only produce something of that shape.
//!
//! * [`open`] reads one term and either produces its value outright or describes
//!   the operator's operands to the driver as a [`Frame`].
//! * A [`Frame`] holds one pending operator plus its resume state, and
//!   [`Frame::resume`] turns a finished operand into the next [`Step`].
//! * [`Step::Tail`] is how `ite` short-circuits: the frame is dropped and the
//!   *taken* branch is opened in its place, so the untaken branch is never
//!   visited and a chain of `ite`s costs no frames at all.
//!
//! There is deliberately **no memo table**.  A term's cost is therefore its
//! *tree* size rather than its DAG size, exactly as before the conversion.

use crate::prelude::*;
use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, TermManager, bv_fold};
use smallvec::SmallVec;

use super::Solver;

#[cfg(test)]
mod tests;

/// A folded bit-vector value: the number, and the width it was folded at.
///
/// The width travels with the value because it is not always the operand's
/// declared sort — a variable takes both from the equality that bound it, and
/// `concat` and `extract` produce a width neither operand has.
type BvValue = (BigInt, u32);

/// The bound bit-vector variables, as `collect_bv_var_equalities` builds them.
type Bindings = FxHashMap<TermId, (BigInt, u32)>;

/// The operand list of an n-ary connective, matching `TermKind`'s own inline
/// capacity so the common case does not spill to the heap.
type Operands = SmallVec<[TermId; 4]>;

/// Which kind of value a position must produce.
#[derive(Clone, Copy)]
enum Position {
    /// A bit-vector expression.
    Bits,
    /// A condition, i.e. the first operand of an `ite`.
    Truth,
}

/// A folded value on its way back to the frame that asked for it.
enum Value {
    /// A bit-vector value.
    Bits(BvValue),
    /// A truth value.
    Truth(bool),
}

impl Value {
    /// The bit-vector payload.
    ///
    /// `None` when a truth value arrived instead.  That cannot happen: a frame
    /// names the [`Position`] each of its operands is opened in, and [`open`]
    /// only ever produces a value of the position it was given.  It is written
    /// as an `Option` rather than an assertion because "not evaluable" is the
    /// honest answer for a shape this evaluator cannot make sense of, and
    /// because it is the *conservative* answer: it can only cost a conflict, and
    /// a fabricated value could invent one.
    fn bits(self) -> Option<BvValue> {
        match self {
            Value::Bits(value) => Some(value),
            Value::Truth(_) => None,
        }
    }

    /// The truth payload; `None` when a bit-vector value arrived instead.  See
    /// [`Value::bits`] for why this is an `Option`.
    fn truth(self) -> Option<bool> {
        match self {
            Value::Truth(truth) => Some(truth),
            Value::Bits(_) => None,
        }
    }
}

/// A binary bit-vector operator: every `TermKind` variant of bit-vector sort
/// that takes two bit-vector operands.
#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Udiv,
    Urem,
    Sdiv,
    Srem,
    And,
    Or,
    Xor,
    Shl,
    Lshr,
    Ashr,
    Concat,
}

/// A comparison between two bit-vector operands, usable as an `ite` condition.
#[derive(Clone, Copy)]
enum Compare {
    Eq,
    Ult,
    Ule,
    Slt,
    Sle,
}

/// One pending operator on the frame stack, with its resume state.
enum Frame {
    /// `bvnot`, waiting for its only operand.
    Complement,
    /// `(_ extract high low)`, waiting for its only operand.
    Extract {
        /// Index of the highest bit kept.
        high: u32,
        /// Index of the lowest bit kept.
        low: u32,
    },
    /// A binary operator waiting for its **left** operand; the right one has not
    /// been opened yet.
    Left {
        /// The operator to apply once both operands are in hand.
        op: BinaryOp,
        /// The right operand, still a term.
        right: TermId,
    },
    /// A binary operator waiting for its **right** operand; the left one has
    /// already been folded.
    Right {
        /// The operator to apply once the right operand arrives.
        op: BinaryOp,
        /// The already-folded left operand.
        left: BvValue,
    },
    /// `not`, waiting for the condition it negates.
    Negate,
    /// A comparison waiting for its **left** operand.
    CompareLeft {
        /// The comparison to apply.
        cmp: Compare,
        /// The right operand, still a term.
        right: TermId,
    },
    /// A comparison waiting for its **right** operand.
    CompareRight {
        /// The comparison to apply.
        cmp: Compare,
        /// The already-folded left operand.
        left: BvValue,
    },
    /// `and` / `or`, part way through its operands.
    Connective {
        /// `true` for `and` — the value that lets the scan continue, and the
        /// value of the connective over no operands.
        all: bool,
        /// The whole operand list; `operands[next..]` is what remains.
        operands: Operands,
        /// Index of the next operand to open.
        next: usize,
    },
    /// `ite`, waiting for its condition.  Only the taken branch is ever opened.
    Branch {
        /// The branch taken when the condition holds.
        then_branch: TermId,
        /// The branch taken when it does not.
        else_branch: TermId,
    },
}

/// What a frame does with a finished operand.
enum Step {
    /// Open `term`, with the frame pushed back on to receive it.
    Need {
        /// The operand to open.
        term: TermId,
        /// The shape it must produce.
        position: Position,
        /// The frame, to be pushed back.
        frame: Frame,
    },
    /// Open `term` in the frame's *place*: the frame is finished and is not
    /// pushed back.  This is `ite`'s short-circuit.
    Tail {
        /// The term to open.
        term: TermId,
        /// The shape it must produce.
        position: Position,
    },
    /// The frame is finished and produced a value.
    Done(Value),
}

/// What one term turns into when it is opened.
enum Opened {
    /// The term folds on its own, with no operands to visit.
    Value(Value),
    /// The term is an operator: push `frame` and open `first` next.
    Operator {
        /// The pending operator.
        frame: Frame,
        /// Its first operand.
        first: TermId,
        /// The shape that operand must produce.
        position: Position,
    },
}

/// The driver's position: a term waiting to be opened in a given shape, or a
/// folded value on its way back to the frame that asked for it.
enum Cursor {
    Open(TermId, Position),
    Close(Value),
}

impl Solver {
    /// Evaluate a BV expression to a concrete (value, width) pair given variable bindings.
    /// Returns None if the expression cannot be fully evaluated.
    ///
    /// "Cannot be fully evaluated" covers an unbound variable, an operator with
    /// no arm here, a width disagreement between two operands that must match,
    /// an out-of-range `extract` index, and an `ite` whose condition does not
    /// fold.  Each of those aborts the *whole* walk rather than the arm that saw
    /// it, because every arm of the recursive version this replaced propagated
    /// the failure with `?` up to the root.  Callers treat `None` as "this check
    /// does not apply", so it is never a value standing in for one.
    pub(super) fn evaluate_bv_expr(
        &self,
        term: TermId,
        var_equalities: &Bindings,
        manager: &TermManager,
    ) -> Option<BvValue> {
        let mut frames: Vec<Frame> = Vec::new();
        let mut cursor = Cursor::Open(term, Position::Bits);

        loop {
            match cursor {
                Cursor::Open(current, position) => {
                    match open(current, position, var_equalities, manager)? {
                        Opened::Value(value) => cursor = Cursor::Close(value),
                        Opened::Operator {
                            frame,
                            first,
                            position,
                        } => {
                            frames.push(frame);
                            cursor = Cursor::Open(first, position);
                        }
                    }
                }
                Cursor::Close(value) => {
                    // An empty stack means the root itself just folded.  The
                    // root was opened in `Bits` position, so this is a value.
                    let Some(frame) = frames.pop() else {
                        return value.bits();
                    };
                    match frame.resume(value)? {
                        Step::Need {
                            term,
                            position,
                            frame,
                        } => {
                            frames.push(frame);
                            cursor = Cursor::Open(term, position);
                        }
                        Step::Tail { term, position } => cursor = Cursor::Open(term, position),
                        Step::Done(value) => cursor = Cursor::Close(value),
                    }
                }
            }
        }
    }
}

impl Frame {
    /// Hand a finished operand to this frame and get its next step.
    fn resume(self, value: Value) -> Option<Step> {
        match self {
            Frame::Complement => Some(Step::Done(Value::Bits(complement(value.bits()?)))),
            Frame::Extract { high, low } => Some(Step::Done(Value::Bits(apply_extract(
                high,
                low,
                value.bits()?,
            )?))),
            Frame::Left { op, right } => Some(Step::Need {
                term: right,
                position: Position::Bits,
                frame: Frame::Right {
                    op,
                    left: value.bits()?,
                },
            }),
            Frame::Right { op, left } => Some(Step::Done(Value::Bits(apply_binary(
                op,
                left,
                value.bits()?,
            )?))),
            Frame::Negate => Some(Step::Done(Value::Truth(!value.truth()?))),
            Frame::CompareLeft { cmp, right } => Some(Step::Need {
                term: right,
                position: Position::Bits,
                frame: Frame::CompareRight {
                    cmp,
                    left: value.bits()?,
                },
            }),
            Frame::CompareRight { cmp, left } => Some(Step::Done(Value::Truth(apply_compare(
                cmp,
                left,
                value.bits()?,
            )?))),
            Frame::Connective {
                all,
                operands,
                next,
            } => {
                let truth = value.truth()?;
                if truth != all {
                    // A `false` conjunct or a `true` disjunct decides the whole
                    // connective; the remaining operands are never opened.
                    return Some(Step::Done(Value::Truth(truth)));
                }
                match operands.get(next).copied() {
                    Some(term) => Some(Step::Need {
                        term,
                        position: Position::Truth,
                        frame: Frame::Connective {
                            all,
                            operands,
                            next: next + 1,
                        },
                    }),
                    None => Some(Step::Done(Value::Truth(all))),
                }
            }
            Frame::Branch {
                then_branch,
                else_branch,
            } => {
                let taken = if value.truth()? {
                    then_branch
                } else {
                    else_branch
                };
                // The untaken branch is never opened, and this frame is done, so
                // the taken branch takes its place on the stack.
                Some(Step::Tail {
                    term: taken,
                    position: Position::Bits,
                })
            }
        }
    }
}

/// Read one term in a given position: either it folds on its own, or it opens a
/// frame.
///
/// This is the former recursive `evaluate_bv_expr`'s dispatch, minus the
/// recursion: an arm that used to call itself now names its operands for the
/// driver instead of folding them.
fn open(
    term: TermId,
    position: Position,
    var_equalities: &Bindings,
    manager: &TermManager,
) -> Option<Opened> {
    let kind = &manager.get(term)?.kind;
    match position {
        Position::Bits => open_bits(term, kind, var_equalities),
        Position::Truth => open_truth(kind),
    }
}

/// Read a term that must produce a bit-vector value.
///
/// Every `TermKind` variant of bit-vector sort has an arm, plus `ite`.  The
/// catch-all is reached by kinds that are not bit-vector-sorted — the four
/// bit-vector comparisons are Boolean, `select` is an array read with no model
/// here, `Apply` is an uninterpreted function — and it is a catch-all rather
/// than an exhaustive listing because `TermKind` has well over a hundred
/// variants.  Reaching it costs a possible refutation; see the module docs for
/// why that is a real cost.
fn open_bits(term: TermId, kind: &TermKind, var_equalities: &Bindings) -> Option<Opened> {
    let binary = |op: BinaryOp, left: TermId, right: TermId| {
        Some(Opened::Operator {
            frame: Frame::Left { op, right },
            first: left,
            position: Position::Bits,
        })
    };
    match kind {
        // `bv_fold` takes values already reduced into `[0, 2^width)`; interned
        // literals and recorded variable bindings are not guaranteed to be.
        TermKind::BitVecConst { value, width } => {
            Some(Opened::Value(Value::Bits(reduce(value, *width))))
        }
        TermKind::Var(_) => var_equalities
            .get(&term)
            .map(|(value, width)| Opened::Value(Value::Bits(reduce(value, *width)))),
        TermKind::BvAdd(a, b) => binary(BinaryOp::Add, *a, *b),
        TermKind::BvSub(a, b) => binary(BinaryOp::Sub, *a, *b),
        TermKind::BvMul(a, b) => binary(BinaryOp::Mul, *a, *b),
        TermKind::BvUdiv(a, b) => binary(BinaryOp::Udiv, *a, *b),
        TermKind::BvUrem(a, b) => binary(BinaryOp::Urem, *a, *b),
        TermKind::BvSdiv(a, b) => binary(BinaryOp::Sdiv, *a, *b),
        TermKind::BvSrem(a, b) => binary(BinaryOp::Srem, *a, *b),
        TermKind::BvAnd(a, b) => binary(BinaryOp::And, *a, *b),
        TermKind::BvOr(a, b) => binary(BinaryOp::Or, *a, *b),
        TermKind::BvXor(a, b) => binary(BinaryOp::Xor, *a, *b),
        TermKind::BvShl(a, b) => binary(BinaryOp::Shl, *a, *b),
        TermKind::BvLshr(a, b) => binary(BinaryOp::Lshr, *a, *b),
        TermKind::BvAshr(a, b) => binary(BinaryOp::Ashr, *a, *b),
        TermKind::BvConcat(a, b) => binary(BinaryOp::Concat, *a, *b),
        TermKind::BvNot(a) => Some(Opened::Operator {
            frame: Frame::Complement,
            first: *a,
            position: Position::Bits,
        }),
        TermKind::BvExtract { high, low, arg } => Some(Opened::Operator {
            frame: Frame::Extract {
                high: *high,
                low: *low,
            },
            first: *arg,
            position: Position::Bits,
        }),
        TermKind::Ite(cond, then_branch, else_branch) => Some(Opened::Operator {
            frame: Frame::Branch {
                then_branch: *then_branch,
                else_branch: *else_branch,
            },
            first: *cond,
            position: Position::Truth,
        }),
        _ => None,
    }
}

/// Read a term that must produce a truth value, i.e. an `ite` condition.
///
/// The supported conditions are the ones decidable from bit-vector variable
/// bindings alone: the two Boolean literals, `not`, `and`, `or`, equality
/// between two bit-vector operands, and the four bit-vector comparisons.
///
/// A Boolean *variable* is deliberately absent: nothing here binds one, so it
/// could only be guessed.  Equality between two Boolean operands is also absent
/// — the [`Compare::Eq`] arm opens both sides in [`Position::Bits`], so a
/// Boolean operand simply does not fold, which is the conservative answer.
fn open_truth(kind: &TermKind) -> Option<Opened> {
    let compare = |cmp: Compare, left: TermId, right: TermId| {
        Some(Opened::Operator {
            frame: Frame::CompareLeft { cmp, right },
            first: left,
            position: Position::Bits,
        })
    };
    match kind {
        TermKind::True => Some(Opened::Value(Value::Truth(true))),
        TermKind::False => Some(Opened::Value(Value::Truth(false))),
        TermKind::Not(a) => Some(Opened::Operator {
            frame: Frame::Negate,
            first: *a,
            position: Position::Truth,
        }),
        TermKind::And(args) => Some(open_connective(true, args)),
        TermKind::Or(args) => Some(open_connective(false, args)),
        TermKind::Eq(a, b) => compare(Compare::Eq, *a, *b),
        TermKind::BvUlt(a, b) => compare(Compare::Ult, *a, *b),
        TermKind::BvUle(a, b) => compare(Compare::Ule, *a, *b),
        TermKind::BvSlt(a, b) => compare(Compare::Slt, *a, *b),
        TermKind::BvSle(a, b) => compare(Compare::Sle, *a, *b),
        _ => None,
    }
}

/// Open an `and` (`all = true`) or `or` (`all = false`).
///
/// An empty operand list folds to `all`, the identity of the connective.
fn open_connective(all: bool, args: &Operands) -> Opened {
    match args.first().copied() {
        None => Opened::Value(Value::Truth(all)),
        Some(first) => Opened::Operator {
            frame: Frame::Connective {
                all,
                operands: args.clone(),
                next: 1,
            },
            first,
            position: Position::Truth,
        },
    }
}

/// Reduce a leaf value into the unsigned bit-vector range `bv_fold` expects.
fn reduce(value: &BigInt, width: u32) -> BvValue {
    (bv_fold::bv_wrap_unsigned(value, width), width)
}

/// `bvnot`: flip every bit within the width.
fn complement((value, width): BvValue) -> BvValue {
    (bv_fold::bv_not(&value, width), width)
}

/// `(_ extract high low)`: the bits `high ..= low`, right-aligned, in a sort of
/// `high - low + 1` bits.
///
/// Out-of-range indices are not folded, matching `TermManager::mk_bv_extract`,
/// which leaves malformed indices "for the parser's sort check rather than
/// silently folding to a fabricated value".  The arithmetic is checked, so a
/// malformed index pair can only produce `None`, never a wrapped width.
fn apply_extract(high: u32, low: u32, (value, width): BvValue) -> Option<BvValue> {
    if high >= width {
        return None;
    }
    let span = high.checked_sub(low)?.checked_add(1)?;
    Some((bv_fold::bv_extract(&value, high, low), span))
}

/// Both operand widths must agree, or the term does not fold.
///
/// SMT-LIB requires equal widths for every operator that goes through here, so a
/// disagreement means the term was not well sorted and no answer for it would be
/// meaningful.
fn require_same_width(left: u32, right: u32) -> Option<()> {
    (left == right).then_some(())
}

/// Compare two folded operands.
///
/// The unsigned comparisons are on the reduced values directly; the signed ones
/// reinterpret both through [`bv_fold::to_signed`] first, so `#xff < #x01` is
/// false unsigned and true signed.
fn apply_compare(cmp: Compare, left: BvValue, right: BvValue) -> Option<bool> {
    let (left_value, width) = left;
    let (right_value, right_width) = right;
    require_same_width(width, right_width)?;
    Some(match cmp {
        Compare::Eq => left_value == right_value,
        Compare::Ult => left_value < right_value,
        Compare::Ule => left_value <= right_value,
        Compare::Slt => {
            bv_fold::to_signed(&left_value, width) < bv_fold::to_signed(&right_value, width)
        }
        Compare::Sle => {
            bv_fold::to_signed(&left_value, width) <= bv_fold::to_signed(&right_value, width)
        }
    })
}

/// Apply a binary operator to two folded operands.
///
/// Each arm states its own width rule, because they are not the same:
///
/// * the shifts take their width from the value being shifted and never look at
///   the distance's width (SMT-LIB requires the two to match, and this evaluator
///   has always answered the ill-sorted case rather than rejecting it);
/// * `concat` produces the *sum* of the two widths and so requires no
///   agreement;
/// * every other operator requires the two widths to agree.
fn apply_binary(op: BinaryOp, left: BvValue, right: BvValue) -> Option<BvValue> {
    let (left_value, width) = left;
    let (right_value, right_width) = right;
    match op {
        BinaryOp::Add => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_add(&left_value, &right_value, width), width))
        }
        BinaryOp::Sub => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_sub(&left_value, &right_value, width), width))
        }
        BinaryOp::Mul => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_mul(&left_value, &right_value, width), width))
        }
        BinaryOp::Udiv => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_udiv(&left_value, &right_value, width), width))
        }
        BinaryOp::Urem => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_urem(&left_value, &right_value, width), width))
        }
        BinaryOp::Sdiv => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_sdiv(&left_value, &right_value, width), width))
        }
        BinaryOp::Srem => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_srem(&left_value, &right_value, width), width))
        }
        BinaryOp::And => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_and(&left_value, &right_value, width), width))
        }
        BinaryOp::Or => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_or(&left_value, &right_value, width), width))
        }
        BinaryOp::Xor => {
            require_same_width(width, right_width)?;
            Some((bv_fold::bv_xor(&left_value, &right_value, width), width))
        }
        BinaryOp::Shl => Some((bv_fold::bv_shl(&left_value, &right_value, width), width)),
        BinaryOp::Lshr => Some((bv_fold::bv_lshr(&left_value, &right_value, width), width)),
        BinaryOp::Ashr => Some((bv_fold::bv_ashr(&left_value, &right_value, width), width)),
        BinaryOp::Concat => {
            // The result is wider than either operand, so the joined width is
            // computed with checked arithmetic rather than allowed to wrap.
            let joined = width.checked_add(right_width)?;
            Some((
                bv_fold::bv_concat(&left_value, &right_value, right_width),
                joined,
            ))
        }
    }
}
