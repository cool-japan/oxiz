//! The ground string procedure's concrete evaluator.
//!
//! [`ModelBuilder::eval`] interprets a term under the builder's current string
//! assignment. It is the single evaluator behind *both* directions of the
//! ground fragment: [`super::solve_ground_string_model`] runs it over a
//! candidate assignment to *verify* a `Sat` witness, and
//! [`super::eval_ground_bool`] runs it over an empty model to *refute* a closed
//! sub-formula. What it computes is unchanged by this module existing; how it
//! walks the term is not.
//!
//! # Why the walk is iterative
//!
//! The evaluator used to recurse once per nesting level, and that was the
//! recursion that actually killed the process. `check_sat` reaches it through
//! `check_string_constraints` -> `ground_string_conflict` -> `eval_ground_bool`,
//! and an `(str.++ x1 … xN)` application is folded into `N` nested binary
//! `StrConcat` nodes, so `N` is attacker-controlled. Measured on a 1 MiB worker
//! stack — what an embedder's thread typically gets — the recursive version
//! survived a nesting depth of **2448** and died at 2452, i.e. roughly **428
//! bytes of native stack per level**; an lldb backtrace of the original crash
//! showed ~2425 frames and `EXC_BAD_ACCESS`. That is an abort, not an answer
//! the caller can handle.
//!
//! [`super::MAX_EVAL_DEPTH`] was supposed to prevent exactly this and could
//! never do so: at 4096 it sat almost twice as deep as the stack could reach,
//! so the process always died before the guard was consulted. With the walk on
//! the heap the guard is no longer a stack bound at all — it is now a plain
//! resource bound that genuinely fires, and its outcome is the evaluator's
//! ordinary `None`. Both callers read that in the safe direction: a refutation
//! that cannot be computed is not reported, and a model that cannot be verified
//! is not certified.
//!
//! # Shape
//!
//! The driver holds an explicit frame stack. Short-circuit semantics are a
//! property of the driver, and they are preserved exactly:
//!
//! * `and` stops at the first `false`, `or` at the first `true`, and both keep
//!   scanning past an operand they cannot evaluate — `false ∧ unknown` is
//!   `Some(false)`;
//! * `distinct` evaluates every operand and answers `false` as soon as two
//!   *known* operands collide, even if others are unknown;
//! * `=>` consults its consequent only when the antecedent is not `false`, and
//!   a `true` consequent decides it whatever the antecedent was;
//! * `ite` evaluates the condition and then only the taken branch, so an
//!   unknown hiding in the other branch stays invisible;
//! * every other operator is eager and gives up at the first operand with no
//!   value.
//!
//! Evaluation is pure — `&self`, no cache, no model mutation — so operand
//! *order* is unobservable in the result; it is preserved anyway, because it
//! decides how much work an unknown operand saves.

use super::{MAX_EVAL_DEPTH, ModelBuilder, Val, int_to_str, saturating_index, str_to_int};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::string::regex_membership::{ReplaceReMode, compile_regex, replace_re};
use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, str_fold};
use smallvec::SmallVec;

/// The widest fixed-arity operator the evaluator knows (`str.substr`,
/// `str.indexof`, `str.replace`).
const MAX_EAGER_ARITY: usize = 3;

/// Which comparison an arithmetic atom encodes.
#[derive(Debug, Clone, Copy)]
enum CmpKind {
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
}

impl CmpKind {
    /// Whether this comparison holds for the given operand ordering.
    fn holds(self, ordering: core::cmp::Ordering) -> bool {
        match self {
            CmpKind::Lt => ordering == core::cmp::Ordering::Less,
            CmpKind::Le => ordering != core::cmp::Ordering::Greater,
            CmpKind::Gt => ordering == core::cmp::Ordering::Greater,
            CmpKind::Ge => ordering != core::cmp::Ordering::Less,
        }
    }
}

/// A fixed-arity operator: everything whose operands are evaluated left to
/// right, stopping at the first one with no value.
#[derive(Debug, Clone, Copy)]
enum EagerKind {
    /// `not`
    Not,
    /// `xor`
    Xor,
    /// `=`
    Eq,
    /// unary `-`
    Neg,
    /// binary `-`
    Sub,
    /// One of `<` / `<=` / `>` / `>=`.
    Cmp(CmpKind),
    /// `str.++`
    StrConcat,
    /// `str.len`
    StrLen,
    /// `str.substr`
    StrSubstr,
    /// `str.at`
    StrAt,
    /// `str.contains`
    StrContains,
    /// `str.prefixof`
    StrPrefixOf,
    /// `str.suffixof`
    StrSuffixOf,
    /// `str.indexof`
    StrIndexOf,
    /// `str.replace` (`all = false`) or `str.replace_all`.
    StrReplace {
        /// `true` for `str.replace_all`.
        all: bool,
    },
    /// `str.replace_re` / `str.replace_re_all`.  The middle operand is a
    /// `RegLan` term, so it is *compiled* rather than evaluated and travels in
    /// the operator instead of on the value stack.
    StrReplaceRe {
        /// The regular expression term.
        regex: TermId,
        /// Whether to replace the first match or all of them.
        mode: ReplaceReMode,
    },
    /// `str.<`
    StrLt,
    /// `str.<=`
    StrLe,
    /// `str.to_code`
    StrToCode,
    /// `str.from_code`
    StrFromCode,
    /// `str.to_int`
    StrToInt,
    /// `str.from_int`
    IntToStr,
    /// `str.in_re`.  Like `StrReplaceRe`, the regex operand is compiled.
    StrInRe {
        /// The regular expression term.
        regex: TermId,
    },
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
#[derive(Debug, Clone, Copy)]
enum ImpliesState {
    /// The antecedent has not produced a truth value yet.
    Antecedent,
    /// The antecedent is `true`, so the implication *is* its consequent.
    ConsequentDecides,
    /// The antecedent produced no truth value; only a `true` consequent can
    /// still decide the implication.
    ConsequentMayRescue,
}

/// A pending operator and its progress.
#[derive(Debug)]
enum Op {
    /// A fixed-arity operator; only `operands[..arity]` is meaningful.
    Eager {
        /// Operand term ids in evaluation order.
        operands: [TermId; MAX_EAGER_ARITY],
        /// How many of `operands` this operator takes.
        arity: u8,
        /// What to compute from the operand values.
        kind: EagerKind,
    },
    /// `and` (`conjunction = true`) or `or`.
    Connective {
        /// Operand term ids in evaluation order.
        operands: SmallVec<[TermId; 4]>,
        /// `true` for `and`, `false` for `or`.
        conjunction: bool,
        /// Whether some operand had no truth value.
        saw_unknown: bool,
    },
    /// `distinct`, which evaluates every operand and needs the partial values.
    Distinct {
        /// Operand term ids in evaluation order.
        operands: SmallVec<[TermId; 4]>,
        /// The operand values collected so far, `None` where unknown.
        values: Vec<Option<Val>>,
    },
    /// n-ary `+` (`product = false`) or `*`, folded as each operand arrives.
    Arith {
        /// Operand term ids in evaluation order.
        operands: SmallVec<[TermId; 4]>,
        /// `true` for `*`, `false` for `+`.
        product: bool,
        /// The running sum or product.
        acc: BigInt,
    },
    /// `ite`.
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
    /// `=>`.
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
    /// The operator and its progress.
    op: Op,
    /// How many operands have been consumed so far.
    filled: usize,
    /// Where this frame's operand values start in the driver's value stack.
    base: usize,
    /// The nesting depth of this frame's term, charged against
    /// [`MAX_EVAL_DEPTH`].
    depth: usize,
}

/// What reading a term produced.
enum Opened {
    /// A leaf, an unsupported term kind, or an unknown term id.
    Done(Option<Val>),
    /// A compound term that now needs operands.
    Frame(Frame),
}

/// What the driver must do next for the frame on top of its stack.
enum Step {
    /// Evaluate this operand and hand the result back on the next turn.
    Need(TermId),
    /// The frame is finished; this is its result.
    Done(Option<Val>),
    /// Every operand of a fixed-arity operator has a value; the driver must
    /// combine them.  This is a separate step rather than something the frame
    /// does itself because two of the combiners (`str.in_re`,
    /// `str.replace_re`) have to compile a regular expression, which needs the
    /// builder's [`oxiz_core::ast::TermManager`].
    Combine(EagerKind),
}

impl Frame {
    /// A frame around an arbitrary [`Op`].  `base` is filled in by the driver
    /// when the frame is pushed, once the value stack's height is known.
    fn new(op: Op, depth: usize) -> Self {
        Self {
            op,
            filled: 0,
            base: 0,
            depth,
        }
    }

    /// A frame for a fixed-arity operator.
    fn eager(
        operands: [TermId; MAX_EAGER_ARITY],
        arity: u8,
        kind: EagerKind,
        depth: usize,
    ) -> Self {
        Self::new(
            Op::Eager {
                operands,
                arity,
                kind,
            },
            depth,
        )
    }

    /// A frame for a one-operand operator.
    fn unary(a: TermId, kind: EagerKind, depth: usize) -> Self {
        Self::eager([a, a, a], 1, kind, depth)
    }

    /// A frame for a two-operand operator, evaluated `a` then `b`.
    fn binary(a: TermId, b: TermId, kind: EagerKind, depth: usize) -> Self {
        Self::eager([a, b, b], 2, kind, depth)
    }

    /// A frame for a three-operand operator, evaluated left to right.
    fn ternary(a: TermId, b: TermId, c: TermId, kind: EagerKind, depth: usize) -> Self {
        Self::eager([a, b, c], 3, kind, depth)
    }
}

impl ModelBuilder<'_> {
    /// Evaluate a ground term (under the current `model`) to a concrete value.
    /// Returns `None` when a value cannot be determined (unassigned variable,
    /// unsupported operator, non-ground regex, out-of-range integer, past the
    /// [`MAX_EVAL_DEPTH`] budget, …).
    ///
    /// Note: this is a total structural *interpreter* over the typed SMT term
    /// AST — it walks `TermKind` nodes and computes their SMT-LIB semantics. It
    /// executes no external or user code (the name `eval` refers to term
    /// evaluation, not dynamic code evaluation).
    ///
    /// `depth` is the nesting depth `term` itself sits at.  The walk runs on an
    /// explicit heap stack, so native stack usage is constant in the nesting
    /// depth of `term` — see the module documentation for the measurements that
    /// made that necessary.
    pub(super) fn eval(&self, term: TermId, depth: usize) -> Option<Val> {
        let mut frames: Vec<Frame> = Vec::new();
        // Operand values of every frame on the stack, concatenated; a frame
        // owns `values[frame.base..]` while it is the innermost one.
        let mut values: Vec<Val> = Vec::new();
        // A finished operand result travelling back to the frame that asked
        // for it.
        let mut carry: Option<Option<Val>> = None;

        match self.open(term, depth) {
            Opened::Done(result) => return result,
            Opened::Frame(frame) => frames.push(frame),
        }

        loop {
            let step = {
                let top = frames.last_mut()?;
                top.advance(&mut values, carry.take())
            };

            let finished = match step {
                Step::Need(child) => {
                    let child_depth = match frames.last() {
                        Some(top) => top.depth.saturating_add(1),
                        None => depth,
                    };
                    match self.open(child, child_depth) {
                        Opened::Done(result) => carry = Some(result),
                        Opened::Frame(mut frame) => {
                            frame.base = values.len();
                            frames.push(frame);
                        }
                    }
                    continue;
                }
                Step::Done(result) => result,
                Step::Combine(kind) => {
                    let base = {
                        let top = frames.last()?;
                        top.base
                    };
                    self.combine_eager(kind, &values[base..])
                }
            };

            // The frame that produced `finished` is still on the stack — the
            // `Step::Need` arm above is the only one that does not reach here,
            // and it `continue`s — so this never actually declines.
            let frame = frames.pop()?;
            values.truncate(frame.base);
            if frames.is_empty() {
                return finished;
            }
            carry = Some(finished);
        }
    }

    /// Read one term: either it has a value on its own, or it opens a frame.
    ///
    /// This is the former recursive `eval`'s dispatch, minus the recursion: an
    /// arm that used to call `self.eval(..)` now describes its operands to the
    /// driver instead of evaluating them.
    fn open(&self, term: TermId, depth: usize) -> Opened {
        if depth > MAX_EVAL_DEPTH {
            return Opened::Done(None);
        }
        let Some(td) = self.manager.get(term) else {
            return Opened::Done(None);
        };
        match &td.kind {
            TermKind::True => Opened::Done(Some(Val::Bool(true))),
            TermKind::False => Opened::Done(Some(Val::Bool(false))),
            TermKind::IntConst(n) => Opened::Done(Some(Val::Int(n.clone()))),
            TermKind::StringLit(s) => Opened::Done(Some(Val::Str(s.clone()))),
            TermKind::Var(_) => Opened::Done(self.model.get(&term).map(|s| Val::Str(s.clone()))),

            TermKind::Not(a) => Opened::Frame(Frame::unary(*a, EagerKind::Not, depth)),
            TermKind::And(args) => Opened::Frame(Frame::new(
                Op::Connective {
                    operands: args.clone(),
                    conjunction: true,
                    saw_unknown: false,
                },
                depth,
            )),
            TermKind::Or(args) => Opened::Frame(Frame::new(
                Op::Connective {
                    operands: args.clone(),
                    conjunction: false,
                    saw_unknown: false,
                },
                depth,
            )),
            TermKind::Xor(a, b) => Opened::Frame(Frame::binary(*a, *b, EagerKind::Xor, depth)),
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
            TermKind::Eq(a, b) => Opened::Frame(Frame::binary(*a, *b, EagerKind::Eq, depth)),
            TermKind::Distinct(args) => Opened::Frame(Frame::new(
                Op::Distinct {
                    values: Vec::with_capacity(args.len()),
                    operands: args.clone(),
                },
                depth,
            )),

            TermKind::Neg(a) => Opened::Frame(Frame::unary(*a, EagerKind::Neg, depth)),
            TermKind::Add(args) => Opened::Frame(Frame::new(
                Op::Arith {
                    operands: args.clone(),
                    product: false,
                    acc: BigInt::from(0),
                },
                depth,
            )),
            TermKind::Sub(a, b) => Opened::Frame(Frame::binary(*a, *b, EagerKind::Sub, depth)),
            TermKind::Mul(args) => Opened::Frame(Frame::new(
                Op::Arith {
                    operands: args.clone(),
                    product: true,
                    acc: BigInt::from(1),
                },
                depth,
            )),
            TermKind::Lt(a, b) => {
                Opened::Frame(Frame::binary(*a, *b, EagerKind::Cmp(CmpKind::Lt), depth))
            }
            TermKind::Le(a, b) => {
                Opened::Frame(Frame::binary(*a, *b, EagerKind::Cmp(CmpKind::Le), depth))
            }
            TermKind::Gt(a, b) => {
                Opened::Frame(Frame::binary(*a, *b, EagerKind::Cmp(CmpKind::Gt), depth))
            }
            TermKind::Ge(a, b) => {
                Opened::Frame(Frame::binary(*a, *b, EagerKind::Cmp(CmpKind::Ge), depth))
            }

            TermKind::StrConcat(a, b) => {
                Opened::Frame(Frame::binary(*a, *b, EagerKind::StrConcat, depth))
            }
            TermKind::StrLen(a) => Opened::Frame(Frame::unary(*a, EagerKind::StrLen, depth)),
            TermKind::StrSubstr(s, i, l) => {
                Opened::Frame(Frame::ternary(*s, *i, *l, EagerKind::StrSubstr, depth))
            }
            TermKind::StrAt(s, i) => Opened::Frame(Frame::binary(*s, *i, EagerKind::StrAt, depth)),
            TermKind::StrContains(hay, needle) => {
                Opened::Frame(Frame::binary(*hay, *needle, EagerKind::StrContains, depth))
            }
            TermKind::StrPrefixOf(pre, s) => {
                Opened::Frame(Frame::binary(*pre, *s, EagerKind::StrPrefixOf, depth))
            }
            TermKind::StrSuffixOf(suf, s) => {
                Opened::Frame(Frame::binary(*suf, *s, EagerKind::StrSuffixOf, depth))
            }
            TermKind::StrIndexOf(s, t, i) => {
                Opened::Frame(Frame::ternary(*s, *t, *i, EagerKind::StrIndexOf, depth))
            }
            TermKind::StrReplace(s, t, r) => Opened::Frame(Frame::ternary(
                *s,
                *t,
                *r,
                EagerKind::StrReplace { all: false },
                depth,
            )),
            TermKind::StrReplaceAll(s, t, r) => Opened::Frame(Frame::ternary(
                *s,
                *t,
                *r,
                EagerKind::StrReplace { all: true },
                depth,
            )),
            // The `RegLan` operand is compiled, never evaluated, so only the
            // subject and the replacement reach the value stack — in that
            // order, exactly as the recursive version evaluated them.
            TermKind::StrReplaceRe(s, re, r) => Opened::Frame(Frame::binary(
                *s,
                *r,
                EagerKind::StrReplaceRe {
                    regex: *re,
                    mode: ReplaceReMode::First,
                },
                depth,
            )),
            TermKind::StrReplaceReAll(s, re, r) => Opened::Frame(Frame::binary(
                *s,
                *r,
                EagerKind::StrReplaceRe {
                    regex: *re,
                    mode: ReplaceReMode::All,
                },
                depth,
            )),
            TermKind::StrLt(a, b) => Opened::Frame(Frame::binary(*a, *b, EagerKind::StrLt, depth)),
            TermKind::StrLe(a, b) => Opened::Frame(Frame::binary(*a, *b, EagerKind::StrLe, depth)),
            TermKind::StrToCode(s) => Opened::Frame(Frame::unary(*s, EagerKind::StrToCode, depth)),
            TermKind::StrFromCode(n) => {
                Opened::Frame(Frame::unary(*n, EagerKind::StrFromCode, depth))
            }
            TermKind::StrToInt(s) => Opened::Frame(Frame::unary(*s, EagerKind::StrToInt, depth)),
            TermKind::IntToStr(n) => Opened::Frame(Frame::unary(*n, EagerKind::IntToStr, depth)),
            TermKind::StrInRe(s, re) => {
                Opened::Frame(Frame::unary(*s, EagerKind::StrInRe { regex: *re }, depth))
            }

            _ => Opened::Done(None),
        }
    }

    /// Combine the operand values of a fixed-arity operator.
    ///
    /// The driver only reaches here once every operand produced a value, so
    /// each arm sees exactly the arity its operator declared.
    fn combine_eager(&self, kind: EagerKind, values: &[Val]) -> Option<Val> {
        match (kind, values) {
            (EagerKind::Not, [a]) => Some(Val::Bool(!a.as_bool()?)),
            (EagerKind::Xor, [a, b]) => Some(Val::Bool(a.as_bool()? ^ b.as_bool()?)),
            (EagerKind::Eq, [a, b]) => Some(Val::Bool(a == b)),
            (EagerKind::Neg, [a]) => Some(Val::Int(-a.as_int()?.clone())),
            (EagerKind::Sub, [a, b]) => Some(Val::Int(a.as_int()? - b.as_int()?)),
            (EagerKind::Cmp(cmp), [a, b]) => {
                Some(Val::Bool(cmp.holds(a.as_int()?.cmp(b.as_int()?))))
            }

            (EagerKind::StrConcat, [a, b]) => {
                let mut s = a.as_str()?.to_string();
                s.push_str(b.as_str()?);
                Some(Val::Str(s))
            }
            (EagerKind::StrLen, [a]) => Some(Val::Int(BigInt::from(a.as_str()?.chars().count()))),
            (EagerKind::StrSubstr, [s, i, l]) => eval_substr(s, i, l),
            (EagerKind::StrAt, [s, i]) => eval_at(s, i),
            (EagerKind::StrContains, [hay, needle]) => {
                Some(Val::Bool(hay.as_str()?.contains(needle.as_str()?)))
            }
            (EagerKind::StrPrefixOf, [pre, s]) => {
                Some(Val::Bool(s.as_str()?.starts_with(pre.as_str()?)))
            }
            (EagerKind::StrSuffixOf, [suf, s]) => {
                Some(Val::Bool(s.as_str()?.ends_with(suf.as_str()?)))
            }
            (EagerKind::StrIndexOf, [s, t, i]) => eval_indexof(s, t, i),
            (EagerKind::StrReplace { all }, [s, t, r]) => eval_replace(s, t, r, all),
            (EagerKind::StrReplaceRe { regex, mode }, [s, r]) => {
                let subject = s.as_str()?.to_string();
                let with = r.as_str()?.to_string();
                let compiled = compile_regex(self.manager, regex)?;
                replace_re(&subject, &compiled, &with, mode).map(Val::Str)
            }
            (EagerKind::StrLt, [a, b]) => {
                Some(Val::Bool(str_fold::str_lt(a.as_str()?, b.as_str()?)))
            }
            (EagerKind::StrLe, [a, b]) => {
                Some(Val::Bool(str_fold::str_le(a.as_str()?, b.as_str()?)))
            }
            (EagerKind::StrToCode, [s]) => Some(Val::Int(str_fold::str_to_code(s.as_str()?))),
            (EagerKind::StrFromCode, [n]) => match str_fold::str_from_code(n.as_int()?) {
                str_fold::FromCode::Char(c) => {
                    let mut text = String::new();
                    text.push(c);
                    Some(Val::Str(text))
                }
                str_fold::FromCode::Empty => Some(Val::Str(String::new())),
                // A surrogate is in the theory's alphabet but not in OxiZ's
                // `char`-backed strings; decline rather than fabricate a value
                // of the wrong length.
                str_fold::FromCode::Unrepresentable => None,
            },
            (EagerKind::StrToInt, [s]) => Some(Val::Int(str_to_int(s.as_str()?))),
            (EagerKind::IntToStr, [n]) => Some(Val::Str(int_to_str(n.as_int()?))),
            (EagerKind::StrInRe { regex }, [s]) => {
                let compiled = compile_regex(self.manager, regex)?;
                Some(Val::Bool(compiled.matches(s.as_str()?)))
            }

            // The driver builds each frame with the arity its operator
            // declares, so a mismatch here is an internal inconsistency rather
            // than anything an input can provoke.
            _ => None,
        }
    }
}

impl Frame {
    /// Fold `incoming` (when there is one) into the frame, then say what the
    /// frame needs next.
    fn advance(&mut self, values: &mut Vec<Val>, incoming: Option<Option<Val>>) -> Step {
        if let Some(result) = incoming
            && let Some(finished) = self.accept(values, result)
        {
            return Step::Done(finished);
        }
        self.request()
    }

    /// Fold one operand result into the frame.
    ///
    /// Returns `Some` when that operand ends the frame there and then.
    fn accept(&mut self, values: &mut Vec<Val>, result: Option<Val>) -> Option<Option<Val>> {
        match &mut self.op {
            // Every fixed-arity operator wrote `self.eval(..)?`, which gives up
            // at the first operand with no value.
            Op::Eager { .. } => match result {
                Some(value) => {
                    values.push(value);
                    self.filled += 1;
                    None
                }
                None => Some(None),
            },
            Op::Connective {
                conjunction,
                saw_unknown,
                ..
            } => {
                match result.and_then(|v| v.as_bool()) {
                    // The deciding truth value ends the connective outright,
                    // whatever an earlier operand left behind.
                    Some(b) if b != *conjunction => return Some(Some(Val::Bool(b))),
                    Some(_) => {}
                    None => *saw_unknown = true,
                }
                self.filled += 1;
                None
            }
            Op::Distinct { values: slots, .. } => {
                // Two operands with the *same* value refute the whole
                // `distinct` no matter what the others are, so a collision ends
                // the frame as soon as it appears.
                if let Some(ref value) = result
                    && slots.iter().flatten().any(|seen| seen == value)
                {
                    return Some(Some(Val::Bool(false)));
                }
                slots.push(result);
                self.filled += 1;
                None
            }
            Op::Arith { product, acc, .. } => {
                let Some(operand) = result.as_ref().and_then(Val::as_int) else {
                    return Some(None);
                };
                if *product {
                    *acc *= operand;
                } else {
                    *acc += operand;
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
                IteState::Cond => match result.and_then(|v| v.as_bool()) {
                    Some(true) => {
                        *state = IteState::Branch(*then_branch);
                        None
                    }
                    Some(false) => {
                        *state = IteState::Branch(*else_branch);
                        None
                    }
                    None => Some(None),
                },
                // The taken branch's result *is* the `ite`'s result.
                IteState::Branch(_) => Some(result),
            },
            Op::Implies { state, .. } => match state {
                ImpliesState::Antecedent => match result.and_then(|v| v.as_bool()) {
                    // `false => _` is `true`, with no look at the consequent.
                    Some(false) => Some(Some(Val::Bool(true))),
                    Some(true) => {
                        *state = ImpliesState::ConsequentDecides;
                        None
                    }
                    None => {
                        *state = ImpliesState::ConsequentMayRescue;
                        None
                    }
                },
                ImpliesState::ConsequentDecides => {
                    Some(result.and_then(|v| v.as_bool()).map(Val::Bool))
                }
                // `_ => true` is `true` whatever the antecedent was.
                ImpliesState::ConsequentMayRescue => match result.and_then(|v| v.as_bool()) {
                    Some(true) => Some(Some(Val::Bool(true))),
                    _ => Some(None),
                },
            },
        }
    }

    /// The next operand to evaluate, or the frame's finished result.
    fn request(&self) -> Step {
        match &self.op {
            Op::Eager {
                operands,
                arity,
                kind,
            } => {
                if self.filled < usize::from(*arity) {
                    Step::Need(operands[self.filled])
                } else {
                    Step::Combine(*kind)
                }
            }
            Op::Connective {
                operands,
                conjunction,
                saw_unknown,
            } => {
                if self.filled < operands.len() {
                    Step::Need(operands[self.filled])
                } else if *saw_unknown {
                    Step::Done(None)
                } else {
                    Step::Done(Some(Val::Bool(*conjunction)))
                }
            }
            Op::Distinct { operands, values } => {
                if self.filled < operands.len() {
                    Step::Need(operands[self.filled])
                } else if values.iter().any(Option::is_none) {
                    Step::Done(None)
                } else {
                    Step::Done(Some(Val::Bool(true)))
                }
            }
            Op::Arith { operands, acc, .. } => {
                if self.filled < operands.len() {
                    Step::Need(operands[self.filled])
                } else {
                    Step::Done(Some(Val::Int(acc.clone())))
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

/// SMT-LIB `str.substr`: the substring of length at most `l` starting at `i`,
/// or the empty string when the indices are out of range.
///
/// Spec (SMT-LIB Unicode Strings, `str.substr s m n`): the unique word `w` with
/// `s = a·w·b`, `|a| = m` and `|w| = min(n, |s| - m)` when `0 ≤ m < |s|` and
/// `n > 0`; the empty string in every other case — in particular for `m < 0`,
/// `m ≥ |s|` (the issue #23 shape) and `n ≤ 0`.
fn eval_substr(s: &Val, i: &Val, l: &Val) -> Option<Val> {
    let chars: Vec<char> = s.as_str()?.chars().collect();
    let n = chars.len() as i64;
    let i = saturating_index(i.as_int()?);
    let l = saturating_index(l.as_int()?);
    if i < 0 || i >= n || l <= 0 {
        return Some(Val::Str(String::new()));
    }
    // `i + l` overflows for a length near `i64::MAX`; saturate instead, the
    // result is clamped to `n` anyway.
    let end = i.saturating_add(l).min(n);
    Some(Val::Str(chars[i as usize..end as usize].iter().collect()))
}

/// SMT-LIB `str.at`: the one-character string at index `i`, or empty.
///
/// Spec: `(str.at s n)` is `(str.substr s n 1)`, so an index below `0` or
/// at/after `|s|` yields the empty string rather than being undefined.
fn eval_at(s: &Val, i: &Val) -> Option<Val> {
    let chars: Vec<char> = s.as_str()?.chars().collect();
    let n = chars.len() as i64;
    let i = saturating_index(i.as_int()?);
    if i < 0 || i >= n {
        return Some(Val::Str(String::new()));
    }
    Some(Val::Str(chars[i as usize].to_string()))
}

/// SMT-LIB `str.indexof`: first index of `t` in `s` at or after `i`, or -1.
///
/// Spec (`str.indexof s t m`): the smallest `n ≥ m` such that `s = a·t·b` with
/// `|a| = n`, provided `0 ≤ m ≤ |s|` and such an `n` exists; `-1` otherwise.
/// Two consequences the naive reading misses: an **empty** needle occurs at
/// every position, so the answer is `m` itself whenever `0 ≤ m ≤ |s|`
/// (including `m = |s|`), and a start offset of exactly `|s|` is in range while
/// `|s| + 1` is not.
fn eval_indexof(s: &Val, t: &Val, i: &Val) -> Option<Val> {
    let s_chars: Vec<char> = s.as_str()?.chars().collect();
    let t_chars: Vec<char> = t.as_str()?.chars().collect();
    let n = s_chars.len() as i64;
    let start = saturating_index(i.as_int()?);
    if start < 0 || start > n {
        return Some(Val::Int(BigInt::from(-1)));
    }
    // Empty needle matches at `start`.
    if t_chars.is_empty() {
        return Some(Val::Int(BigInt::from(start)));
    }
    let start = start as usize;
    let tlen = t_chars.len();
    if tlen > s_chars.len() {
        return Some(Val::Int(BigInt::from(-1)));
    }
    let last = s_chars.len() - tlen;
    for begin in start..=last {
        if s_chars[begin..begin + tlen] == t_chars[..] {
            return Some(Val::Int(BigInt::from(begin)));
        }
    }
    Some(Val::Int(BigInt::from(-1)))
}

/// SMT-LIB `str.replace` / `str.replace_all`.
///
/// Spec: `(str.replace s t t')` replaces the **leftmost** occurrence of `t` in
/// `s` by `t'` and leaves `s` unchanged when `t` does not occur;
/// `(str.replace_all s t t')` replaces every non-overlapping occurrence,
/// scanning left to right over the original `s`. The empty pattern is the
/// asymmetric case: `t = ""` occurs at position `0`, so `str.replace` yields
/// `t' ++ s`, whereas `str.replace_all` is *defined* to return `s` unchanged
/// (it cannot replace infinitely many empty occurrences).
fn eval_replace(s: &Val, t: &Val, r: &Val, all: bool) -> Option<Val> {
    let s = s.as_str()?.to_string();
    let t = t.as_str()?.to_string();
    let r = r.as_str()?.to_string();
    if t.is_empty() {
        // Match Z3/SMT-LIB empty-pattern semantics: `str.replace` (first)
        // prepends the replacement (`r ++ s`), whereas `str.replace_all`
        // leaves the string unchanged (it cannot replace infinitely).
        let out = if all { s } else { format!("{r}{s}") };
        return Some(Val::Str(out));
    }
    let out = if all {
        s.replace(&t, &r)
    } else {
        s.replacen(&t, &r, 1)
    };
    Some(Val::Str(out))
}

#[cfg(test)]
mod tests {
    use super::MAX_EVAL_DEPTH;
    use crate::string::eval_ground_bool;
    use oxiz_core::ast::TermManager;

    /// The stack the budget-calibrated regression tests run on: 1 MiB, what an
    /// embedder's worker thread typically gets.  A native stack overflow aborts
    /// the process, so "the closure returned at all" is itself an assertion.
    ///
    /// Both tests that use this pin their nesting depth against
    /// [`MAX_EVAL_DEPTH`], a production constant, so neither those depths nor
    /// this stack can be scaled without changing what they mean.
    // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — pins the
    // realistic embedder worker-thread budget, not a scaled test depth.
    // See TODO.md "v0.3.2 backlog".
    const WORKER_STACK: usize = 1 << 20;

    /// The stack the far-past-the-budget test runs on: one eighth of
    /// [`WORKER_STACK`], paired with one eighth of the chain length.  Only the
    /// ratio -- about one byte of stack per nesting level -- decides what that
    /// test detects, while a chain eight times shorter costs proportionally
    /// less to build.  Never raise one without the other.
    const DEEP_WORKER_STACK: usize = 1 << 17;

    /// Run `body` on a fresh thread with `stack_bytes` of stack and return its
    /// result.
    fn on_stack<T: Send + 'static>(
        stack_bytes: usize,
        body: impl FnOnce() -> T + Send + 'static,
    ) -> T {
        std::thread::Builder::new()
            .stack_size(stack_bytes)
            .spawn(body)
            .expect("spawn worker thread")
            .join()
            .expect("worker thread must return, not abort")
    }

    /// Run `body` on a fresh 1 MiB thread and return its result.
    fn on_worker_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
        on_stack(WORKER_STACK, body)
    }

    /// `(= (str.++ "a" (str.++ "a" … "")) target)` with `levels` nested
    /// `str.++` nodes, evaluated as a closed ground formula.
    ///
    /// The chain is built with an iterative loop; a recursive helper would move
    /// the stack overflow into the test itself.  `mk_str_concat` does no
    /// constant folding, so the nesting really is `levels` deep.
    fn concat_chain_equals(levels: usize, target: &str) -> Option<bool> {
        let mut manager = TermManager::new();
        let mut chain = manager.mk_string_lit("");
        let unit = manager.mk_string_lit("a");
        for _ in 0..levels {
            chain = manager.mk_str_concat(unit, chain);
        }
        let expected = manager.mk_string_lit(target);
        let assertion = manager.mk_eq(chain, expected);
        eval_ground_bool(&manager, assertion)
    }

    /// A `str.++` chain just inside the depth budget is evaluated on the heap
    /// and still computes the exact right answer, in both directions.
    ///
    /// The recursive evaluator survived a nesting depth of only 2448 on this
    /// stack (~428 bytes per level), so 4000 levels aborted the process.
    #[test]
    fn deep_concat_chain_evaluates_exactly_on_a_worker_stack() {
        const LEVELS: usize = 4000;

        let (matching, mismatching) = on_worker_stack(|| {
            let target: String = core::iter::repeat_n('a', LEVELS).collect();
            let matching = concat_chain_equals(LEVELS, &target);
            let mismatching = concat_chain_equals(LEVELS, "b");
            (matching, mismatching)
        });

        assert_eq!(matching, Some(true));
        assert_eq!(mismatching, Some(false));
    }

    /// Past the depth budget the evaluator declines with `None` — the visible
    /// outcome of the resource bound — instead of aborting the process on the
    /// way there.  Both callers read `None` in the safe direction: no
    /// refutation is reported and no model is certified.
    #[test]
    fn past_the_depth_budget_declines_rather_than_aborting() {
        let levels = MAX_EVAL_DEPTH + 1;
        let decided = on_worker_stack(move || concat_chain_equals(levels, "a"));
        assert_eq!(decided, None);
    }

    /// A chain far beyond anything the recursive version could reach still
    /// returns.  The recursive evaluator managed ~2448 levels per MiB of stack,
    /// so ~306 on [`DEEP_WORKER_STACK`]: this is still 400× that.
    #[test]
    fn a_chain_far_beyond_recursion_still_returns() {
        const LEVELS: usize = 125_000;

        let decided = on_stack(DEEP_WORKER_STACK, || concat_chain_equals(LEVELS, "a"));
        assert_eq!(decided, None);
    }

    /// Short-circuit semantics survive the conversion: `false ∧ unknown` is
    /// still `Some(false)` and `true ∨ unknown` is still `Some(true)`, while
    /// the same connectives with only an unknown operand stay undecided.
    #[test]
    fn connectives_still_short_circuit_past_unknowns() {
        let mut manager = TermManager::new();
        let string_sort = manager.sorts.string_sort();
        let unknown = manager.mk_var("x", string_sort);
        let empty = manager.mk_string_lit("");
        let unknown_atom = manager.mk_eq(unknown, empty);
        let a = manager.mk_string_lit("a");
        let b = manager.mk_string_lit("b");
        let falsehood = manager.mk_eq(a, b);
        let truth = manager.mk_eq(a, a);

        let conjunction = manager.mk_and([falsehood, unknown_atom]);
        assert_eq!(eval_ground_bool(&manager, conjunction), Some(false));

        let disjunction = manager.mk_or([truth, unknown_atom]);
        assert_eq!(eval_ground_bool(&manager, disjunction), Some(true));

        let undecided = manager.mk_and([truth, unknown_atom]);
        assert_eq!(eval_ground_bool(&manager, undecided), None);
    }
}
