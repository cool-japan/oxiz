//! The evaluator's term reader: one `TermKind` at a time, plus the store chain
//! a read-over-write walks.
//!
//! Split out of `model_eval.rs` when the `#P2b-33`/`#P2b-35` work pushed that
//! file past the workspace's 2000-line limit.  `theory_manager/intern.rs` is
//! the precedent: a self-contained concern lifted into its own child module,
//! with `impl Solver` reopened here rather than in the parent file.
//!
//! Nothing here decides anything on its own — [`Solver::open_in_model`] either
//! answers a leaf outright or describes a compound term's operands to the
//! driver in `model_eval.rs`, and [`Solver::store_chain`] hands that driver the
//! levels of a `select`'s store chain.  The readings they implement
//! ([`SelectSemantics`], [`LeafSource`]) are documented on those types.

use super::*;

impl Solver {
    /// Read one term: either it has an outcome on its own, or it opens a frame.
    ///
    /// This is the former recursive `eval_in_model`'s dispatch, minus the
    /// recursion: an arm that used to call itself now describes its operands to
    /// the driver instead of evaluating them.
    pub(super) fn open_in_model(
        &self,
        term: TermId,
        model: &Model,
        manager: &TermManager,
        depth: u32,
        selects: SelectSemantics,
        leaf: LeafSource,
        interps: &FuncInterps,
        select_index: &mut SelectIndex,
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
                // `(get-value)` asks for the model's entry instead
                // (`LeafSource::Model`): it prints what the model says, and
                // the tableau may hold nothing, or a stale value, for a
                // variable another engine decided.
                if leaf == LeafSource::Tableau
                    && (sort == manager.sorts.int_sort || sort == manager.sorts.real_sort)
                {
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
            // `TermKind::Div` is both SMT-LIB divisions (the parser routes
            // `div` and `/` to it).  Only the real one is folded here; the
            // integer one is Euclidean, has its meaning from the defining
            // axioms `arith_axioms` asserts, and is read back from the model
            // entry the tableau gave it (the arm below).
            //
            // And only for `LeafSource::Model`, the `(get-value)` reading this
            // folding was added for (`#P2b-35`).  The gate keeps the reading it
            // had before: a real division is an opaque leaf whose value the
            // tableau decided, read from the model entry by the arm below.
            // Folding it for the gate too would let `Unrepresentable` (a
            // `Rational64` overflow in `checked_div`) refuse a `Sat` the search
            // reached legitimately, which is a new `sat` -> `unknown` path and
            // no part of what `#P2b-35` set out to change.
            TermKind::Div(a, b) if leaf == LeafSource::Model && sort == manager.sorts.real_sort => {
                Opened::Frame(Frame::binary(*a, *b, EagerKind::RealDiv, depth))
            }
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
            // ---- arrays (`#P2b-32`) ------------------------------------
            //
            // A read over a `store` chain is an operator term, and under the
            // gate's reading it is recomputed from its parts like any other:
            // read-over-write down the chain, the published read of the
            // innermost base when every level is missed.  Before this arm a
            // `select` fell into the opaque-leaf arm below, so a read the
            // circuit had left as a *free* leaf — the whole `#P2b-32` family,
            // `(distinct (bvadd (select (store arr i #x05) i) #x01) #x06)`
            // published `sat` with `s = #xff` — evaluated to whatever the free
            // bits said and the gate vouched for it.  A read with no store
            // under it has nothing to recompute and stays the leaf it was, and
            // the instantiator's reading (`SelectSemantics::PublishedLeaf`)
            // never opens the frame at all.
            TermKind::Select(array, index) => {
                let (levels, base) = match selects {
                    SelectSemantics::ReadOverWrite => {
                        self.store_chain(*array, *index, model, manager, select_index)
                    }
                    SelectSemantics::PublishedLeaf => (SmallVec::new(), EvalOutcome::UNDETERMINED),
                };
                if levels.is_empty() {
                    // A read with no `store` under it is a leaf — unless the
                    // array is an array constant, which answers *every* index
                    // with its default and which `store_chain` reports as the
                    // chain's fallback even when the chain itself is empty
                    // (`#P2b-36`).  Under `PublishedLeaf` the fallback is
                    // always `Undetermined`, so that reading is untouched.
                    if matches!(base, EvalOutcome::Value(_)) {
                        return Opened::Done(base);
                    }
                    return Opened::Done(match model.get(term) {
                        Some(value_term) => parse_value_term(value_term, manager),
                        None => EvalOutcome::UNDETERMINED,
                    });
                }
                Opened::Frame(Frame::new(
                    Op::Select {
                        index: *index,
                        levels,
                        base,
                        state: SelectState::Index,
                    },
                    depth,
                ))
            }
            // An uninterpreted application the model does not pin: under
            // `(get-value)`'s reading the function's interpretation answers it
            // (`#P2b-35`, `Op::ApplyInterp`).  The gate keeps the leaf reading.
            TermKind::Apply { func, args, .. }
                if leaf == LeafSource::Model && model.get(term).is_none() =>
            {
                let Some(entries) = interps.get(&func.into_inner().get()) else {
                    return Opened::Done(EvalOutcome::UNDETERMINED);
                };
                Opened::Frame(Frame::new(
                    Op::ApplyInterp {
                        operands: args.iter().copied().collect(),
                        entries: entries.clone(),
                    },
                    depth,
                ))
            }
            // Opaque leaves (uninterpreted applications, …): the model may pin
            // a concrete value; otherwise inconclusive.
            _ => Opened::Done(match model.get(term) {
                Some(value_term) => parse_value_term(value_term, manager),
                None => EvalOutcome::UNDETERMINED,
            }),
        }
    }

    /// The store chain under a read's array operand, outermost level first,
    /// together with the outcome the read falls back to once every level is
    /// definitely missed: the model's entry for the innermost base's read at
    /// the same index — the term the read-over-write lemma interns
    /// (`select(base, index)`), which `build_model` records like any other
    /// opaque leaf — or `Undetermined` when the model holds none.
    ///
    /// Terms are hash-consed, so a chain is a path, never a cycle, and its
    /// length is bounded by the term's size; the levels live on the heap.
    /// The `(array, index)` index over the model is built once per
    /// evaluation, on the first chain that needs it: a `Model` is keyed by
    /// term id, and the base's read is a *different* term from the one being
    /// evaluated.
    pub(super) fn store_chain(
        &self,
        array: TermId,
        index: TermId,
        model: &Model,
        manager: &TermManager,
        select_index: &mut SelectIndex,
    ) -> (SmallVec<[(TermId, TermId); 4]>, EvalOutcome) {
        let mut levels: SmallVec<[(TermId, TermId); 4]> = SmallVec::new();
        let mut base = array;
        while let Some(TermKind::Store(inner, store_index, value)) =
            manager.get(base).map(|t| &t.kind)
        {
            levels.push((*store_index, *value));
            base = *inner;
        }
        // An array constant bottoms the chain out with a definite value: it
        // reads back its default at every index, so no published read of the
        // base is needed — or exists (`#P2b-36`).
        if let Some(default) = crate::solver::array_axioms::const_array_default(base, manager) {
            let value = parse_value_term(default, manager);
            if matches!(value, EvalOutcome::Value(_)) {
                return (levels, value);
            }
            // A non-literal default (an array constant over a variable) is
            // still a leaf to this evaluator; fall through to the published
            // read rather than claim a value it cannot compute.
        }
        if levels.is_empty() {
            return (levels, EvalOutcome::UNDETERMINED);
        }
        let reads = select_index.get_or_insert_with(|| {
            let mut reads: FxHashMap<(TermId, TermId), TermId> = FxHashMap::default();
            for (&key, &value) in model.assignments() {
                if let Some(TermKind::Select(read_array, read_index)) =
                    manager.get(key).map(|t| &t.kind)
                {
                    reads.insert((*read_array, *read_index), value);
                }
            }
            reads
        });
        let fallback = reads
            .get(&(base, index))
            .map_or(EvalOutcome::UNDETERMINED, |&value| {
                parse_value_term(value, manager)
            });
        (levels, fallback)
    }
}
