//! Theory-variable tracking: the walk that gives Int/Real/BitVector terms a
//! theory variable (and hence a model value) before encoding.
//!
//! Split out of `encode.rs` into this child module once making the walk
//! exhaustive over every [`TermKind`] variant grew it past what `encode.rs`
//! -- already close to the workspace's 2000-line-per-file ceiling -- had room
//! for.  `encode/skolem_candidates.rs` is the immediate precedent: the same
//! concern (a term walk, made explicit-stack and exhaustive) split into the
//! same `encode/` child-module directory.

use super::*;

#[cfg(test)]
mod tests;

impl Solver {
    /// Claim `term_id` as *fully traversed* by the theory-variable walk.
    ///
    /// Returns `false` when the term was already claimed, in which case the
    /// caller must not re-visit its children: the memo exists precisely so a
    /// sub-expression shared by several parent constraints is walked once
    /// instead of once per parent (an O(depth) re-walk each time).
    ///
    /// The claim is journalled so a `pop` un-claims it — a term claimed inside
    /// a scope that is later retracted must be walkable again, otherwise the
    /// next assertion mentioning it would silently skip registering its
    /// variables.
    fn claim_tracked_compound(&mut self, term_id: TermId) -> bool {
        if !self.tracked_compound_terms.insert(term_id) {
            return false;
        }
        self.trail
            .push(TrailOp::TrackedCompoundAdded { term: term_id });
        true
    }

    /// Track theory variables in a term for model extraction.
    ///
    /// Scans a term to find Int/Real/BV variables and registers them with the
    /// arithmetic and bitvector solvers, so that every theory atom the encoder
    /// hands to SAT has an operand the theory solvers can actually assign.
    ///
    /// Compound terms that have already been fully traversed are recorded in
    /// `tracked_compound_terms` to avoid redundant O(depth) re-walks when the
    /// same sub-expression appears in multiple parent constraints.
    ///
    /// # Explicit stack, not native recursion
    ///
    /// The walk uses an explicit `Vec<TermId>` worklist.  Native recursion here
    /// was a latent process abort: a stack overflow is not a panic, it is a
    /// fatal `SIGSEGV`-class abort that no `Result` can report and
    /// `catch_unwind` cannot intercept, so an embedder calling OxiZ on a worker
    /// thread with a conventional stack would lose the whole process rather
    /// than get an answer.  Measured on a 1 MiB thread before this conversion,
    /// the recursive version survived a chain of ~4370 levels at `opt-level =
    /// 1` (~240 bytes per native frame) but only ~1556 at `opt-level = 0`
    /// (~671 bytes per frame) — the latter *below* the depth gate's value at
    /// the time (2000; see [`super::super::ENCODE_DEPTH_LIMIT`], since
    /// lowered), so an assertion that passed
    /// `Solver::term_exceeds_encode_depth` could still abort here.  The walk is
    /// additionally reachable from paths that never consult that gate at all
    /// (MBQI instantiation and the axiom passes call [`Solver::encode`]
    /// directly), so the gate was never a stack bound in the first place.
    ///
    /// A plain worklist suffices — no resume-state frame enum is needed, unlike
    /// [`oxiz_core::ast::TermManager::rebuild_substituted`]'s
    /// `Expand`/`Combine` frames: every arm below performs its whole effect
    /// (memo claim, trail push, interning, flag set) *before* descending, and
    /// no arm has work to do after its children or consumes anything the
    /// children produced.  Child order is still preserved exactly — children
    /// are pushed in reverse so they pop left-to-right — because the trail this
    /// walk appends to is observable (`pop` replays it) and
    /// `ArithSolver::intern` hands out `VarId`s in call order.
    ///
    /// # Exhaustive by construction
    ///
    /// The match has **no `_` arm**: all 111 [`TermKind`] variants are listed,
    /// so a future variant is a compile error here rather than a silent new
    /// gap.  The variants that are deliberately *not* descended into are
    /// grouped in the final arm, which documents why each family is a no-op.
    pub(super) fn track_theory_vars(&mut self, term_id: TermId, manager: &TermManager) {
        let mut stack: Vec<TermId> = vec![term_id];

        while let Some(current) = stack.pop() {
            let Some(term) = manager.get(current) else {
                continue;
            };

            match &term.kind {
                TermKind::Var(_) => {
                    // Found a variable - check its sort and track appropriately
                    let is_int = term.sort == manager.sorts.int_sort;
                    let is_real = term.sort == manager.sorts.real_sort;

                    if is_int || is_real {
                        if !self.arith_terms.contains(&current) {
                            self.arith_terms.insert(current);
                            self.trail.push(TrailOp::ArithTermAdded { term: current });
                            self.arith.intern(current);
                        }
                    } else if let Some(sort) = manager.sorts.get(term.sort)
                        && sort.is_bitvec()
                        && !self.bv_terms.contains(&current)
                    {
                        self.bv_terms.insert(current);
                        self.trail.push(TrailOp::BvTermAdded { term: current });
                        if let Some(width) = sort.bitvec_width() {
                            self.bv.new_bv(current, width);
                        }
                        // Also intern in ArithSolver for BV comparison constraints
                        // (BV comparisons are handled as bounded integer arithmetic)
                        self.arith.intern(current);
                    }
                }

                // Compound terms: descend into every operand.  The memo claim
                // is what keeps a shared sub-DAG from being re-walked once per
                // parent edge.
                TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::And(args)
                | TermKind::Or(args) => {
                    if !self.claim_tracked_compound(current) {
                        continue;
                    }
                    for &arg in args.iter().rev() {
                        stack.push(arg);
                    }
                }
                TermKind::Sub(lhs, rhs)
                | TermKind::Eq(lhs, rhs)
                | TermKind::Lt(lhs, rhs)
                | TermKind::Le(lhs, rhs)
                | TermKind::Gt(lhs, rhs)
                | TermKind::Ge(lhs, rhs)
                | TermKind::BvAdd(lhs, rhs)
                | TermKind::BvSub(lhs, rhs)
                | TermKind::BvMul(lhs, rhs)
                | TermKind::BvAnd(lhs, rhs)
                | TermKind::BvOr(lhs, rhs)
                | TermKind::BvXor(lhs, rhs)
                | TermKind::BvUlt(lhs, rhs)
                | TermKind::BvUle(lhs, rhs)
                | TermKind::BvSlt(lhs, rhs)
                | TermKind::BvSle(lhs, rhs)
                // Shifts and concatenation: descend so leaf operands are
                // tracked (and thus get model values for counterexamples).
                | TermKind::BvShl(lhs, rhs)
                | TermKind::BvLshr(lhs, rhs)
                | TermKind::BvAshr(lhs, rhs)
                | TermKind::BvConcat(lhs, rhs) => {
                    if !self.claim_tracked_compound(current) {
                        continue;
                    }
                    stack.push(*rhs);
                    stack.push(*lhs);
                }
                // Bit extraction: descend into the single source operand.
                TermKind::BvExtract { arg, .. } => {
                    if !self.claim_tracked_compound(current) {
                        continue;
                    }
                    stack.push(*arg);
                }
                // BV arithmetic operations (division/remainder).
                // These need the has_bv_arith_ops flag for conflict detection.
                TermKind::BvUdiv(lhs, rhs)
                | TermKind::BvSdiv(lhs, rhs)
                | TermKind::BvUrem(lhs, rhs)
                | TermKind::BvSrem(lhs, rhs) => {
                    if !self.claim_tracked_compound(current) {
                        continue;
                    }
                    self.has_bv_arith_ops = true;
                    stack.push(*rhs);
                    stack.push(*lhs);
                }
                TermKind::Neg(arg) | TermKind::Not(arg) | TermKind::BvNot(arg) => {
                    if !self.claim_tracked_compound(current) {
                        continue;
                    }
                    stack.push(*arg);
                }
                TermKind::Ite(cond, then_br, else_br) => {
                    if !self.claim_tracked_compound(current) {
                        continue;
                    }
                    // An Int/Real-sorted conditional is an arithmetic *atom* in
                    // its own right (this is how `abs`/`min`/`max` reach the
                    // solver), so register it alongside its operands;
                    // `instantiate_arith_axioms` then supplies
                    // `c => ite = t` / `¬c => ite = e`.
                    self.register_arith_atom(current, term.sort, manager);
                    stack.push(*else_br);
                    stack.push(*then_br);
                    stack.push(*cond);
                }
                // Integer `div`/`mod`: opaque arithmetic atoms whose meaning is
                // supplied by the defining axioms in `arith_axioms`.
                // Registering them here is what makes the axiom pass see them
                // at all, and descending is what gives the dividend a theory
                // variable — without it, `(>= (mod i0 7) 7)` left `i0`
                // completely untracked.
                TermKind::Div(lhs, rhs) | TermKind::Mod(lhs, rhs) => {
                    if !self.claim_tracked_compound(current) {
                        continue;
                    }
                    self.register_arith_atom(current, term.sort, manager);
                    stack.push(*rhs);
                    stack.push(*lhs);
                }

                // Uninterpreted function application: if the sort is numeric
                // (Int or Real), treat the whole application as an opaque
                // arithmetic variable.  This supports the UFLIA / UFLRA
                // combination: `f(k)` appearing in `(> (f k) 10)` must be
                // tracked so that its model value is extracted and available to
                // the MBQI counterexample generator.
                //
                // We do NOT descend into the arguments here -- argument terms
                // are arithmetic values passed to an opaque symbol, not
                // arithmetic variables in their own right within this
                // constraint.  (They will be tracked separately when they
                // appear in other constraints.)
                //
                // A *nested* numeric application such as `f(g(a))` is
                // registered on the same footing as a flat one.  It used to be
                // skipped whenever its argument `g(a)` already had an
                // arithmetic model value, on the grounds that EUF's congruence
                // `g(a) = v ⇒ f(g(a)) = f(v)` would then fight arith's
                // independent value for `f(g(a))` and manufacture a
                // theory-combination conflict.  That fight was not inherent: it
                // came from `TheoryManager::model_based_combination` reporting a
                // mere *model disagreement* as a refutation, and from conflict
                // clauses that dropped the congruence-derived equality's
                // justification.  Both are fixed — a disagreement is now
                // resolved by handing the explained equality to the tableau —
                // so skipping the term buys nothing and costs correctness: an
                // unregistered term leaves its atom a free boolean, and
                // `(> (f a) (f b)) ∧ (> (f b) (f (f a))) ∧ (> (f (f a)) (f a))`
                // was answered `sat`.
                TermKind::Apply { .. } => {
                    let is_int = term.sort == manager.sorts.int_sort;
                    let is_real = term.sort == manager.sorts.real_sort;
                    if (is_int || is_real) && !self.arith_terms.contains(&current) {
                        self.arith_terms.insert(current);
                        self.trail.push(TrailOp::ArithTermAdded { term: current });
                        self.arith.intern(current);
                    }
                }

                // Array select with numeric sort: `(select a i) : Int/Real` is
                // an opaque arithmetic variable -- the array theory handles
                // equality propagation for equal indices, while arithmetic sees
                // the result as an unconstrained integer/real.  We register it
                // here so that constraints like `(> (select a 0) 7)` are tracked
                // by the arithmetic solver and model values are extracted
                // correctly.
                // Array write: `(store a i v)` carries no arithmetic variable
                // of its own (its sort is `Array`, never `Int`/`Real`), but its
                // *presence* is what tells `check_core` that the lazy
                // array-axiom refinement loop has work to do.  Without this
                // flag a store-only formula — one that never reads, e.g. the
                // store-commutativity family
                // `(not (= (store (store a 1 x) 2 y) (store (store a 2 y) 1 x)))`
                // — skips `instantiate_array_axioms` entirely, so
                // extensionality is never instantiated, the array disequality
                // stays a free Boolean and an `unsat` comes back `sat`.
                //
                // Descent stays as it was (the operands of a store are not
                // arithmetic variables of any constraint this walk is
                // registering); only the flag is new.
                TermKind::Store(_, _, _) => {
                    self.has_array_ops = true;
                }

                TermKind::Select(_, _) => {
                    self.has_array_ops = true;
                    let is_int = term.sort == manager.sorts.int_sort;
                    let is_real = term.sort == manager.sorts.real_sort;
                    if (is_int || is_real) && !self.arith_terms.contains(&current) {
                        self.arith_terms.insert(current);
                        self.trail.push(TrailOp::ArithTermAdded { term: current });
                        self.arith.intern(current);
                    }
                }

                // Datatype accessor of numeric sort: an opaque arithmetic atom,
                // for the same reason as `select` above.  Registering it is what
                // gives `(head l)` an arithmetic model value and lets two
                // occurrences of the one term be forced to agree.  The accessor
                // *argument* is a datatype term, not an arithmetic one, so there
                // is nothing to descend into.
                TermKind::DtSelector { .. } => {
                    self.register_arith_atom(current, term.sort, manager);
                }

                // ---------------------------------------------------------
                // Deliberate no-ops.  Listed explicitly (rather than left to a
                // `_` catch-all) so that adding a `TermKind` variant is a
                // compile error here instead of a silent new gap.  Nothing
                // below claims the memo, so nothing below is journalled: these
                // arms leave the solver state exactly as they found it, which
                // is why revisiting one through a second parent edge is
                // harmless.
                //
                // * Leaves — `True`/`False`, the numeric and bitvector
                //   constants, string literals and the floating-point literal
                //   forms — have no children and no theory *variable* to
                //   register: a constant needs no model value, it *is* one.
                //   (`extract_linear_terms` folds `IntConst`/`RealConst`/
                //   `BitVecConst` straight into a constraint's constant term.)
                //
                // * `Xor`, `Implies`, `Distinct`, `Let`, `Match`,
                //   `Forall`, `Exists`, `DtConstructor`, `DtTester`, and the
                //   string and floating-point operations are *not* leaves, yet
                //   are not descended into either.  This mirrors
                //   `Solver::extract_linear_terms`, which treats exactly the
                //   same set as non-linear: a comparison whose operand is one of
                //   these fails the linear parse, so no arithmetic constraint is
                //   built from it and the `encode_guards` honesty gate is what
                //   reports the resulting incompleteness (`Unknown`) rather than
                //   a model.  Registering theory variables underneath such an
                //   operand would therefore not make any atom decidable — the
                //   parse would still fail — so the walk stops here.  The
                //   Boolean structure of `Xor`/`Implies`/`Distinct`/`Let` is
                //   reached anyway, by `Solver::encode_depth`'s own descent,
                //   which calls this walk again on each theory atom it finds
                //   inside them; and quantifier bodies are handled by the MBQI
                //   and E-matching registration passes, not here (a bound
                //   variable has no ground model value to extract).
                //   `oxiz-solver/src/solver/encode/track_theory_vars/tests.rs`
                //   pins this set, so widening it is a deliberate, tested
                //   decision rather than an accident.
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_)
                | TermKind::FpLit { .. }
                | TermKind::FpPlusInfinity { .. }
                | TermKind::FpMinusInfinity { .. }
                | TermKind::FpPlusZero { .. }
                | TermKind::FpMinusZero { .. }
                | TermKind::FpNaN { .. }
                | TermKind::Xor(_, _)
                | TermKind::Implies(_, _)
                | TermKind::Distinct(_)
                | TermKind::Let { .. }
                | TermKind::Match { .. }
                | TermKind::Forall { .. }
                | TermKind::Exists { .. }
                | TermKind::DtConstructor { .. }
                | TermKind::DtTester { .. }
                | TermKind::StrConcat(_, _)
                | TermKind::StrLen(_)
                | TermKind::StrSubstr(_, _, _)
                | TermKind::StrAt(_, _)
                | TermKind::StrContains(_, _)
                | TermKind::StrPrefixOf(_, _)
                | TermKind::StrSuffixOf(_, _)
                | TermKind::StrIndexOf(_, _, _)
                | TermKind::StrReplace(_, _, _)
                | TermKind::StrReplaceAll(_, _, _)
                | TermKind::StrReplaceRe(_, _, _)
                | TermKind::StrReplaceReAll(_, _, _)
                | TermKind::StrToInt(_)
                | TermKind::IntToStr(_)
                | TermKind::StrInRe(_, _)
                | TermKind::StrLt(_, _)
                | TermKind::StrLe(_, _)
                | TermKind::StrToCode(_)
                | TermKind::StrFromCode(_)
                | TermKind::FpAbs(_)
                | TermKind::FpNeg(_)
                | TermKind::FpSqrt(_, _)
                | TermKind::FpRoundToIntegral(_, _)
                | TermKind::FpAdd(_, _, _)
                | TermKind::FpSub(_, _, _)
                | TermKind::FpMul(_, _, _)
                | TermKind::FpDiv(_, _, _)
                | TermKind::FpRem(_, _)
                | TermKind::FpMin(_, _)
                | TermKind::FpMax(_, _)
                | TermKind::FpLeq(_, _)
                | TermKind::FpLt(_, _)
                | TermKind::FpGeq(_, _)
                | TermKind::FpGt(_, _)
                | TermKind::FpEq(_, _)
                | TermKind::FpFma(_, _, _, _)
                | TermKind::FpIsNormal(_)
                | TermKind::FpIsSubnormal(_)
                | TermKind::FpIsZero(_)
                | TermKind::FpIsInfinite(_)
                | TermKind::FpIsNaN(_)
                | TermKind::FpIsNegative(_)
                | TermKind::FpIsPositive(_)
                | TermKind::FpToFp { .. }
                | TermKind::FpToSBV { .. }
                | TermKind::FpToUBV { .. }
                | TermKind::FpToReal(_)
                | TermKind::RealToFp { .. }
                | TermKind::SBVToFp { .. }
                | TermKind::UBVToFp { .. } => {}
            }
        }
    }
}
