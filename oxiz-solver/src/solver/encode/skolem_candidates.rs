//! Skolem-candidate collection for MBQI cross-quantifier instantiation.
//!
//! Split out of `encode.rs` into this child module once making the walk
//! below exhaustive over every [`TermKind`] variant grew it past what
//! `encode.rs` -- already close to the workspace's 2000-line-per-file
//! ceiling -- had room for. `encode_guards.rs` (a sibling of `encode.rs`
//! carrying its own split-out concerns) is the precedent for splitting a
//! concern out of `encode.rs` this way; this one specifically goes in the
//! `encode/` child-module directory per this change's own instructions.

use super::*;

impl Solver {
    /// Walk a (possibly Skolemized) quantifier term and collect every `Apply`
    /// term whose function name starts with `"sk"` or `"skf"` as an MBQI
    /// instantiation candidate.
    ///
    /// These Skolem function applications (e.g. `sk!0(x)`) must be in the
    /// candidate pool so that MBQI can instantiate *other* universal
    /// quantifiers with Skolem terms, enabling cross-quantifier
    /// contradictions (e.g. a pigeonhole-style argument that only closes once
    /// a Skolem witness introduced by one quantifier is substituted into a
    /// different one).
    ///
    /// See [`Self::collect_skolem_candidates_rec`] for the traversal itself,
    /// including why it is an *exhaustive* match with no wildcard arm and why
    /// it deliberately does not look inside `Forall`/`Exists` `patterns`.
    pub(super) fn collect_skolem_candidates(&mut self, term: TermId, manager: &TermManager) {
        let mut visited = FxHashSet::default();
        self.collect_skolem_candidates_rec(term, manager, &mut visited);
    }

    /// Iterative worklist behind [`Self::collect_skolem_candidates`].
    ///
    /// A term built directly through the `TermManager` builder API can nest
    /// arbitrarily deep -- there is no parser-side cap once the builder is
    /// called directly -- and this walk is reachable from `check_sat`. Native
    /// recursion here would reach a native stack overflow (a fatal process
    /// abort no `Result` can report) on a sufficiently deep term; the
    /// explicit `Vec<TermId>` worklist removes that dependency on the native
    /// call stack, while `visited` (threaded by the caller across the whole
    /// walk) dedupes a shared sub-DAG so no node is processed twice.
    ///
    /// ### Exhaustive by construction
    ///
    /// This match has **no `_` arm**: every [`TermKind`] variant is listed
    /// explicitly, either pushing its children onto the worklist or (for
    /// genuinely childless leaves) doing nothing in a grouped arm. Before this
    /// change, the walk covered only `Apply`, `Forall`/`Exists`, `And`/`Or`,
    /// `Not`/`Neg`, `Implies`/`Eq`/`Lt`/`Le`/`Gt`/`Ge`/`Sub`/`Div`/`Mod`,
    /// `Add`/`Mul`, `Ite`, and `Select`/`Store`; everything else -- every
    /// BitVector operation, every FloatingPoint operation, every String
    /// operation, `Distinct`, `Let`, `Match`, `Xor`, and the three datatype
    /// constructor/tester/selector forms -- fell into a `_ => {}` catch-all
    /// and was silently never descended into. A Skolem application nested
    /// under any of those (`(bvadd (sk!0 x) y)`, `(distinct (sk!1 x) y)`,
    /// `(let ((a (sk!2 x))) ...)`, ...) was therefore never found as an MBQI
    /// candidate -- the exact "unhandled input silently dropped" shape this
    /// change closes. Listing every arm explicitly means a *future*
    /// `TermKind` variant is a compile error here, not a silent new gap --
    /// the same discipline `TermManager::rebuild_substituted` (Reference:
    /// `oxiz-core/src/ast/manager/query/substitute.rs`) already applies, for
    /// the identical reason.
    ///
    /// [`oxiz_core::ast::traversal::get_children`] offers a uniform child
    /// list for every `TermKind` and was considered as a way to avoid
    /// hand-enumerating the arms below. It is deliberately **not** used here:
    /// delegating the generic case to it (`other => get_children(other)...`)
    /// would itself be a wildcard arm from this match's point of view, so a
    /// newly added `TermKind` variant would silently keep compiling as long
    /// as `get_children` (a different module, in a crate this change does not
    /// own or modify) had *already* been updated for it -- the exhaustiveness
    /// guarantee this change is meant to provide would then live in a file
    /// outside this change's control instead of here. Hand-enumerating keeps
    /// the guarantee local and self-evident from this file alone, matching
    /// `rebuild_substituted`'s own choice to do the same rather than delegate
    /// to `get_children`.
    ///
    /// ### Quantifier `patterns` (triggers) are deliberately not descended into
    ///
    /// `Forall`/`Exists` push only `body`, never `patterns`. This is a
    /// deliberate decision, not an oversight:
    ///
    /// * The registration path that reaches this walk
    ///   (`Solver::register_asserted_forall`, in `encode.rs`) already feeds
    ///   every trigger term to `MBQIIntegration::collect_ground_terms`
    ///   separately (once for the top-level quantifier's own `patterns`, both
    ///   here and in `Solver::register_asserted_quantifiers`), which seeds
    ///   `extra_candidates` from patterns using a deliberately *ground-only*
    ///   filter: a subterm is registered only if its entire subtree contains
    ///   no `Var` node at all. Patterns are therefore not an unhandled input
    ///   here; they go through a different, already-existing, sound channel,
    ///   and adding them here too would at best be redundant.
    /// * Worse than redundant: a trigger is only useful for E-matching if it
    ///   mentions the quantifier's own bound variables (that is the entire
    ///   point of a trigger), so a Skolem application inside a pattern is, in
    ///   the overwhelmingly common case, *not* ground -- e.g. a pattern
    ///   `(sk!0 x)` on `forall ((x Int))` still contains the bound `x` as a
    ///   plain `TermKind::Var` node. Registering it via `add_candidate` --
    ///   which, unlike `collect_ground_terms`, does not check groundness --
    ///   would splice a term containing a bound-variable-shaped `Var` node
    ///   into `extra_candidates`, from where it can be substituted wholesale
    ///   into a *different* quantifier's body. That is a real soundness
    ///   hazard (the stray `Var` node aliases, via hash-consing, with any
    ///   unrelated same-named same-sort variable elsewhere in the problem),
    ///   not merely a missed optimization.
    /// * A well-formed trigger is, by construction of both this codebase's
    ///   Skolemization and of pattern selection generally, a subterm *also
    ///   reachable from the body* it was lifted from -- and because
    ///   `TermManager` hash-conses structurally identical terms to the same
    ///   `TermId`, such a term is not just semantically but *referentially*
    ///   the same node the body walk below already visits (so it is already
    ///   found, via `body`, without needing `patterns` at all). A Skolem
    ///   application that exists **only** inside a pattern, with no
    ///   occurrence anywhere in the body, would be a degenerate annotation;
    ///   this codebase's own quantifier/Skolemization machinery does not
    ///   produce one. The redundancy argument plus the soundness hazard above
    ///   together justify never walking `patterns` here.
    pub(super) fn collect_skolem_candidates_rec(
        &mut self,
        term: TermId,
        manager: &TermManager,
        visited: &mut FxHashSet<TermId>,
    ) {
        let mut stack: Vec<TermId> = vec![term];

        while let Some(term) = stack.pop() {
            if !visited.insert(term) {
                continue;
            }
            let Some(t) = manager.get(term) else {
                continue;
            };
            match &t.kind {
                // Function application: the one case with a side effect
                // beyond traversal -- register the term as an MBQI candidate
                // when its function name looks like a Skolem symbol, then
                // recurse into its arguments like any other n-ary node.
                TermKind::Apply { func, args } => {
                    let fname = manager.resolve_str(*func);
                    if oxiz_core::smtlib::is_reserved_tag(fname, "sk")
                        || oxiz_core::smtlib::is_reserved_tag(fname, "skf")
                    {
                        // Register the whole application as a candidate
                        self.mbqi.add_candidate(term, t.sort);
                    }
                    for &arg in args.iter().rev() {
                        stack.push(arg);
                    }
                }

                // Quantifiers: `body` only -- see "Quantifier patterns" above.
                TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => {
                    stack.push(*body);
                }

                // Let: every bound value, then the body.
                TermKind::Let { bindings, body } => {
                    stack.push(*body);
                    for (_, value) in bindings.iter().rev() {
                        stack.push(*value);
                    }
                }

                // Match: the scrutinee and every case body. A case's
                // *pattern* binds constructor field names (`Spur`s), not
                // `TermId`s, so there is nothing else to push.
                TermKind::Match { scrutinee, cases } => {
                    for case in cases.iter().rev() {
                        stack.push(case.body);
                    }
                    stack.push(*scrutinee);
                }

                // Core Boolean / equality / arithmetic connectives.
                TermKind::And(args) | TermKind::Or(args) => {
                    for &a in args.iter().rev() {
                        stack.push(a);
                    }
                }
                TermKind::Not(a) | TermKind::Neg(a) => {
                    stack.push(*a);
                }
                TermKind::Xor(a, b)
                | TermKind::Implies(a, b)
                | TermKind::Eq(a, b)
                | TermKind::Lt(a, b)
                | TermKind::Le(a, b)
                | TermKind::Gt(a, b)
                | TermKind::Ge(a, b)
                | TermKind::Sub(a, b)
                | TermKind::Div(a, b)
                | TermKind::Mod(a, b) => {
                    stack.push(*b);
                    stack.push(*a);
                }
                TermKind::Add(args) | TermKind::Mul(args) | TermKind::Distinct(args) => {
                    for &a in args.iter().rev() {
                        stack.push(a);
                    }
                }
                TermKind::Ite(c, t_br, e) => {
                    stack.push(*e);
                    stack.push(*t_br);
                    stack.push(*c);
                }

                // Arrays.
                TermKind::Select(a, i) => {
                    stack.push(*i);
                    stack.push(*a);
                }
                TermKind::Store(a, i, v) => {
                    stack.push(*v);
                    stack.push(*i);
                    stack.push(*a);
                }

                // Datatypes.
                TermKind::DtConstructor { args, .. } => {
                    for &a in args.iter().rev() {
                        stack.push(a);
                    }
                }
                TermKind::DtTester { arg, .. } | TermKind::DtSelector { arg, .. } => {
                    stack.push(*arg);
                }

                // BitVector operations -- unary.
                TermKind::BvNot(a) => {
                    stack.push(*a);
                }
                TermKind::BvExtract { arg, .. } => {
                    stack.push(*arg);
                }
                // BitVector operations -- binary (includes comparisons and
                // the arithmetic-division/remainder family).
                TermKind::BvConcat(a, b)
                | TermKind::BvAnd(a, b)
                | TermKind::BvOr(a, b)
                | TermKind::BvXor(a, b)
                | TermKind::BvAdd(a, b)
                | TermKind::BvSub(a, b)
                | TermKind::BvMul(a, b)
                | TermKind::BvUdiv(a, b)
                | TermKind::BvSdiv(a, b)
                | TermKind::BvUrem(a, b)
                | TermKind::BvSrem(a, b)
                | TermKind::BvShl(a, b)
                | TermKind::BvLshr(a, b)
                | TermKind::BvAshr(a, b)
                | TermKind::BvUlt(a, b)
                | TermKind::BvUle(a, b)
                | TermKind::BvSlt(a, b)
                | TermKind::BvSle(a, b) => {
                    stack.push(*b);
                    stack.push(*a);
                }

                // String operations -- unary.
                TermKind::StrLen(a)
                | TermKind::StrToInt(a)
                | TermKind::IntToStr(a)
                | TermKind::StrToCode(a)
                | TermKind::StrFromCode(a) => {
                    stack.push(*a);
                }
                // String operations -- binary.
                TermKind::StrConcat(a, b)
                | TermKind::StrAt(a, b)
                | TermKind::StrContains(a, b)
                | TermKind::StrPrefixOf(a, b)
                | TermKind::StrSuffixOf(a, b)
                | TermKind::StrInRe(a, b)
                | TermKind::StrLt(a, b)
                | TermKind::StrLe(a, b) => {
                    stack.push(*b);
                    stack.push(*a);
                }
                // String operations -- ternary.
                TermKind::StrSubstr(a, b, c)
                | TermKind::StrIndexOf(a, b, c)
                | TermKind::StrReplace(a, b, c)
                | TermKind::StrReplaceAll(a, b, c)
                | TermKind::StrReplaceRe(a, b, c)
                | TermKind::StrReplaceReAll(a, b, c) => {
                    stack.push(*c);
                    stack.push(*b);
                    stack.push(*a);
                }

                // Floating-point operations -- unary (a leading `RoundingMode`
                // argument, where present, carries no `TermId` and is simply
                // not pushed).
                TermKind::FpAbs(a)
                | TermKind::FpNeg(a)
                | TermKind::FpSqrt(_, a)
                | TermKind::FpRoundToIntegral(_, a)
                | TermKind::FpIsNormal(a)
                | TermKind::FpIsSubnormal(a)
                | TermKind::FpIsZero(a)
                | TermKind::FpIsInfinite(a)
                | TermKind::FpIsNaN(a)
                | TermKind::FpIsNegative(a)
                | TermKind::FpIsPositive(a)
                | TermKind::FpToReal(a)
                | TermKind::FpToFp { arg: a, .. }
                | TermKind::FpToSBV { arg: a, .. }
                | TermKind::FpToUBV { arg: a, .. }
                | TermKind::RealToFp { arg: a, .. }
                | TermKind::SBVToFp { arg: a, .. }
                | TermKind::UBVToFp { arg: a, .. } => {
                    stack.push(*a);
                }
                // Floating-point operations -- binary.
                TermKind::FpAdd(_, a, b)
                | TermKind::FpSub(_, a, b)
                | TermKind::FpMul(_, a, b)
                | TermKind::FpDiv(_, a, b)
                | TermKind::FpRem(a, b)
                | TermKind::FpMin(a, b)
                | TermKind::FpMax(a, b)
                | TermKind::FpLeq(a, b)
                | TermKind::FpLt(a, b)
                | TermKind::FpGeq(a, b)
                | TermKind::FpGt(a, b)
                | TermKind::FpEq(a, b) => {
                    stack.push(*b);
                    stack.push(*a);
                }
                // Floating-point operations -- ternary (fused multiply-add).
                TermKind::FpFma(_, a, b, c) => {
                    stack.push(*c);
                    stack.push(*b);
                    stack.push(*a);
                }

                // Leaves: constants and variables have no children to recurse
                // into. Listed explicitly (rather than falling out of a
                // wildcard) so the match above stays exhaustive.
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::Var(_)
                | TermKind::StringLit(_)
                | TermKind::FpLit { .. }
                | TermKind::FpPlusInfinity { .. }
                | TermKind::FpMinusInfinity { .. }
                | TermKind::FpPlusZero { .. }
                | TermKind::FpMinusZero { .. }
                | TermKind::FpNaN { .. } => {}
            }
        }
    }
}
