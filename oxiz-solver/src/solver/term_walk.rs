//! Generic structural term traversal helpers.
//!
//! These are theory-agnostic and are shared by the FP and string atom-presence
//! detectors (`check_fp.rs`, `check_string.rs`). Keeping the walker here avoids a
//! cross-file `Solver::` coupling between the FP-specific and string-specific
//! modules and keeps each theory file focused on its own reasoning.
//!
//! Two walks live here and they must never be confused:
//!
//! * [`collect_structural_children`] enumerates *every* sub-term. It is for term
//!   **discovery** (does this problem mention an FP atom at all?) and says
//!   nothing about whether a sub-term is asserted.
//! * [`asserted_children`] enumerates only the sub-terms that are
//!   **unconditionally asserted**. It is the only walk a definite-conflict
//!   collector may use to reach new facts.

use oxiz_core::ast::{TermId, TermKind};

/// Push every immediate sub-term of `kind` onto `out`.
///
/// This is a fully generic structural walk used by the FP and string
/// atom-presence detectors; it deliberately traverses through *all*
/// compound kinds (Boolean, arithmetic, bit-vector, array, datatype,
/// quantifier, FP, and string) so that a theory atom nested arbitrarily
/// deep is still discovered.
pub(crate) fn collect_structural_children(kind: &TermKind, out: &mut Vec<TermId>) {
    match kind {
        // Single sub-term
        TermKind::Not(a)
        | TermKind::Neg(a)
        | TermKind::BvNot(a)
        | TermKind::StrLen(a)
        | TermKind::StrToInt(a)
        | TermKind::IntToStr(a)
        | TermKind::StrToCode(a)
        | TermKind::StrFromCode(a)
        | TermKind::FpAbs(a)
        | TermKind::FpNeg(a)
        | TermKind::FpToReal(a)
        | TermKind::FpIsNormal(a)
        | TermKind::FpIsSubnormal(a)
        | TermKind::FpIsZero(a)
        | TermKind::FpIsInfinite(a)
        | TermKind::FpIsNaN(a)
        | TermKind::FpIsNegative(a)
        | TermKind::FpIsPositive(a)
        | TermKind::FpSqrt(_, a)
        | TermKind::FpRoundToIntegral(_, a) => out.push(*a),
        TermKind::BvExtract { arg, .. }
        | TermKind::DtTester { arg, .. }
        | TermKind::DtSelector { arg, .. }
        | TermKind::FpToFp { arg, .. }
        | TermKind::FpToSBV { arg, .. }
        | TermKind::FpToUBV { arg, .. }
        | TermKind::RealToFp { arg, .. }
        | TermKind::SBVToFp { arg, .. }
        | TermKind::UBVToFp { arg, .. } => out.push(*arg),
        // Two sub-terms
        TermKind::Xor(a, b)
        | TermKind::Implies(a, b)
        | TermKind::Eq(a, b)
        | TermKind::Sub(a, b)
        | TermKind::Div(a, b)
        | TermKind::Mod(a, b)
        | TermKind::Lt(a, b)
        | TermKind::Le(a, b)
        | TermKind::Gt(a, b)
        | TermKind::Ge(a, b)
        | TermKind::BvConcat(a, b)
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
        | TermKind::BvSle(a, b)
        | TermKind::Select(a, b)
        | TermKind::StrConcat(a, b)
        | TermKind::StrAt(a, b)
        | TermKind::StrContains(a, b)
        | TermKind::StrPrefixOf(a, b)
        | TermKind::StrSuffixOf(a, b)
        | TermKind::StrInRe(a, b)
        | TermKind::StrLt(a, b)
        | TermKind::StrLe(a, b)
        | TermKind::FpRem(a, b)
        | TermKind::FpMin(a, b)
        | TermKind::FpMax(a, b)
        | TermKind::FpLeq(a, b)
        | TermKind::FpLt(a, b)
        | TermKind::FpGeq(a, b)
        | TermKind::FpGt(a, b)
        | TermKind::FpEq(a, b) => {
            out.push(*a);
            out.push(*b);
        }
        TermKind::FpAdd(_, a, b)
        | TermKind::FpSub(_, a, b)
        | TermKind::FpMul(_, a, b)
        | TermKind::FpDiv(_, a, b) => {
            out.push(*a);
            out.push(*b);
        }
        // Three sub-terms
        TermKind::Ite(a, b, c)
        | TermKind::Store(a, b, c)
        | TermKind::StrSubstr(a, b, c)
        | TermKind::StrIndexOf(a, b, c)
        | TermKind::StrReplace(a, b, c)
        | TermKind::StrReplaceAll(a, b, c)
        | TermKind::StrReplaceRe(a, b, c)
        | TermKind::StrReplaceReAll(a, b, c) => {
            out.push(*a);
            out.push(*b);
            out.push(*c);
        }
        TermKind::FpFma(_, a, b, c) => {
            out.push(*a);
            out.push(*b);
            out.push(*c);
        }
        // Variadic
        TermKind::And(args)
        | TermKind::Or(args)
        | TermKind::Distinct(args)
        | TermKind::Add(args)
        | TermKind::Mul(args)
        | TermKind::Apply { args, .. }
        | TermKind::DtConstructor { args, .. } => {
            for &arg in args.iter() {
                out.push(arg);
            }
        }
        // Quantifiers / binders
        TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => out.push(*body),
        TermKind::Let { bindings, body } => {
            for &(_, value) in bindings.iter() {
                out.push(value);
            }
            out.push(*body);
        }
        TermKind::Match { scrutinee, cases } => {
            out.push(*scrutinee);
            for case in cases.iter() {
                out.push(case.body);
            }
        }
        // Leaves (constants, variables, FP special values) — no sub-terms.
        _ => {}
    }
}

/// The sub-terms of `kind` that are **unconditionally asserted**, given that
/// `kind` itself is asserted with polarity `positive`, each paired with its own
/// polarity.
///
/// # Why this exists
///
/// The `check_*.rs` modules host per-theory *definite-conflict* collectors: they
/// harvest facts out of the assertion set and, when a check fires, the solver
/// answers `Unsat` outright. Such a collector may only ever record facts that
/// the assertion set genuinely entails, so it may only descend through nodes
/// that preserve unconditional assertedness. Getting that wrong yields a false
/// `Unsat` on a satisfiable formula — a soundness bug that has now been found
/// independently in the string, bit-vector, datatype, array and floating-point
/// collectors. This function is the single place that rule is written down.
///
/// # The rule
///
/// * `And` at positive polarity — every conjunct is asserted, at positive
///   polarity.
/// * `And` at **negative** polarity — *nothing*. `(not (and a b))` is
///   `(or (not a) (not b))`, a disjunction, so neither conjunct is entailed.
///   This is the de Morgan trap that a naive `in_positive_context` pass-through
///   falls into.
/// * `Or` at negative polarity — every disjunct is asserted negatively, since
///   `(not (or a b))` is `(and (not a) (not b))`.
/// * `Or` at positive polarity — *nothing*; a disjunct is conditional.
/// * `Not` — the body, with the polarity flipped.
/// * Everything else — *nothing*. In particular `Eq`, `Implies`, `Xor` and
///   `Ite` are all polarity boundaries. `Eq` deserves the emphasis: this AST has
///   no `Iff`, so a Boolean `(= a b)` is a `TermKind::Eq`, and it is satisfied
///   just as well with *both* operands false — neither operand is entailed.
///
/// A collector that needs to reach conditional sub-terms for a different
/// purpose (enumerating conversion terms, discovering theory atoms) must use a
/// separate walk that cannot write into the conflict maps — see
/// `check_array.rs`'s `collect_facts` flag and `check_fp.rs`'s
/// `collect_fp_constraints_extended_recurse`.
pub(super) fn asserted_children(kind: &TermKind, positive: bool) -> Vec<(TermId, bool)> {
    match kind {
        TermKind::And(args) if positive => args.iter().map(|&arg| (arg, true)).collect(),
        TermKind::Or(args) if !positive => args.iter().map(|&arg| (arg, false)).collect(),
        TermKind::Not(inner) => vec![(*inner, !positive)],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::asserted_children;
    use oxiz_core::ast::{TermKind, TermManager};

    /// `And` hands out its conjuncts only while the polarity is positive:
    /// `(not (and a b))` entails neither `a` nor `b`.
    #[test]
    fn and_conjuncts_are_asserted_only_at_positive_polarity() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let a = manager.mk_var("a", bool_sort);
        let b = manager.mk_var("b", bool_sort);
        let conj = manager.mk_and(vec![a, b]);
        let kind = manager.get(conj).map(|t| t.kind.clone()).expect("term");

        assert_eq!(asserted_children(&kind, true), vec![(a, true), (b, true)]);
        assert!(asserted_children(&kind, false).is_empty());
    }

    /// `Or` is the mirror image: its disjuncts are entailed only under a `Not`.
    #[test]
    fn or_disjuncts_are_asserted_only_at_negative_polarity() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let a = manager.mk_var("a", bool_sort);
        let b = manager.mk_var("b", bool_sort);
        let disj = manager.mk_or(vec![a, b]);
        let kind = manager.get(disj).map(|t| t.kind.clone()).expect("term");

        assert!(asserted_children(&kind, true).is_empty());
        assert_eq!(
            asserted_children(&kind, false),
            vec![(a, false), (b, false)]
        );
    }

    /// `Not` flips the polarity and always yields its body.
    #[test]
    fn not_flips_polarity() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let a = manager.mk_var("a", bool_sort);
        let negated = manager.mk_not(a);
        let kind = manager.get(negated).map(|t| t.kind.clone()).expect("term");

        assert_eq!(asserted_children(&kind, true), vec![(a, false)]);
        assert_eq!(asserted_children(&kind, false), vec![(a, true)]);
    }

    /// `Eq`, `Implies`, `Xor` and `Ite` are polarity boundaries at *either*
    /// polarity — nothing below them is entailed.
    #[test]
    fn polarity_boundaries_yield_nothing() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let a = manager.mk_var("a", bool_sort);
        let b = manager.mk_var("b", bool_sort);
        let c = manager.mk_var("c", bool_sort);

        // A Bool-sorted `mk_eq` really is a `TermKind::Eq` — this AST has no
        // `Iff`, which is exactly why the boundary is easy to miss.
        let equality = manager.mk_eq(a, b);
        assert!(matches!(
            manager.get(equality).map(|t| &t.kind),
            Some(TermKind::Eq(_, _))
        ));

        for boundary in [
            equality,
            manager.mk_implies(a, b),
            manager.mk_xor(a, b),
            manager.mk_ite(a, b, c),
        ] {
            let kind = manager.get(boundary).map(|t| t.kind.clone()).expect("term");
            assert!(asserted_children(&kind, true).is_empty());
            assert!(asserted_children(&kind, false).is_empty());
        }
    }
}
