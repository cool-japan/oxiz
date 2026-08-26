//! Bool/EUF encoding pre-passes that give congruence closure the facts it
//! needs to see through a non-Bool `ite` or a compound Bool argument.
//!
//! Split out of `encode.rs` into this child module, following the same
//! `encode/` convention as `track_theory_vars.rs`/`skolem_candidates.rs`.
//!
//! # Why these live here and not inside the generic Tseitin encoder
//!
//! Both passes work by hoisting a subterm out to a *fresh, globally scoped*
//! variable plus a defining side-condition, then conjoining the
//! side-condition onto the term being rewritten. That is only sound when the
//! whole rewritten conjunction is asserted as a hard, top-level fact: the
//! fresh variable's defining equation must hold in *every* model, not only
//! in the branch of a larger formula where the rewritten subterm happens to
//! be used positively. Folding the rewrite into [`Solver::encode_depth`]
//! instead -- which is polarity-sensitive and reused for sub-expressions
//! nested under `Not`/`Or`/an implication's antecedent -- would let the SAT
//! search satisfy the *negative* occurrence of the rewritten term by
//! assigning the fresh variable an incorrect value instead of by making the
//! real fact false, since a Tseitin `AND` gate's defining clauses only
//! constrain its conjuncts when the gate is forced *true*. So both passes
//! run exactly once, over the whole assertion, before any encoding starts
//! (see [`Solver::assert`]/[`Solver::assert_named`]), and the result is
//! asserted as a single top-level clause.
//!
//! [`Solver::encode_nonbool_ite_equality`] (below) is the
//! complementary *narrow* mechanism: a one-directional `eq_var -> branch
//! consequence` clause added from inside the generic recursive encoder. It
//! does not introduce a fresh variable and only reinforces an
//! already-determined `eq_var`, so it is safe under any polarity and also
//! covers the paths that never go through `Solver::assert` at all -- MBQI
//! instantiation and the axiom passes call [`Solver::encode`] directly on
//! ground lemmas.

use rustc_hash::FxHashSet;

use oxiz_core::ast::get_children;

use super::*;

/// Whether an `ite` of sort `sort` is [`Solver::eliminate_nonbool_ite`]'s
/// business at all.
///
/// EUF is the theory that treats a non-Bool `ite` as an opaque leaf with no
/// way back to its conditional-equality meaning, so hoisting it out to a
/// fresh, EUF-visible constant is only the *right* fix for a sort EUF
/// actually owns: an uninterpreted sort, or (redundantly but harmlessly,
/// since `arith_axioms.rs` already axiomatises these) `Int`/`Real`.
///
/// `BitVec`, `Array`, `String` and `FloatingPoint` sorts each have their own
/// theory-specific encoder that recurses through `ite` directly as part of
/// bit-blasting / axiom instantiation / ground solving -- and, critically,
/// some bit-vector operators (`bvsmod`, `bvcomp`, the rotates,
/// `zero_extend`/`sign_extend`) are *desugared into a `BitVec`-sorted `ite`*
/// by the term builder itself, so a real assertion can easily contain one
/// without the user ever writing `ite`. Replacing that `ite` with a fresh
/// opaque variable before the bit-blaster ever sees it deletes the very
/// structure the blaster recurses on: this was caught by
/// `test_bvsmod_symbolic_divisor_operand_unsat`/`test_bvsmod_controls_stay_sat`
/// going from correct to a false `sat` (the model no longer matched the
/// operation's reference semantics) the first time this pass ran
/// unconditionally.
pub(in crate::solver) fn needs_ite_elimination(sort: SortId, manager: &TermManager) -> bool {
    if sort == manager.sorts.bool_sort {
        return false;
    }
    match manager.sorts.get(sort) {
        Some(s) => {
            !(s.is_bitvec()
                || s.is_string()
                || s.is_float()
                || matches!(s.kind, oxiz_core::sort::SortKind::Array { .. }))
        }
        None => false,
    }
}

/// Collect every subterm of `term` reachable *without* descending into a
/// quantifier's bound body (`Forall`/`Exists`) or a `let`'s bindings/body.
///
/// Both rewrites in this module replace a subterm with a fresh, unbound
/// variable. That is unsound under a binder: a subterm that mentions the
/// bound variable denotes a different value at every instantiation, so one
/// global replacement cannot stand in for all of them. Treating a binder as
/// opaque here means neither pass ever looks inside one -- whatever ground
/// instances MBQI produces from it are handled when *they* are encoded
/// (through [`Solver::encode_nonbool_ite_equality`], which runs on every
/// encoded equality regardless of how it was reached).
///
/// Order is a stable pre-order (children pushed in reverse so they pop
/// left-to-right), matching this crate's other explicit-stack term walks, so
/// that a caller building a side-condition list from the result gets a
/// deterministic order rather than one that depends on hash iteration.
///
/// # Why `Let` stays opaque here
///
/// A vacuous `Let` wrapper used to silently disable both rewrites in this
/// module (and every other assert-time pre-pass) for the whole assertion
/// beneath it: the SMT-LIB parser wrapped each assertion in a `Let` whose
/// bindings it had *already substituted* into the body, so the node bound
/// nothing yet stopped this walk dead.  Two formulas differing only by an
/// unused `(let ((q 0)) ...)` got different verdicts, one of them a wrong
/// `sat`.
///
/// That is fixed at the source — the parser now returns the substituted body
/// directly and emits no `Let` at all (see `close_frame` in
/// `oxiz-core/src/smtlib/parser/terms.rs`) — deliberately *not* by making this
/// walk descend.  The rewrites this function feeds hoist a subterm to a fresh
/// **unbound** variable, which is genuinely unsound under a real binder: a
/// subterm mentioning the bound name denotes a different value per
/// instantiation, and one global replacement cannot stand for all of them.  So
/// `Let` keeps its place beside `Forall`/`Exists` below, and a `Let` reaching
/// here (only possible from a programmatic `TermManager::mk_let`, where the
/// body may really have free occurrences of the bound name) is correctly left
/// alone.
///
/// Arithmetic disequalities no longer depend on any such walk at all: the
/// trichotomy is attached to the `Eq` **atom** inside `encode` by
/// `Solver::add_numeric_trichotomy`, which a `Let` cannot hide from because the
/// atom must be encoded to get a SAT variable.
pub(in crate::solver) fn collect_ground_subterms(
    term: TermId,
    manager: &TermManager,
) -> Vec<TermId> {
    let mut out = Vec::new();
    let mut seen: FxHashSet<TermId> = FxHashSet::default();
    let mut stack = vec![term];
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        out.push(t);
        let Some(node) = manager.get(t) else {
            continue;
        };
        if matches!(
            node.kind,
            TermKind::Forall { .. } | TermKind::Exists { .. } | TermKind::Let { .. }
        ) {
            continue; // opaque: never descend into a binder's scope
        }
        let children = get_children(&node.kind);
        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }
    out
}

impl Solver {
    /// Eliminate every non-Bool `ite` reachable from `term` (outside any
    /// binder), returning an equisatisfiable term with the eliminations'
    /// side-conditions conjoined.
    ///
    /// EUF has no built-in notion of `ite`: left alone, `(ite c t e)` is
    /// interned as an opaque leaf, so the conditional equality it stands for
    /// -- `c -> a = t` and `~c -> a = e` for whatever context uses it --
    /// never reaches congruence closure. On mux-heavy `QF_UF` problems (the
    /// `firewire_tree` shape: `ite` chains selecting between arguments of
    /// uninterpreted functions under a web of equalities) that gap is enough
    /// to report a satisfiable model for a genuinely unsatisfiable formula.
    ///
    /// Each non-Bool `(ite c t e)` of sort `s` found by
    /// [`collect_ground_subterms`] is replaced everywhere by a fresh
    /// constant `v` of sort `s`. Two side-conditions are conjoined onto the
    /// result: `(=> c (= v t))` and `(=> (not c) (= v e))`. Once asserted,
    /// these pin `v` to whichever branch `c` selects, so EUF sees the
    /// conditional equality as an ordinary asserted fact and can merge (or
    /// refute) through it exactly as it would for a direct assertion.
    ///
    /// The fresh constant is keyed by the `ite` term's own [`TermId`]
    /// (hash-consed by name, via [`TermManager::mk_var`]), so a structurally
    /// identical `ite` reachable from a *different* assertion resolves to
    /// the same variable and the same side-conditions -- `mk_and` then
    /// hash-conses the whole rewritten conjunction, so re-asserting an
    /// already-seen shape costs a re-walk, not a duplicate variable.
    ///
    /// Bool-sorted `ite`s are left untouched; the gate encoder in
    /// `encode_depth_uncached` already gives them correct Tseitin semantics
    /// (`Bool` completion, via `Constraint::BoolApp`, is what lets EUF see
    /// *those*).
    pub(super) fn eliminate_nonbool_ite(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        let mut fresh_of: FxHashMap<TermId, TermId> = FxHashMap::default();
        // First pass: name a fresh variable for each eligible `ite` subterm,
        // in the walk's stable order.
        let mut ite_terms: Vec<TermId> = Vec::new();
        for st in collect_ground_subterms(term, manager) {
            let Some(t) = manager.get(st) else {
                continue;
            };
            if matches!(t.kind, TermKind::Ite(..)) && needs_ite_elimination(t.sort, manager) {
                let v = manager.mk_var(&format!("$encode-ite-elim!{}", st.0), t.sort);
                fresh_of.insert(st, v);
                ite_terms.push(st);
            }
        }
        if ite_terms.is_empty() {
            return term;
        }

        // Second pass: build the side-conditions. An `ite` nested inside
        // another `ite`'s branches has already been assigned its own fresh
        // variable above, so substituting `fresh_of` into the branches
        // before building the defining equation folds inner eliminations in
        // automatically -- no separate recursion needed.
        let mut side_conditions: Vec<TermId> = Vec::with_capacity(ite_terms.len() * 2);
        for ite_term in ite_terms {
            let Some(t) = manager.get(ite_term) else {
                continue;
            };
            let TermKind::Ite(cond, then_branch, else_branch) = &t.kind else {
                continue;
            };
            let (cond, then_branch, else_branch) = (*cond, *then_branch, *else_branch);
            let v = fresh_of[&ite_term];

            let cond_sub = manager.substitute(cond, &fresh_of);
            let then_sub = manager.substitute(then_branch, &fresh_of);
            let else_sub = manager.substitute(else_branch, &fresh_of);

            let v_eq_then = manager.mk_eq(v, then_sub);
            let v_eq_else = manager.mk_eq(v, else_sub);
            let not_cond = manager.mk_not(cond_sub);
            side_conditions.push(manager.mk_implies(cond_sub, v_eq_then));
            side_conditions.push(manager.mk_implies(not_cond, v_eq_else));
        }

        let rewritten = manager.substitute(term, &fresh_of);
        let mut conjuncts = Vec::with_capacity(1 + side_conditions.len());
        conjuncts.push(rewritten);
        conjuncts.extend(side_conditions);
        manager.mk_and(conjuncts)
    }

    /// Abstract every compound Bool subterm reachable from `term` (outside
    /// any binder) that is used as an uninterpreted-function argument,
    /// replacing each with a fresh Bool variable plus a defining equality.
    ///
    /// Congruence closure decides `f(x) = f(y)` by comparing `x`'s and `y`'s
    /// classes; the Bool-completion path (`Constraint::BoolApp`, wired from
    /// `TermKind::Var` and `TermKind::Apply` in `encode_depth_uncached`) is
    /// what merges a Bool-sorted class with the canonical true/false node by
    /// SAT-assigned value. A *compound* Bool argument such as `(and p q)`
    /// passed straight into `f` is neither a `Var` nor an `Apply`, so it
    /// never gets completed: two such arguments that the SAT assignment
    /// makes equal in truth value stay in separate EUF classes forever, and
    /// the congruence over `f` that should follow from that never fires.
    ///
    /// Replacing `(and p q)` with a fresh Bool variable `v` plus `(= v (and
    /// p q))` turns the argument into something Bool-completion *does*
    /// cover (`v` is now a plain `Var`), while the defining equality keeps
    /// `v` tied to the gate's real value through the ordinary Tseitin iff
    /// encoding. `Apply` arguments are left alone -- `Constraint::BoolApp`
    /// is already registered unconditionally for every Bool-sorted `Apply`
    /// term in `encode_depth_uncached`. A plain `Var` argument needs no
    /// rewriting, but *is* marked via [`Self::mark_bool_uf_arg`] so the
    /// `TermKind::Var` arm knows to complete it too: without that, `f(b1)`
    /// and `f(b2))` for two independent Bool variables that the SAT
    /// assignment happens to make equal in value would never be recognised
    /// as congruent, because plain Bool variables were not completed at all
    /// before this pass existed.
    pub(super) fn abstract_compound_bool_args(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        let bool_sort = manager.sorts.bool_sort;
        let mut compound_args: Vec<TermId> = Vec::new();
        let mut plain_var_args: Vec<TermId> = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        for st in collect_ground_subterms(term, manager) {
            let Some(t) = manager.get(st) else {
                continue;
            };
            let TermKind::Apply { args, .. } = &t.kind else {
                continue;
            };
            for &arg in args {
                let Some(arg_t) = manager.get(arg) else {
                    continue;
                };
                if arg_t.sort != bool_sort || !seen.insert(arg) {
                    continue;
                }
                match arg_t.kind {
                    TermKind::Var(_) => plain_var_args.push(arg),
                    TermKind::Apply { .. } => {} // already unconditionally completed
                    _ => compound_args.push(arg),
                }
            }
        }
        for v in plain_var_args {
            self.mark_bool_uf_arg(v);
        }
        if compound_args.is_empty() {
            return term;
        }

        let mut fresh_of: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut side_conditions: Vec<TermId> = Vec::with_capacity(compound_args.len());
        for arg in compound_args {
            let v = manager.mk_var(&format!("$encode-bool-arg!{}", arg.0), bool_sort);
            side_conditions.push(manager.mk_eq(v, arg));
            self.mark_bool_uf_arg(v);
            fresh_of.insert(arg, v);
        }

        let rewritten = manager.substitute(term, &fresh_of);
        let mut conjuncts = Vec::with_capacity(1 + side_conditions.len());
        conjuncts.push(rewritten);
        conjuncts.extend(side_conditions);
        manager.mk_and(conjuncts)
    }

    /// Add the forward half of a non-Bool `ite`'s conditional-equality
    /// semantics, for an `ite` appearing as a direct operand of the theory
    /// equality `eq_var` was just created for.
    ///
    /// `(= a (ite c t e))` (non-Bool sort) holds iff `(c -> a = t) & (~c ->
    /// a = e)`. Unlike [`Solver::eliminate_nonbool_ite`], this does not
    /// introduce a fresh variable or touch `manager`'s term graph -- it adds
    /// two clauses built from literals `encode_depth` already produces for
    /// existing terms (`cond`, and the branch equalities), gated on `eq_var`
    /// itself:
    ///
    /// ```text
    /// eq_var &  cond  ->  (a = t)
    /// eq_var & ~cond  ->  (a = e)
    /// ```
    ///
    /// Only the forward direction is added -- soundness needs the theory to
    /// *detect* every conflict `eq_var = true` implies, not to fully define
    /// `eq_var`'s own semantics (something else already does that: either
    /// `eliminate_nonbool_ite` if this equality came from `assert`, or
    /// simply EUF treating the `ite` as an opaque leaf, which stays sound —
    /// just incomplete — on its own). Because it reinforces an
    /// already-determined `eq_var` rather than defining a fresh variable,
    /// this is safe to run from inside the polarity-sensitive recursive
    /// encoder: unlike a defining biconditional, a one-directional
    /// implication gated on `eq_var`'s own value cannot be defeated by
    /// choosing `eq_var`'s negation.
    ///
    /// Each generated branch-equality atom (`a = t`, `a = e`) is itself
    /// encoded through `encode_depth`, so an `ite` nested inside `t`/`e` is
    /// handled by the recursion reaching this same method again.
    pub(super) fn encode_nonbool_ite_equality(
        &mut self,
        eq_var: Var,
        lhs: TermId,
        rhs: TermId,
        manager: &mut TermManager,
        depth: u32,
    ) {
        let eq_neg = Lit::neg(eq_var);
        for (a, b) in [(lhs, rhs), (rhs, lhs)] {
            let Some(b_term) = manager.get(b) else {
                continue;
            };
            let TermKind::Ite(cond, then_branch, else_branch) = &b_term.kind else {
                continue;
            };
            // Same scope as `eliminate_nonbool_ite`: Bool is the gate
            // encoder's business, and BitVec/Array/String/FloatingPoint each
            // have a theory-specific encoder that already recurses through
            // `ite` directly (see `needs_ite_elimination`'s doc for why a
            // BitVec `ite` in particular must not be touched here either).
            if !needs_ite_elimination(b_term.sort, manager) {
                continue;
            }
            let (cond, then_branch, else_branch) = (*cond, *then_branch, *else_branch);
            let cond_lit = self.encode_depth(cond, manager, depth + 1);
            let a_eq_then = manager.mk_eq(a, then_branch);
            let a_eq_else = manager.mk_eq(a, else_branch);
            let then_lit = self.encode_depth(a_eq_then, manager, depth + 1);
            let else_lit = self.encode_depth(a_eq_else, manager, depth + 1);
            self.sat.add_clause([eq_neg, cond_lit.negate(), then_lit]);
            self.sat.add_clause([eq_neg, cond_lit, else_lit]);
            // `lhs`/`rhs` are handled by one loop iteration each; an `ite` in
            // *both* operands is covered because both iterations run.
        }
    }
}

#[cfg(test)]
mod tests;
