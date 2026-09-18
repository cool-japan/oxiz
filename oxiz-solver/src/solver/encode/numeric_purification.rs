//! Numeric (Int/Real) uninterpreted-function-argument purification.
//!
//! Split out of `encode.rs` into this child module, following the same
//! `encode/` convention as `bool_euf_encoding.rs`/`track_theory_vars.rs`.
//!
//! # The gap this closes
//!
//! `Solver::track_theory_vars` deliberately does
//! **not** descend into an uninterpreted-function application's arguments —
//! see its own doc comment ("argument terms are arithmetic values passed to
//! an opaque symbol, not arithmetic variables in their own right within this
//! constraint"). A bare `Var` argument still gets an arithmetic variable
//! whenever it *also* appears in some other arithmetic atom, but a constant
//! or compound argument such as the `3` in `f(3)` or the `fmt1 + 1` in
//! `f(fmt1 + 1)` never does: it is folded straight into whatever linear
//! constraint mentions it and never becomes a shared interface term of its
//! own.
//!
//! That is fatal to Nelson-Oppen combination for exactly the case that
//! matters most: if arithmetic entails `y = 3` (say from `y = x + 1` and `x =
//! 2`), the equality has nothing to attach to on the EUF side — `3` was never
//! interned as a term `f`'s argument could be compared against — so the
//! congruence `f(y) = f(3)` that entailment should trigger is silently
//! missed. Left unpurified, `f(y) != f(3)` conjoined with the arithmetic
//! facts above answers a spurious `sat`.
//!
//! # The fix
//!
//! Mirroring `Solver::abstract_compound_bool_args`'s
//! treatment of compound Bool UF arguments, every numeric UF-application
//! argument that is not already one of the shapes `track_theory_vars` tracks
//! on its own (`Var`, `Apply`, `Select`, `Ite`, `Div`, `Mod`, `DtSelector`) is
//! hoisted out to a fresh, globally-scoped proxy variable plus a defining
//! equality. The proxy is a plain `Var`, so it *is* one of the shapes
//! `track_theory_vars` tracks — it becomes a genuine arithmetic interface
//! term, and (being a UF argument itself) also flows into
//! `TheoryManager::care_graph_candidates`, so the entailed equality reaches
//! EUF the same way an ordinary shared variable would.
//!
//! Reference: this is the standard Nelson-Oppen *purification* step (see
//! e.g. cvc5's/Z3's theory-combination front ends), specialised here to only
//! the arguments that actually need it — purifying every numeric term in the
//! formula would need none of this care and would only inflate the SAT
//! problem for no soundness gain.

use rustc_hash::{FxHashMap, FxHashSet};

use super::bool_euf_encoding::collect_ground_subterms;
use super::*;

impl Solver {
    /// Purify numeric (Int/Real) uninterpreted-function-application arguments
    /// reachable from `term` (outside any binder) into fresh shared
    /// variables, returning an equisatisfiable term with the purifications'
    /// defining equalities conjoined.
    ///
    /// Every numeric argument found — proxied or already `track_theory_vars`
    /// -shared — is registered via [`Self::mark_numeric_uf_arg`], which is
    /// what lets the non-convex integer case-split refinement find the
    /// terms worth an explicit finite-domain case split.
    ///
    /// Scoped away from any UF symbol that occurs as the head of an `Apply`
    /// inside a *registered quantifier's body*
    /// (`self.quantifier_uf_funcs`, populated by
    /// `Solver::collect_quantifier_uf_funcs` as each quantifier is
    /// registered): replacing a ground literal UF argument like the `1` in
    /// `f(1)` with a fresh proxy variable plus a defining equality is
    /// exactly the kind of "new Boolean/EUF structure" that
    /// `axiomatize_arith_constant_equalities`-style combination passes in
    /// other engines keep away from MBQI, because MBQI's e-matching resolves
    /// a trigger like `f(i)` against `f(1)` by direct syntactic unification
    /// but needs an EUF-class lookup for `f(v)` -- one round later, since `v
    /// = 1` is a *derived* fact, not part of `v`'s own syntax. That extra
    /// round shifts *when* an instantiation is produced, which
    /// `scope_rebase_tests::a_no_op_push_pop_between_checks_does_not_re_encode_the_goal`
    /// caught as a one-time step in the original-clause count after the
    /// first `check` on a quantified goal containing a ground UF literal
    /// argument of the *same* function a quantifier's trigger applies.
    ///
    /// # What this gate actually costs (it is not free)
    ///
    /// This doc used to claim the skip was "still sound (the goal is
    /// unchanged)". That is **wrong**, and the correction matters. The goal is
    /// indeed unchanged, but purification is not a rewrite for its own sake --
    /// it is what makes a ground UF argument visible to arithmetic/EUF
    /// equality sharing in the first place. Skipping it for a trigger function
    /// leaves exactly the false-`sat` this module exists to close, on any
    /// ground disequality over that same function:
    ///
    /// ```smtlib
    /// (assert (forall ((z Int)) (>= (f z) 0)))
    /// (assert (= x 2)) (assert (= y (+ x 1)))
    /// (assert (not (= (f y) (f 3))))   ; ground part alone is unsat
    /// ```
    ///
    /// `f(3)`'s argument is not purified, `3` never becomes an arithmetic
    /// interface term, the entailed `y = 3` has nothing on the EUF side to
    /// attach to, and `f(y) = f(3)` is never derived.
    ///
    /// So the real trade-off is: **soundness of the ground fragment is not
    /// sacrificed, but completeness is, and only because a separate gate
    /// catches the difference.** The quantified `Sat` exits in `check_core`
    /// run `Solver::quantified_model_refutes_ground_assertions`, which refuses
    /// any candidate model that falsifies a ground assertion or is not a
    /// function over the ground applications -- so the shape above answers
    /// `unknown` rather than a wrong `sat` (see
    /// `pr30_soundness::test_pr30_quantifier_trigger_function_ground_diseq_is_not_sat`).
    /// Purifying under binders too would recover the full `unsat`; it is not
    /// done here because of the e-matching perturbation described above.
    ///
    /// The gate used to be the coarser `self.has_quantifiers` (true once
    /// *any* quantifier had been registered, regardless of which function it
    /// applies), which made the soundness fix itself order-dependent:
    /// `(assert (forall ((z Int)) (> (g z) 0))) (assert (not (= (f y) (f
    /// 3))))` set `has_quantifiers` from `g`'s quantifier before `f`'s
    /// ground disequality was ever purified, reintroducing the exact
    /// false-`sat` this module exists to close even though `f` never occurs
    /// under a binder. Scoping per function symbol keeps every function that
    /// is not itself a quantifier trigger purified in any assertion order --
    /// the soundness fix this module exists for targets QF_UFLIA / QF_UFIDL
    /// specifically (see the module docs), where no function is ever a
    /// quantifier trigger, so this gate costs nothing there.
    pub(super) fn purify_numeric_uf_args(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        let int_sort = manager.sorts.int_sort;
        let real_sort = manager.sorts.real_sort;
        let ground = collect_ground_subterms(term, manager);

        // A numeric *constant* that also occurs as a `Mul` coefficient
        // elsewhere in `term` must not be proxied: `substitute` below rewrites
        // every occurrence of a given `TermId` (the interner hash-conses
        // literals, so the "same" `3` used as `f`'s argument and as `(* 3 x)`'s
        // coefficient is literally one `TermId`), and rewriting a coefficient
        // to a fresh variable would manufacture nonlinearity the arithmetic
        // parser then honestly — but unhelpfully — rejects as `Unknown`.
        let mut mul_coefficients: FxHashSet<TermId> = FxHashSet::default();
        for &st in &ground {
            let Some(t) = manager.get(st) else { continue };
            let TermKind::Mul(args) = &t.kind else {
                continue;
            };
            for &a in args {
                if manager.get(a).is_some_and(|at| {
                    matches!(at.kind, TermKind::IntConst(_) | TermKind::RealConst(_))
                }) {
                    mul_coefficients.insert(a);
                }
            }
        }

        // Arguments of a quantifier-trigger function are left exactly as
        // `track_theory_vars` sees them today (see the module doc and this
        // function's doc comment for why): a first pass over every ground
        // `Apply` decides *per argument*, not per assertion, since the same
        // `term` can mention both a plain function and one that also occurs
        // under a binder elsewhere in the script.
        let mut forbidden_args: FxHashSet<TermId> = FxHashSet::default();
        for &st in &ground {
            let Some(t) = manager.get(st) else { continue };
            let TermKind::Apply { func, args } = &t.kind else {
                continue;
            };
            if self.quantifier_uf_funcs.contains(func) {
                forbidden_args.extend(args.iter().copied());
            }
        }

        let mut to_proxy: Vec<TermId> = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        for &st in &ground {
            let Some(t) = manager.get(st) else { continue };
            let TermKind::Apply { func, args } = &t.kind else {
                continue;
            };
            // The SMT-LIB array constant is an `Apply` only because it has no
            // term kind of its own; it is an *interpreted* symbol whose single
            // argument is the array's default value, not an uninterpreted
            // function whose numeric arguments need an interface variable.
            // Proxying that argument replaced the default with a fresh
            // variable, and both the array-constant read axiom and the model
            // evaluator recognise an array constant by its default being a
            // value — so `(= 2 (select ((as const (Array Int Int)) 1)
            // (+ 0 1)))` lost its axiom and `(not (distinct 2 (select
            // ((as const (Array Int Int)) 1) (+ 0 1))))` answered `sat`
            // (`#P2b-37`).
            if manager.resolve_str(*func) == crate::solver::array_axioms::CONST_ARRAY_FUNC {
                continue;
            }
            for &arg in args {
                let Some(arg_t) = manager.get(arg) else {
                    continue;
                };
                let numeric = arg_t.sort == int_sort || arg_t.sort == real_sort;
                if !numeric || !seen.insert(arg) || forbidden_args.contains(&arg) {
                    continue;
                }
                match &arg_t.kind {
                    TermKind::Var(_)
                    | TermKind::Apply { .. }
                    | TermKind::Select(_, _)
                    | TermKind::Ite(_, _, _)
                    | TermKind::Div(_, _)
                    | TermKind::Mod(_, _)
                    | TermKind::DtSelector { .. } => {
                        // Already (or about to be) an arithmetic interface
                        // term of its own via `track_theory_vars` — no
                        // rewrite needed, but it is still an int-case-split
                        // candidate.
                        self.mark_numeric_uf_arg(arg);
                    }
                    TermKind::IntConst(_) | TermKind::RealConst(_)
                        if mul_coefficients.contains(&arg) =>
                    {
                        // Unsafe to hoist (see above); a bare literal has no
                        // bounds to split on either, so it is not registered.
                    }
                    _ => to_proxy.push(arg),
                }
            }
        }
        if to_proxy.is_empty() {
            return term;
        }

        let mut proxy_of: FxHashMap<TermId, TermId> = FxHashMap::default();
        for arg in to_proxy {
            if proxy_of.contains_key(&arg) {
                continue;
            }
            let Some(arg_t) = manager.get(arg) else {
                continue;
            };
            let v = manager.mk_var(
                &oxiz_core::smtlib::reserved_name("numarg", &arg.0.to_string()),
                arg_t.sort,
            );
            self.mark_numeric_uf_arg(v);
            proxy_of.insert(arg, v);
        }

        // Alias every ground *application* subterm whose shape actually
        // changes under the substitution back to its purified counterpart.
        // `self.assertions` (and hence any later `(get-value ((f 3)))`
        // query) still names the pre-purification shape `f(3)`, but only
        // `f(v)` is ever interned into EUF/arithmetic — without this alias
        // `build_model` has no way to give the original term the value its
        // purified twin was assigned, and a genuinely satisfiable model
        // would print `(f 3)` back unevaluated instead of its value. Other
        // term shapes (`Eq`, `Lt`, ...) do not need this: they are already
        // recomputed structurally by the model evaluator, only an opaque
        // `Apply` is looked up by identity.
        for &st in &ground {
            let Some(t) = manager.get(st) else { continue };
            if !matches!(t.kind, TermKind::Apply { .. }) {
                continue;
            }
            let purified_st = manager.substitute(st, &proxy_of);
            if purified_st != st {
                self.mark_numeric_purify_alias(st, purified_st);
            }
        }

        let mut side_conditions: Vec<TermId> = Vec::with_capacity(proxy_of.len());
        for (&arg, &v) in &proxy_of {
            side_conditions.push(manager.mk_eq(v, arg));
        }
        let rewritten = manager.substitute(term, &proxy_of);
        let mut parts: Vec<TermId> = Vec::with_capacity(1 + side_conditions.len());
        parts.push(rewritten);
        parts.extend(side_conditions);
        manager.mk_and(parts)
    }

    /// Record `term` as a numeric argument of some uninterpreted-function
    /// application — either a fresh purification proxy or a term
    /// `track_theory_vars` already shares on its own (a bare `Var`, most
    /// commonly). Idempotent and trailed, mirroring `Solver::mark_bool_uf_arg`'s pattern.
    pub(super) fn mark_numeric_uf_arg(&mut self, term: TermId) {
        if self.numeric_uf_arg_terms.insert(term) {
            self.trail.push(TrailOp::NumericUfArgAdded { term });
        }
    }

    /// Record that the pre-purification term `original` (an uninterpreted-
    /// function application, still what `self.assertions` and any external
    /// `get-value` query name) denotes the same value as `purified` (what
    /// was actually interned into EUF/arithmetic). `Solver::build_model`
    /// consults this to give `original` a model value it could otherwise
    /// never resolve. Idempotent and trailed.
    fn mark_numeric_purify_alias(&mut self, original: TermId, purified: TermId) {
        if let crate::prelude::hash_map::Entry::Vacant(slot) =
            self.numeric_purify_aliases.entry(original)
        {
            slot.insert(purified);
            self.trail
                .push(TrailOp::NumericPurifyAliasAdded { term: original });
        }
    }
}
