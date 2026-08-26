//! Z3 API Compatibility Layer — Extension 3
//!
//! This module adds three further Z3-compatible surfaces on top of the core
//! types in [`crate::z3_compat`] and the earlier extension layers
//! ([`crate::z3_compat::ext`], [`crate::z3_compat::ext2`]):
//!
//! - **Sort introspection** — [`Z3Sort`] / [`Z3SortKind`].  Mirrors Z3's
//!   `Sort::kind()`, `bv_size()`, `array_domain()`, `array_range()` and
//!   `name()`, plus helpers on [`Z3Context`] to recover the sort of a term
//!   ([`Z3Context::sort_of_bool`], [`Z3Context::sort_of_int`],
//!   [`Z3Context::sort_of_real`], [`Z3Context::sort_of_bv`],
//!   [`Z3Context::sort_of_term`]).
//! - **Term substitution** — [`Z3Context::substitute`], which replaces
//!   subterms throughout an expression.  A thin adapter over
//!   [`TermManager::substitute`]: it converts Z3's `&[(from, to)]` pair slice
//!   into the map that routine takes, and inherits its iterative,
//!   capture-avoiding, exhaustive-by-construction behaviour.
//! - **Quantifier patterns / triggers** — [`Z3Pattern`],
//!   [`Z3Context::mk_pattern`], [`Z3Context::forall_with_patterns`] and
//!   [`Z3Context::exists_with_patterns`].  Backed by
//!   [`TermManager::mk_forall_with_patterns`] /
//!   [`TermManager::mk_exists_with_patterns`].
//!
//! [`TermManager::mk_forall_with_patterns`]: oxiz_core::ast::TermManager::mk_forall_with_patterns
//! [`TermManager::mk_exists_with_patterns`]: oxiz_core::ast::TermManager::mk_exists_with_patterns
//! [`TermManager::substitute`]: oxiz_core::ast::TermManager::substitute

use std::rc::Rc;

use rustc_hash::FxHashMap;

use oxiz_core::ast::{TermId, TermManager};
use oxiz_core::sort::{SortId, SortKind};

use crate::z3_compat::{BV, Bool, Int, Real, Z3Context};

// ─── Z3SortKind ───────────────────────────────────────────────────────────────

/// The high-level kind of a [`Z3Sort`], mirroring `z3::SortKind`.
///
/// This collapses OxiZ's richer [`SortKind`] into the
/// categories that Z3 exposes through its public API.  Sorts that have no Z3
/// analogue (sort parameters, parametric applications) are reported as
/// [`Z3SortKind::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Z3SortKind {
    /// The boolean sort.
    Bool,
    /// The integer sort.
    Int,
    /// The real sort.
    Real,
    /// A bit-vector sort of some fixed width.
    BitVec,
    /// An array sort with a domain and a range.
    Array,
    /// An algebraic datatype sort.
    Datatype,
    /// An uninterpreted sort.
    Uninterpreted,
    /// Any sort with no direct Z3 analogue (string, floating-point,
    /// rounding-mode, sort parameter, parametric application).
    Other,
}

// ─── Z3Sort ───────────────────────────────────────────────────────────────────

/// Analogue of `z3::Sort`.
///
/// A lightweight handle pairing a [`SortId`] with the owning context's
/// [`TermManager`], so that the sort can be introspected (kind, bit-width,
/// array domain/range, name) after the fact.
#[derive(Clone)]
pub struct Z3Sort {
    /// The underlying sort identifier.
    pub id: SortId,
    /// Back-reference to the owning context's term manager.
    ctx: Rc<core::cell::RefCell<TermManager>>,
}

impl Z3Sort {
    /// Wrap a raw [`SortId`] together with the context it belongs to.
    #[must_use]
    pub fn new(ctx: &Z3Context, id: SortId) -> Self {
        Self {
            id,
            ctx: ctx.tm_handle(),
        }
    }

    /// Internal constructor from a raw term-manager handle.
    fn from_handle(ctx: Rc<core::cell::RefCell<TermManager>>, id: SortId) -> Self {
        Self { id, ctx }
    }

    /// Return the high-level [`Z3SortKind`] of this sort.
    #[must_use]
    pub fn kind(&self) -> Z3SortKind {
        let tm = self.ctx.borrow();
        match tm.sorts.get(self.id).map(|s| &s.kind) {
            Some(SortKind::Bool) => Z3SortKind::Bool,
            Some(SortKind::Int) => Z3SortKind::Int,
            Some(SortKind::Real) => Z3SortKind::Real,
            Some(SortKind::BitVec(_)) => Z3SortKind::BitVec,
            Some(SortKind::Array { .. }) => Z3SortKind::Array,
            Some(SortKind::Datatype(_)) => Z3SortKind::Datatype,
            Some(SortKind::Uninterpreted(_)) => Z3SortKind::Uninterpreted,
            // `RoundingMode` joins the `Other` bucket alongside the
            // floating-point sorts it belongs to, which this shim already maps
            // there.  It must *not* be folded into `Uninterpreted`: it is a
            // reserved built-in with five named elements, and a client that
            // saw `Uninterpreted` would treat those elements as an
            // unconstrained free domain.
            Some(
                SortKind::String
                | SortKind::FloatingPoint { .. }
                | SortKind::RoundingMode
                | SortKind::Parameter(_)
                | SortKind::Parametric { .. },
            )
            | None => Z3SortKind::Other,
        }
    }

    /// If this is a bit-vector sort, return its width in bits.
    ///
    /// Returns `None` for every other sort kind.
    #[must_use]
    pub fn bv_size(&self) -> Option<u32> {
        let tm = self.ctx.borrow();
        match tm.sorts.get(self.id).map(|s| &s.kind) {
            Some(&SortKind::BitVec(width)) => Some(width),
            _ => None,
        }
    }

    /// If this is an array sort, return its domain (index) sort.
    ///
    /// Returns `None` for every other sort kind.
    #[must_use]
    pub fn array_domain(&self) -> Option<Z3Sort> {
        let domain = {
            let tm = self.ctx.borrow();
            match tm.sorts.get(self.id).map(|s| &s.kind) {
                Some(&SortKind::Array { domain, .. }) => domain,
                _ => return None,
            }
        };
        Some(Z3Sort::from_handle(self.ctx.clone(), domain))
    }

    /// If this is an array sort, return its range (element) sort.
    ///
    /// Returns `None` for every other sort kind.
    #[must_use]
    pub fn array_range(&self) -> Option<Z3Sort> {
        let range = {
            let tm = self.ctx.borrow();
            match tm.sorts.get(self.id).map(|s| &s.kind) {
                Some(&SortKind::Array { range, .. }) => range,
                _ => return None,
            }
        };
        Some(Z3Sort::from_handle(self.ctx.clone(), range))
    }

    /// Return a human-readable name for this sort.
    ///
    /// Mirrors Z3's `Sort::to_string`, e.g. `"Bool"`, `"Int"`, `"Real"`,
    /// `"BitVec(32)"`, `"Array"`, or the declared name of an uninterpreted /
    /// datatype sort.
    #[must_use]
    pub fn name(&self) -> String {
        let tm = self.ctx.borrow();
        tm.sorts
            .sort_name(self.id)
            .unwrap_or_else(|| "Unknown".to_string())
    }
}

impl core::fmt::Debug for Z3Sort {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Z3Sort")
            .field("id", &self.id)
            .field("kind", &self.kind())
            .finish()
    }
}

impl core::fmt::Display for Z3Sort {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.name())
    }
}

// ─── Z3Context: sort accessor helpers ────────────────────────────────────────

impl Z3Context {
    /// Internal: clone the shared term-manager handle.
    ///
    /// Kept private to this module so the `tm` field need not become public.
    fn tm_handle(&self) -> Rc<core::cell::RefCell<TermManager>> {
        self.tm.clone()
    }

    /// Return the [`Z3Sort`] of an arbitrary term identifier.
    ///
    /// Looks up the term in the manager and reads its sort.  If the term is not
    /// present in this context's manager, the boolean sort is returned as a
    /// conservative default.
    #[must_use]
    pub fn sort_of_term(&self, term: TermId) -> Z3Sort {
        let sort_id = {
            let tm = self.tm.borrow();
            tm.get(term).map_or(tm.sorts.bool_sort, |t| t.sort)
        };
        Z3Sort::from_handle(self.tm.clone(), sort_id)
    }

    /// Return the [`Z3Sort`] of a boolean term.
    #[must_use]
    pub fn sort_of_bool(&self, b: &Bool) -> Z3Sort {
        self.sort_of_term(b.id)
    }

    /// Return the [`Z3Sort`] of an integer term.
    #[must_use]
    pub fn sort_of_int(&self, x: &Int) -> Z3Sort {
        self.sort_of_term(x.id)
    }

    /// Return the [`Z3Sort`] of a real term.
    #[must_use]
    pub fn sort_of_real(&self, x: &Real) -> Z3Sort {
        self.sort_of_term(x.id)
    }

    /// Return the [`Z3Sort`] of a bit-vector term.
    #[must_use]
    pub fn sort_of_bv(&self, b: &BV) -> Z3Sort {
        self.sort_of_term(b.id)
    }

    /// Return the [`Z3Sort`] wrapping a known [`SortId`] in this context.
    #[must_use]
    pub fn wrap_sort(&self, id: SortId) -> Z3Sort {
        Z3Sort::from_handle(self.tm.clone(), id)
    }
}

// ─── Term substitution ────────────────────────────────────────────────────────

impl Z3Context {
    /// Substitute subterms within `expr`.
    ///
    /// Each `(from, to)` pair replaces every occurrence of the subterm `from`
    /// with `to`.  Substitution is memoized so the cost is linear in the size
    /// of the term DAG even when subterms are shared.
    ///
    /// # Delegates to the core substitution
    ///
    /// This is a thin adapter over
    /// [`TermManager::substitute`](oxiz_core::ast::TermManager::substitute) --
    /// it only turns Z3's `&[(from, to)]` pair slice into the `FxHashMap` that
    /// routine takes -- so it inherits all of that routine's properties:
    ///
    /// * it descends into `Forall`/`Exists`/`Let`/`Match` bodies, bindings,
    ///   cases *and* trigger patterns, so a genuinely free (unshadowed)
    ///   variable occurring under any binder is replaced;
    /// * it is capture-avoiding: a bound variable whose name would capture a
    ///   free variable of some replacement term is alpha-renamed first
    ///   (OxiZ has no separate bound-variable representation -- a
    ///   quantifier's bound occurrences are ordinary `TermKind::Var(name)`
    ///   terms, hash-consed identically to free ones -- so this renaming is
    ///   what makes substitution under a binder correct at all);
    /// * every `TermKind` variant is handled explicitly there, with no
    ///   catch-all arm, so a newly added variant is a compile error rather
    ///   than a silently skipped node;
    /// * it uses an explicit heap stack, so no input depth overflows the
    ///   native call stack.
    ///
    /// This module used to carry its own bottom-up rebuild (`subst_rebuild`
    /// and friends) instead, justified by the core routine "not recursing
    /// through bit-vector operators or function applications". That
    /// justification was stale: `TermManager::rebuild_substituted` covers
    /// every bit-vector kind and `Apply` explicitly. The duplicate had two
    /// defects the delegation removes -- it treated all four binder forms as
    /// opaque (so `(forall ((x Int)) (Q x z))[z := t]` came back completely
    /// untouched, a silent under-substitution), and it was a second
    /// independent implementation of "rebuild a term given a subterm
    /// replacement map", exactly the duplication that
    /// `oxiz_core::ast::traversal::map_terms` retired its own
    /// `transform_children` over.
    #[must_use]
    pub fn substitute(&self, expr: TermId, subst: &[(TermId, TermId)]) -> TermId {
        if subst.is_empty() {
            return expr;
        }
        let map: FxHashMap<TermId, TermId> = subst.iter().copied().collect();
        self.tm.borrow_mut().substitute(expr, &map)
    }
}

// ─── Quantifier patterns / triggers ───────────────────────────────────────────

/// Analogue of `z3::Pattern`.
///
/// A pattern (a.k.a. *trigger*) is a list of terms that guides e-matching
/// instantiation of a quantifier.  In OxiZ a pattern is materialised as the
/// list of trigger terms it carries; construct one with
/// [`Z3Context::mk_pattern`] and attach it to a quantifier with
/// [`Z3Context::forall_with_patterns`] / [`Z3Context::exists_with_patterns`].
#[derive(Debug, Clone)]
pub struct Z3Pattern {
    /// The trigger terms making up this pattern.
    pub terms: Vec<TermId>,
}

impl Z3Pattern {
    /// Number of trigger terms in this pattern.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Returns `true` if the pattern carries no trigger terms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl Z3Context {
    /// Build a multi-pattern (trigger) from a slice of terms.
    ///
    /// Mirrors Z3's `mk_pattern`.  The terms are stored verbatim; they are only
    /// interpreted when the pattern is attached to a quantifier via
    /// [`Z3Context::forall_with_patterns`] or
    /// [`Z3Context::exists_with_patterns`].
    #[must_use]
    pub fn mk_pattern(&self, terms: &[TermId]) -> Z3Pattern {
        Z3Pattern {
            terms: terms.to_vec(),
        }
    }

    /// Build a universal quantifier with explicit instantiation patterns.
    ///
    /// `bound` names the quantified variables as `(name, sort)` pairs (matching
    /// the convention of [`forall_bool`](crate::z3_compat::ext::forall_bool)).
    /// Each [`Z3Pattern`] becomes one trigger guiding e-matching; the trigger
    /// terms should reference the bound variables by the same names.
    ///
    /// Delegates to
    /// [`TermManager::mk_forall_with_patterns`](oxiz_core::ast::TermManager::mk_forall_with_patterns).
    #[must_use]
    pub fn forall_with_patterns(
        &self,
        bound: &[(&str, SortId)],
        patterns: &[Z3Pattern],
        body: &Bool,
    ) -> Bool {
        let vars: Vec<(&str, SortId)> = bound.to_vec();
        let pats: Vec<Vec<TermId>> = patterns.iter().map(|p| p.terms.clone()).collect();
        let id = self
            .tm
            .borrow_mut()
            .mk_forall_with_patterns(vars, body.id, pats);
        Bool::from_id(id)
    }

    /// Build an existential quantifier with explicit instantiation patterns.
    ///
    /// Counterpart to [`Z3Context::forall_with_patterns`]; delegates to
    /// [`TermManager::mk_exists_with_patterns`](oxiz_core::ast::TermManager::mk_exists_with_patterns).
    #[must_use]
    pub fn exists_with_patterns(
        &self,
        bound: &[(&str, SortId)],
        patterns: &[Z3Pattern],
        body: &Bool,
    ) -> Bool {
        let vars: Vec<(&str, SortId)> = bound.to_vec();
        let pats: Vec<Vec<TermId>> = patterns.iter().map(|p| p.terms.clone()).collect();
        let id = self
            .tm
            .borrow_mut()
            .mk_exists_with_patterns(vars, body.id, pats);
        Bool::from_id(id)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::z3_compat::Z3Config;
    use oxiz_core::ast::TermKind;

    fn ctx() -> Z3Context {
        Z3Context::new(&Z3Config::new())
    }

    #[test]
    fn unit_sort_kinds() {
        let c = ctx();
        assert_eq!(c.wrap_sort(c.bool_sort()).kind(), Z3SortKind::Bool);
        assert_eq!(c.wrap_sort(c.int_sort()).kind(), Z3SortKind::Int);
        assert_eq!(c.wrap_sort(c.real_sort()).kind(), Z3SortKind::Real);
        assert_eq!(c.wrap_sort(c.bv_sort(8)).kind(), Z3SortKind::BitVec);
    }

    #[test]
    fn unit_bv_size_and_array() {
        let c = ctx();
        assert_eq!(c.wrap_sort(c.bv_sort(16)).bv_size(), Some(16));
        assert_eq!(c.wrap_sort(c.bool_sort()).bv_size(), None);

        let arr = c.array_sort(c.int_sort(), c.bool_sort());
        let s = c.wrap_sort(arr);
        assert_eq!(s.kind(), Z3SortKind::Array);
        assert_eq!(s.array_domain().map(|d| d.kind()), Some(Z3SortKind::Int));
        assert_eq!(s.array_range().map(|r| r.kind()), Some(Z3SortKind::Bool));
    }

    #[test]
    fn unit_substitute_identity() {
        let c = ctx();
        let x = Int::new_const(&c, "x");
        let y = Int::new_const(&c, "y");
        let sum = Int::add(&c, &[x.clone(), y.clone()]);
        // No matching pair leaves the term untouched.
        assert_eq!(c.substitute(sum.id, &[]), sum.id);
    }

    #[test]
    fn unit_pattern_basic() {
        let c = ctx();
        let p = c.mk_pattern(&[]);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    /// Run `f` to completion on a dedicated thread with a 128 KiB stack --
    /// deliberately far smaller than the default (several-MiB) main-thread
    /// stack -- and return whatever it returns. Mirrors the same helper in
    /// `oxiz-core`'s `ast/manager/query/tests.rs`: a stack overflow aborts
    /// the whole process rather than failing a single test gracefully, so
    /// for the deep-nesting test below, the call *returning at all* is
    /// itself part of what is being asserted.
    ///
    /// This stack and the depth below were scaled down together by a factor
    /// of 8 (from 1 MiB / 100 000).  The pin is the ~10 bytes of stack per
    /// nesting level, not the absolute depth, and the smaller pair keeps the
    /// interned terms out of swap.  Never raise one without the other.
    fn run_on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(f)
            .expect("spawning the constrained-stack test thread should succeed")
            .join()
            .expect("the constrained-stack thread must not panic")
    }

    #[test]
    fn substitute_survives_deep_not_chain_on_tiny_stack() {
        // Regression: this module's own (since retired) `subst_rebuild`
        // used to recurse natively once per level of term nesting, with no
        // depth guard at all; `TermManager::substitute`, which
        // `Z3Context::substitute` now delegates to, uses an explicit heap
        // stack. Built
        // iteratively (never recursively, which would overflow before the
        // assertion runs) and run inside a thread with a deliberately
        // small 128 KiB stack: the call returning at all is part of the
        // assertion, but the result must also be exactly correct -- the
        // substitution must reach all the way through the chain down to
        // the leaf, replacing it, not silently stop partway.
        const DEPTH: usize = 12_500;

        let (reached_leaf, old_leaf_gone) = run_on_small_stack(|| {
            let c = ctx();
            let bool_sort = c.bool_sort();
            let x = Bool::new_const(&c, "x");
            let y = Bool::new_const(&c, "y");

            // Built via repeated `mk_apply` of the same 1-ary uninterpreted
            // function `f` (`f(f(f(...f(x)...)))`) rather than `mk_not`:
            // `TermManager::intern` is `pub(crate)` in oxiz-core and not
            // reachable from this crate, and `mk_not` would collapse a
            // chain of an even number of negations back down to `x`
            // itself. Uninterpreted function application has no such
            // simplification, so this reaches genuine depth through only
            // the public API.
            let mut chain = x.id;
            {
                let mut tm = c.tm.borrow_mut();
                for _ in 0..DEPTH {
                    chain = tm.mk_apply("f", [chain], bool_sort);
                }
            }

            let result = c.substitute(chain, &[(x.id, y.id)]);

            // Peel the same number of `f(...)` layers back off the result
            // and confirm the leaf underneath is now `y`, not `x`.
            let mut current = result;
            for _ in 0..DEPTH {
                let kind = c.tm.borrow().get(current).map(|t| t.kind.clone());
                match kind {
                    Some(TermKind::Apply { args, .. }) if args.len() == 1 => current = args[0],
                    _ => break,
                }
            }
            (current == y.id, current != x.id)
        });

        assert!(
            reached_leaf,
            "substitute must reach the bottom of a 12,500-deep application chain"
        );
        assert!(
            old_leaf_gone,
            "the old leaf x must not remain after substitution"
        );
    }

    /// Regression: `Z3Context::substitute` used to treat `Forall`/`Exists`/
    /// `Let`/`Match` as opaque, so a genuinely free (unshadowed) variable
    /// occurring under any binder was silently left in place -- an
    /// under-substitution the caller has no way to detect.
    #[test]
    fn substitute_descends_into_forall_body() {
        let c = ctx();
        let int_sort = c.int_sort();
        let bool_sort = c.bool_sort();
        let x = Int::new_const(&c, "x");
        let z = Int::new_const(&c, "z");
        let w = Int::new_const(&c, "w");

        // (forall ((x Int)) (Q x z)) with z free.
        let body = c.tm.borrow_mut().mk_apply("Q", [x.id, z.id], bool_sort);
        let forall = c.forall_with_patterns(&[("x", int_sort)], &[], &Bool::from_id(body));

        let result = c.substitute(forall.id, &[(z.id, w.id)]);

        let expected_body = c.tm.borrow_mut().mk_apply("Q", [x.id, w.id], bool_sort);
        let expected =
            c.forall_with_patterns(&[("x", int_sort)], &[], &Bool::from_id(expected_body));
        assert_eq!(
            result, expected.id,
            "the free z under the forall must be replaced by w"
        );
    }

    /// Companion to [`substitute_descends_into_forall_body`]: now that the
    /// walk descends into binders, it must also be capture-avoiding --
    /// substituting a replacement whose free variable collides with a bound
    /// name has to alpha-rename that binder rather than capture it.
    #[test]
    fn substitute_under_binder_is_capture_avoiding() {
        let c = ctx();
        let int_sort = c.int_sort();
        let bool_sort = c.bool_sort();
        let x = Int::new_const(&c, "x");
        let z = Int::new_const(&c, "z");

        // (forall ((x Int)) (Q x z))[z := x]: naive substitution would
        // produce (forall ((x Int)) (Q x x)), capturing the substituted x.
        let body = c.tm.borrow_mut().mk_apply("Q", [x.id, z.id], bool_sort);
        let forall = c.forall_with_patterns(&[("x", int_sort)], &[], &Bool::from_id(body));

        let result = c.substitute(forall.id, &[(z.id, x.id)]);

        let kind =
            c.tm.borrow()
                .get(result)
                .map(|t| t.kind.clone())
                .expect("result term must exist");
        let TermKind::Forall { vars, body, .. } = kind else {
            panic!("expected the result to still be a Forall, got {kind:?}");
        };
        let renamed = vars
            .first()
            .map(|&(name, sort)| (c.tm.borrow().resolve_str(name).to_string(), sort))
            .expect("the rebuilt Forall must bind exactly one variable");
        assert_ne!(
            renamed.0, "x",
            "the bound x must have been alpha-renamed away from the substituted free x"
        );

        let fresh = c.tm.borrow_mut().mk_var(&renamed.0, renamed.1);
        let expected_body = c.tm.borrow_mut().mk_apply("Q", [fresh, x.id], bool_sort);
        assert_eq!(
            body, expected_body,
            "the bound occurrence must be renamed and the substituted x left free"
        );
    }
}
