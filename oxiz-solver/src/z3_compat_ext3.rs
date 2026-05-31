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
//! - **Term substitution** — [`Z3Context::substitute`], a capture-avoiding
//!   bottom-up rebuild that replaces ground subterms.  Implemented directly
//!   here (rather than delegating to `TermManager::substitute`) because the
//!   core routine does not recurse through bit-vector operators or function
//!   applications, both of which are reachable through the Z3 compat surface.
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

use oxiz_core::ast::{TermId, TermKind, TermManager};
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
    /// Any sort with no direct Z3 analogue (string, floating-point, sort
    /// parameter, parametric application).
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
            Some(
                SortKind::String
                | SortKind::FloatingPoint { .. }
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
    /// with `to`.  Substitution is performed bottom-up with memoization so the
    /// cost is linear in the size of the term DAG even when subterms are
    /// shared.
    ///
    /// This is capture-avoiding for **ground** replacement (the standard Z3
    /// `substitute` use case): the `from`/`to` terms are treated as opaque, and
    /// because OxiZ quantifiers carry their bound variables as `(name, sort)`
    /// pairs (not as free `Var` terms inside the body that could clash with a
    /// replacement), rebuilding through the manager's `mk_*` constructors cannot
    /// introduce variable capture.
    ///
    /// # Why not delegate to `TermManager::substitute`?
    ///
    /// The core routine intentionally stops at "complex" terms — it does not
    /// recurse through bit-vector operators or function applications, returning
    /// them unchanged.  Those node kinds are directly reachable through the Z3
    /// compat surface (`BV::*`, `FuncDecl::apply`), so a faithful Z3
    /// `substitute` must descend into them; hence the dedicated rebuild here.
    #[must_use]
    pub fn substitute(&self, expr: TermId, subst: &[(TermId, TermId)]) -> TermId {
        if subst.is_empty() {
            return expr;
        }
        let map: FxHashMap<TermId, TermId> = subst.iter().copied().collect();
        let mut cache: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut tm = self.tm.borrow_mut();
        subst_rebuild(&mut tm, expr, &map, &mut cache)
    }
}

/// Recursively rebuild `id`, replacing any node present in `map` and otherwise
/// reconstructing the term from substituted children.
///
/// `cache` memoizes already-rewritten nodes so shared subterms are visited once.
fn subst_rebuild(
    tm: &mut TermManager,
    id: TermId,
    map: &FxHashMap<TermId, TermId>,
    cache: &mut FxHashMap<TermId, TermId>,
) -> TermId {
    // Direct replacement takes precedence over structural recursion.
    if let Some(&to) = map.get(&id) {
        return to;
    }
    if let Some(&done) = cache.get(&id) {
        return done;
    }

    let kind = match tm.get(id).map(|t| t.kind.clone()) {
        Some(k) => k,
        None => return id,
    };

    // Helper closures cannot borrow `tm` mutably while also being called in a
    // loop, so children are rewritten inline via a small macro.
    macro_rules! rec {
        ($child:expr) => {
            subst_rebuild(tm, $child, map, cache)
        };
    }

    let result = match kind {
        // Leaves: never structurally rewritten (direct replacement handled
        // above). `Var` is a leaf too — a bound/free variable replaced only by
        // an explicit (from, to) pair.
        TermKind::True
        | TermKind::False
        | TermKind::IntConst(_)
        | TermKind::RealConst(_)
        | TermKind::BitVecConst { .. }
        | TermKind::StringLit(_)
        | TermKind::Var(_) => id,

        // ── Boolean ──────────────────────────────────────────────────────
        TermKind::Not(a) => {
            let na = rec!(a);
            if na == a { id } else { tm.mk_not(na) }
        }
        TermKind::And(args) => rebuild_nary(tm, id, &args, map, cache, |tm, a| tm.mk_and(a)),
        TermKind::Or(args) => rebuild_nary(tm, id, &args, map, cache, |tm, a| tm.mk_or(a)),
        TermKind::Xor(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_xor(na, nb)
            }
        }
        TermKind::Implies(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_implies(na, nb)
            }
        }
        TermKind::Ite(c, t, e) => {
            let (nc, nt, ne) = (rec!(c), rec!(t), rec!(e));
            if nc == c && nt == t && ne == e {
                id
            } else {
                tm.mk_ite(nc, nt, ne)
            }
        }

        // ── Equality / distinct ──────────────────────────────────────────
        TermKind::Eq(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_eq(na, nb)
            }
        }
        TermKind::Distinct(args) => {
            rebuild_nary(tm, id, &args, map, cache, |tm, a| tm.mk_distinct(a))
        }

        // ── Arithmetic ───────────────────────────────────────────────────
        TermKind::Neg(a) => {
            let na = rec!(a);
            if na == a { id } else { tm.mk_neg(na) }
        }
        TermKind::Add(args) => rebuild_nary(tm, id, &args, map, cache, |tm, a| tm.mk_add(a)),
        TermKind::Mul(args) => rebuild_nary(tm, id, &args, map, cache, |tm, a| tm.mk_mul(a)),
        TermKind::Sub(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_sub(na, nb)
            }
        }
        TermKind::Div(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_div(na, nb)
            }
        }
        TermKind::Mod(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_mod(na, nb)
            }
        }
        TermKind::Lt(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_lt(na, nb)
            }
        }
        TermKind::Le(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_le(na, nb)
            }
        }
        TermKind::Gt(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_gt(na, nb)
            }
        }
        TermKind::Ge(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_ge(na, nb)
            }
        }

        // ── Bit-vectors ──────────────────────────────────────────────────
        TermKind::BvConcat(a, b) => {
            let (na, nb) = (rec!(a), rec!(b));
            if na == a && nb == b {
                id
            } else {
                tm.mk_bv_concat(na, nb)
            }
        }
        TermKind::BvExtract { high, low, arg } => {
            let na = rec!(arg);
            if na == arg {
                id
            } else {
                tm.mk_bv_extract(high, low, na)
            }
        }
        TermKind::BvNot(a) => {
            let na = rec!(a);
            if na == a { id } else { tm.mk_bv_not(na) }
        }
        // Note: bit-vector negation has no dedicated `TermKind`; the builder
        // desugars `mk_bv_neg` into `BvSub(0, x)`, handled by the `BvSub` arm.
        TermKind::BvAnd(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_and),
        TermKind::BvOr(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_or),
        TermKind::BvXor(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_xor),
        TermKind::BvAdd(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_add),
        TermKind::BvSub(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_sub),
        TermKind::BvMul(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_mul),
        TermKind::BvUdiv(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_udiv),
        TermKind::BvSdiv(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_sdiv),
        TermKind::BvUrem(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_urem),
        TermKind::BvSrem(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_srem),
        TermKind::BvShl(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_shl),
        TermKind::BvLshr(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_lshr),
        TermKind::BvAshr(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_ashr),
        TermKind::BvUlt(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_ult),
        TermKind::BvUle(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_ule),
        TermKind::BvSlt(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_slt),
        TermKind::BvSle(a, b) => rebuild_bin(tm, id, a, b, map, cache, TermManager::mk_bv_sle),

        // ── Arrays ───────────────────────────────────────────────────────
        TermKind::Select(arr, idx) => {
            let (na, ni) = (rec!(arr), rec!(idx));
            if na == arr && ni == idx {
                id
            } else {
                tm.mk_select(na, ni)
            }
        }
        TermKind::Store(arr, idx, val) => {
            let (na, ni, nv) = (rec!(arr), rec!(idx), rec!(val));
            if na == arr && ni == idx && nv == val {
                id
            } else {
                tm.mk_store(na, ni, nv)
            }
        }

        // ── Uninterpreted-function application ───────────────────────────
        TermKind::Apply { func, args } => {
            let new_args: smallvec::SmallVec<[TermId; 4]> = args.iter().map(|&a| rec!(a)).collect();
            if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                id
            } else {
                let func_name = tm.resolve_str(func).to_string();
                let sort = tm.get(id).map_or(tm.sorts.bool_sort, |t| t.sort);
                tm.mk_apply(&func_name, new_args, sort)
            }
        }

        // Any other term kind (string operators, floating-point, datatypes,
        // quantifiers, let/match) is treated as opaque and replaced only by an
        // explicit (from, to) pair, which was already handled above. This is
        // safe and conservative for the ground-substitution contract.
        _ => id,
    };

    cache.insert(id, result);
    result
}

/// Rewrite the children of an n-ary node and rebuild it via `build` only if any
/// child changed.
///
/// `build` is generic over a closure so it can call the manager's n-ary
/// constructors (which take `impl IntoIterator<Item = TermId>`) directly,
/// avoiding fragile coercion of generic methods to function pointers.
fn rebuild_nary<F>(
    tm: &mut TermManager,
    id: TermId,
    args: &[TermId],
    map: &FxHashMap<TermId, TermId>,
    cache: &mut FxHashMap<TermId, TermId>,
    build: F,
) -> TermId
where
    F: FnOnce(&mut TermManager, smallvec::SmallVec<[TermId; 4]>) -> TermId,
{
    let new_args: smallvec::SmallVec<[TermId; 4]> = args
        .iter()
        .map(|&a| subst_rebuild(tm, a, map, cache))
        .collect();
    if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
        id
    } else {
        build(tm, new_args)
    }
}

/// Rewrite both operands of a binary node and rebuild it via `build` only if
/// either operand changed.
fn rebuild_bin<F>(
    tm: &mut TermManager,
    id: TermId,
    a: TermId,
    b: TermId,
    map: &FxHashMap<TermId, TermId>,
    cache: &mut FxHashMap<TermId, TermId>,
    build: F,
) -> TermId
where
    F: FnOnce(&mut TermManager, TermId, TermId) -> TermId,
{
    let na = subst_rebuild(tm, a, map, cache);
    let nb = subst_rebuild(tm, b, map, cache);
    if na == a && nb == b {
        id
    } else {
        build(tm, na, nb)
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
}
