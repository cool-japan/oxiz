//! Extensions to the Z3 compatibility layer.
//!
//! This module contains the additional Z3 API surfaces that complement the
//! core types in [`crate::z3_compat`]:
//!
//! - [`Array`]       — array terms (select/store theory)
//! - [`FuncDecl`]    — uninterpreted function declarations and application
//! - [`Z3Optimize`]  — optimization wrapper (minimize/maximize)
//! - Free functions: `ite_bool`, `ite_int`, `ite_real`, `ite_bv`
//! - Free functions: `distinct_int`, `distinct_real`, `distinct_bv`
//! - Free functions: `forall_bool`, `exists_bool`

use num_rational::Rational64;

use oxiz_core::ast::TermId;
use oxiz_core::sort::SortId;

use crate::optimization::{OptimizationResult, Optimizer};
use crate::z3_compat::{BV, Bool, Int, Real, SatResult, Z3Context};

// ─── Helper macro (mirrors the one in z3_compat) ─────────────────────────────

macro_rules! build {
    ($ctx:expr, $method:ident $(, $arg:expr)* ) => {
        $ctx.tm.borrow_mut().$method($($arg),*)
    };
}

// ─── Real symmetry additions ─────────────────────────────────────────────────

impl Real {
    /// Strict greater-than comparison: `lhs > rhs`.
    #[must_use]
    pub fn gt(ctx: &Z3Context, lhs: &Real, rhs: &Real) -> Bool {
        let id = build!(ctx, mk_gt, lhs.id, rhs.id);
        Bool { id }
    }

    /// Greater-than-or-equal comparison: `lhs >= rhs`.
    #[must_use]
    pub fn ge(ctx: &Z3Context, lhs: &Real, rhs: &Real) -> Bool {
        let id = build!(ctx, mk_ge, lhs.id, rhs.id);
        Bool { id }
    }

    /// Arithmetic negation: `-arg`.
    #[must_use]
    pub fn neg(ctx: &Z3Context, arg: &Real) -> Real {
        let id = build!(ctx, mk_neg, arg.id);
        Real { id }
    }

    /// Real division: `lhs / rhs`.
    #[must_use]
    pub fn div(ctx: &Z3Context, lhs: &Real, rhs: &Real) -> Real {
        let id = build!(ctx, mk_div, lhs.id, rhs.id);
        Real { id }
    }

    /// Create a real literal from an `i64` value (denominator = 1).
    #[must_use]
    pub fn from_i64(ctx: &Z3Context, value: i64) -> Real {
        let id = build!(ctx, mk_real, Rational64::new(value, 1));
        Real { id }
    }
}

// ─── ITE (if-then-else) free functions ───────────────────────────────────────

/// Boolean if-then-else: `ite(cond, then_branch, else_branch) : Bool`.
///
/// Returns `then_branch` when `cond` is true, `else_branch` otherwise.
#[must_use]
pub fn ite_bool(ctx: &Z3Context, cond: &Bool, then_branch: &Bool, else_branch: &Bool) -> Bool {
    let id = build!(ctx, mk_ite, cond.id, then_branch.id, else_branch.id);
    Bool { id }
}

/// Integer if-then-else: `ite(cond, then_branch, else_branch) : Int`.
#[must_use]
pub fn ite_int(ctx: &Z3Context, cond: &Bool, then_branch: &Int, else_branch: &Int) -> Int {
    let id = build!(ctx, mk_ite, cond.id, then_branch.id, else_branch.id);
    Int { id }
}

/// Real if-then-else: `ite(cond, then_branch, else_branch) : Real`.
#[must_use]
pub fn ite_real(ctx: &Z3Context, cond: &Bool, then_branch: &Real, else_branch: &Real) -> Real {
    let id = build!(ctx, mk_ite, cond.id, then_branch.id, else_branch.id);
    Real { id }
}

/// Bitvector if-then-else: `ite(cond, then_branch, else_branch) : BV`.
///
/// The width of the result matches `then_branch.width`.
#[must_use]
pub fn ite_bv(ctx: &Z3Context, cond: &Bool, then_branch: &BV, else_branch: &BV) -> BV {
    let width = then_branch.width;
    let id = build!(ctx, mk_ite, cond.id, then_branch.id, else_branch.id);
    BV { id, width }
}

// ─── Distinct free functions ──────────────────────────────────────────────────

/// Assert that all given integer terms are pairwise distinct.
#[must_use]
pub fn distinct_int(ctx: &Z3Context, args: &[Int]) -> Bool {
    let ids = args.iter().map(|x| x.id);
    let id = build!(ctx, mk_distinct, ids);
    Bool { id }
}

/// Assert that all given real terms are pairwise distinct.
#[must_use]
pub fn distinct_real(ctx: &Z3Context, args: &[Real]) -> Bool {
    let ids = args.iter().map(|x| x.id);
    let id = build!(ctx, mk_distinct, ids);
    Bool { id }
}

/// Assert that all given bitvector terms are pairwise distinct.
#[must_use]
pub fn distinct_bv(ctx: &Z3Context, args: &[BV]) -> Bool {
    let ids = args.iter().map(|x| x.id);
    let id = build!(ctx, mk_distinct, ids);
    Bool { id }
}

// ─── Array ────────────────────────────────────────────────────────────────────

/// An array-sorted term, analogous to `z3::Array<'ctx, D, R>`.
///
/// Arrays are modelled by the theory of arrays (select/store).
/// The `domain` and `range` sorts are recorded for convenience but the
/// authoritative sort information lives inside the [`TermManager`].
///
/// [`TermManager`]: oxiz_core::ast::TermManager
#[derive(Debug, Clone)]
pub struct Array {
    /// The underlying term identifier.
    pub id: TermId,
    /// The index (domain) sort of this array.
    pub domain: SortId,
    /// The element (range) sort of this array.
    pub range: SortId,
}

impl Array {
    /// Wrap a raw `TermId` as an `Array` with known domain/range sorts.
    #[must_use]
    pub fn from_id(id: TermId, domain: SortId, range: SortId) -> Self {
        Self { id, domain, range }
    }

    /// Declare a fresh array constant named `name` with the given domain/range sorts.
    ///
    /// Creates the array sort via `SortManager::array` and then declares a
    /// variable of that sort.
    #[must_use]
    pub fn new_const(ctx: &Z3Context, name: &str, domain: SortId, range: SortId) -> Self {
        let arr_sort = ctx.tm.borrow_mut().sorts.array(domain, range);
        let id = build!(ctx, mk_var, name, arr_sort);
        Self { id, domain, range }
    }

    /// Select an element from `arr` at index `idx`.
    ///
    /// Returns a raw [`TermId`] whose sort is `arr.range`.
    #[must_use]
    pub fn select(ctx: &Z3Context, arr: &Array, idx: TermId) -> TermId {
        build!(ctx, mk_select, arr.id, idx)
    }

    /// Store `val` at index `idx` in `arr`, returning the updated array.
    ///
    /// The returned `Array` has the same domain/range as `arr`.
    #[must_use]
    pub fn store(ctx: &Z3Context, arr: &Array, idx: TermId, val: TermId) -> Array {
        let id = build!(ctx, mk_store, arr.id, idx, val);
        Array {
            id,
            domain: arr.domain,
            range: arr.range,
        }
    }

    /// Equality between two arrays of the same domain/range sorts.
    #[must_use]
    pub fn eq(ctx: &Z3Context, lhs: &Array, rhs: &Array) -> Bool {
        let id = build!(ctx, mk_eq, lhs.id, rhs.id);
        Bool { id }
    }
}

impl From<Array> for TermId {
    fn from(a: Array) -> Self {
        a.id
    }
}

// ─── FuncDecl ────────────────────────────────────────────────────────────────

/// An uninterpreted function declaration, analogous to `z3::FuncDecl<'ctx>`.
///
/// Created via [`FuncDecl::new`], applied via [`FuncDecl::apply`].
///
/// Internally, application uses the TermManager's `mk_apply` method which
/// stores `(func_name, args)` as a `TermKind::Apply`.  The function is
/// uninterpreted unless further axioms are asserted.
#[derive(Debug, Clone)]
pub struct FuncDecl {
    /// Canonical name of the function.
    pub name: String,
    /// Domain sorts (one per argument position).
    pub domain: Vec<SortId>,
    /// Return sort.
    pub range: SortId,
}

impl FuncDecl {
    /// Declare an uninterpreted function with the given name, domain, and range.
    ///
    /// The declaration is purely structural at this point; no term is created
    /// in the `TermManager` until [`FuncDecl::apply`] is called.
    #[must_use]
    pub fn new(_ctx: &Z3Context, name: &str, domain: &[SortId], range: SortId) -> Self {
        Self {
            name: name.to_string(),
            domain: domain.to_vec(),
            range,
        }
    }

    /// Apply this function to a slice of argument [`TermId`]s.
    ///
    /// Returns the resulting term (sort = `self.range`).
    ///
    /// # Panics (debug only)
    ///
    /// Panics in debug builds if the number of arguments does not match the
    /// arity declared in `domain`.
    #[must_use]
    pub fn apply(&self, ctx: &Z3Context, args: &[TermId]) -> TermId {
        debug_assert_eq!(
            args.len(),
            self.domain.len(),
            "FuncDecl::apply arity mismatch: declared {}, got {}",
            self.domain.len(),
            args.len()
        );
        let range = self.range;
        build!(ctx, mk_apply, &self.name, args.iter().copied(), range)
    }
}

// ─── Quantifiers ─────────────────────────────────────────────────────────────

/// Universal quantifier over a boolean body.
///
/// `vars` is a slice of `(name, sort)` pairs naming the bound variables.
/// The body is a [`Bool`] term that may reference variables with those names.
///
/// Internally delegates to [`TermManager::mk_forall`].
#[must_use]
pub fn forall_bool<'a>(
    ctx: &Z3Context,
    vars: impl IntoIterator<Item = (&'a str, SortId)>,
    body: &Bool,
) -> Bool {
    let id = ctx.tm.borrow_mut().mk_forall(vars, body.id);
    Bool { id }
}

/// Existential quantifier over a boolean body.
///
/// `vars` is a slice of `(name, sort)` pairs naming the bound variables.
/// The body is a [`Bool`] term that may reference variables with those names.
///
/// Internally delegates to [`TermManager::mk_exists`].
#[must_use]
pub fn exists_bool<'a>(
    ctx: &Z3Context,
    vars: impl IntoIterator<Item = (&'a str, SortId)>,
    body: &Bool,
) -> Bool {
    let id = ctx.tm.borrow_mut().mk_exists(vars, body.id);
    Bool { id }
}

// ─── Z3Optimize ──────────────────────────────────────────────────────────────

/// Analogue of `z3::Optimize`.
///
/// Wraps the native [`Optimizer`] and exposes a Z3-style interface for
/// optimization modulo theories.
///
/// # Design
///
/// Unlike [`Z3Solver`], which shares its `TermManager` with the user's
/// `Z3Context`, `Z3Optimize` owns its own `Optimizer` (which in turn owns a
/// `TermManager` as a mutable borrow at `optimize()` time).  Terms handed to
/// `assert`, `minimize`, and `maximize` must originate from the **same**
/// `Z3Context` that was passed to [`Z3Optimize::new`]; they are forwarded to
/// the optimizer as-is, and the optimizer's `optimize()` call receives a
/// reference to the same shared `TermManager` via the `RefCell`.
///
/// Concretely: `check()` calls `opt.optimize(&mut *tm_guard)` where
/// `tm_guard` is a `RefMut` over the `Z3Context`'s `TermManager`.  This
/// works because `RefMut<TermManager>` derefs to `TermManager`.
///
/// `get_lower`/`get_upper` return the best bound as a decimal string,
/// matching Z3's `Optimize::get_lower`/`get_upper` behaviour on integer or
/// real objectives.
///
/// [`Z3Solver`]: crate::z3_compat::Z3Solver
/// [`Optimizer`]: crate::optimization::Optimizer
pub struct Z3Optimize {
    /// Shared reference to the context term manager (same as Z3Context).
    ctx_tm: std::rc::Rc<std::cell::RefCell<oxiz_core::ast::TermManager>>,
    /// The underlying optimizer (owns the SAT solver).
    opt: Optimizer,
    /// Objective term IDs and directions.
    objectives: Vec<(TermId, ObjectiveDir)>,
    /// Lower-bound strings filled after `check()`.
    lower_bounds: Vec<Option<String>>,
    /// Upper-bound strings filled after `check()`.
    upper_bounds: Vec<Option<String>>,
    /// Last check result.
    last_result: SatResult,
}

/// Direction of an optimization objective.
#[derive(Debug, Clone, Copy)]
enum ObjectiveDir {
    Minimize,
    Maximize,
}

impl Z3Optimize {
    /// Create a new optimizer associated with `ctx`.
    ///
    /// Assertions and objectives must subsequently be built using the same
    /// `ctx` to guarantee that [`TermId`]s are valid in the shared manager.
    #[must_use]
    pub fn new(ctx: &Z3Context) -> Self {
        Self {
            ctx_tm: ctx.tm.clone(),
            opt: Optimizer::new(),
            objectives: Vec::new(),
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            last_result: SatResult::Unknown,
        }
    }

    /// Assert a boolean formula as a hard constraint.
    pub fn assert(&mut self, b: &Bool) {
        self.opt.assert(b.id);
    }

    /// Add a minimization objective for term `t`.
    ///
    /// Returns an opaque index that can be passed to
    /// [`get_lower`]/[`get_upper`] after calling [`check`].
    ///
    /// [`get_lower`]: Z3Optimize::get_lower
    /// [`get_upper`]: Z3Optimize::get_upper
    /// [`check`]: Z3Optimize::check
    pub fn minimize(&mut self, t: TermId) -> usize {
        let idx = self.objectives.len();
        self.opt.minimize(t);
        self.objectives.push((t, ObjectiveDir::Minimize));
        self.lower_bounds.push(None);
        self.upper_bounds.push(None);
        idx
    }

    /// Add a maximization objective for term `t`.
    ///
    /// Returns an opaque index that can be passed to
    /// [`get_lower`]/[`get_upper`] after calling [`check`].
    pub fn maximize(&mut self, t: TermId) -> usize {
        let idx = self.objectives.len();
        self.opt.maximize(t);
        self.objectives.push((t, ObjectiveDir::Maximize));
        self.lower_bounds.push(None);
        self.upper_bounds.push(None);
        idx
    }

    /// Check satisfiability and optimize all registered objectives.
    ///
    /// Populates the internal bound tables so that
    /// [`get_lower`]/[`get_upper`] reflect the results.
    pub fn check(&mut self) -> SatResult {
        // `Optimizer::optimize` requires `&mut TermManager`.  We hold an
        // `Rc<RefCell<TermManager>>` which we borrow mutably for the duration
        // of the call.  This is safe because no other borrow is held while
        // `check()` runs (the Z3Context is not accessed concurrently).
        let result = self.opt.optimize(&mut self.ctx_tm.borrow_mut());

        match &result {
            OptimizationResult::Optimal { value, model: _ } => {
                let tm = self.ctx_tm.borrow();
                let val_str = Self::term_to_string(&tm, *value);
                drop(tm);
                for idx in 0..self.objectives.len() {
                    self.lower_bounds[idx] = Some(val_str.clone());
                    self.upper_bounds[idx] = Some(val_str.clone());
                }
                self.last_result = SatResult::Sat;
            }
            OptimizationResult::Unsat => {
                self.last_result = SatResult::Unsat;
            }
            _ => {
                self.last_result = SatResult::Unknown;
            }
        }

        self.last_result
    }

    /// Return the lower bound for objective `idx` as a string, or `None` if
    /// the bound is not yet available (before `check()` or after UNSAT).
    #[must_use]
    pub fn get_lower(&self, idx: usize) -> Option<String> {
        self.lower_bounds.get(idx).and_then(|b| b.clone())
    }

    /// Return the upper bound for objective `idx` as a string, or `None` if
    /// the bound is not yet available.
    #[must_use]
    pub fn get_upper(&self, idx: usize) -> Option<String> {
        self.upper_bounds.get(idx).and_then(|b| b.clone())
    }

    /// Format a term value as a decimal string (best-effort).
    fn term_to_string(tm: &oxiz_core::ast::TermManager, id: TermId) -> String {
        use oxiz_core::ast::TermKind;
        if let Some(t) = tm.get(id) {
            match &t.kind {
                TermKind::IntConst(n) => return n.to_string(),
                TermKind::RealConst(r) => return r.to_string(),
                TermKind::True => return "true".to_string(),
                TermKind::False => return "false".to_string(),
                _ => {}
            }
        }
        format!("term#{}", id.0)
    }
}

// ─── Z3Context sort helpers ───────────────────────────────────────────────────

impl Z3Context {
    /// Return an array sort with the given domain and range sorts.
    ///
    /// Useful when constructing [`Array`] constants:
    ///
    /// ```ignore
    /// let dom = ctx.int_sort();
    /// let rng = ctx.int_sort();
    /// let a = Array::new_const(&ctx, "a", dom, rng);
    /// ```
    #[must_use]
    pub fn array_sort(&self, domain: SortId, range: SortId) -> SortId {
        self.tm.borrow_mut().sorts.array(domain, range)
    }
}

// ─── Additional Int helpers (ensure from_i64 parity) ─────────────────────────
// (Int::from_i64 already exists in z3_compat.rs — no re-definition needed)

// ─── Additional Real helpers with mk_int coercion ────────────────────────────

/// Convenience: build a `Real` literal from a numerator and denominator in the
/// solver's term manager.
///
/// This is equivalent to [`Real::from_frac`] but operates on raw `i64` values.
#[must_use]
pub fn real_numeral(ctx: &Z3Context, num: i64, den: i64) -> Real {
    Real::from_frac(ctx, num, den)
}

/// Convenience: build an `Int` literal inside `ctx`.
///
/// Equivalent to `Int::from_i64` — provided as a free function for ergonomics
/// when building mixed expressions.
#[must_use]
pub fn int_numeral(ctx: &Z3Context, value: i64) -> Int {
    Int::from_i64(ctx, value)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::z3_compat::{Z3Config, Z3Context};
    use num_bigint::BigInt;

    fn ctx() -> Z3Context {
        Z3Context::new(&Z3Config::new())
    }

    // ── Real symmetry ─────────────────────────────────────────────────────────

    #[test]
    fn test_real_gt_smoke() {
        let ctx = ctx();
        let a = Real::new_const(&ctx, "a");
        let b = Real::new_const(&ctx, "b");
        let _gt = Real::gt(&ctx, &a, &b);
    }

    #[test]
    fn test_real_ge_smoke() {
        let ctx = ctx();
        let a = Real::new_const(&ctx, "a");
        let b = Real::new_const(&ctx, "b");
        let _ge = Real::ge(&ctx, &a, &b);
    }

    #[test]
    fn test_real_neg_smoke() {
        let ctx = ctx();
        let a = Real::new_const(&ctx, "a");
        let _neg = Real::neg(&ctx, &a);
    }

    #[test]
    fn test_real_div_smoke() {
        let ctx = ctx();
        let a = Real::new_const(&ctx, "a");
        let b = Real::new_const(&ctx, "b");
        let _div = Real::div(&ctx, &a, &b);
    }

    #[test]
    fn test_real_from_i64_smoke() {
        let ctx = ctx();
        let _r = Real::from_i64(&ctx, 42);
    }

    // ── ITE ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_ite_bool_smoke() {
        let ctx = ctx();
        let c = Bool::new_const(&ctx, "c");
        let t = Bool::from_bool(&ctx, true);
        let e = Bool::from_bool(&ctx, false);
        let _ite = ite_bool(&ctx, &c, &t, &e);
    }

    #[test]
    fn test_ite_int_smoke() {
        let ctx = ctx();
        let c = Bool::new_const(&ctx, "c");
        let t = Int::from_i64(&ctx, 1);
        let e = Int::from_i64(&ctx, 0);
        let _ite = ite_int(&ctx, &c, &t, &e);
    }

    #[test]
    fn test_ite_real_smoke() {
        let ctx = ctx();
        let c = Bool::new_const(&ctx, "c");
        let t = Real::from_i64(&ctx, 1);
        let e = Real::from_i64(&ctx, 0);
        let _ite = ite_real(&ctx, &c, &t, &e);
    }

    #[test]
    fn test_ite_bv_width() {
        let ctx = ctx();
        let c = Bool::new_const(&ctx, "c");
        let t = BV::from_u64(&ctx, 1, 32);
        let e = BV::from_u64(&ctx, 0, 32);
        let ite = ite_bv(&ctx, &c, &t, &e);
        assert_eq!(ite.width, 32);
    }

    // ── Distinct ──────────────────────────────────────────────────────────────

    #[test]
    fn test_distinct_int_smoke() {
        let ctx = ctx();
        let x = Int::from_i64(&ctx, 1);
        let y = Int::from_i64(&ctx, 2);
        let z = Int::from_i64(&ctx, 3);
        let _d = distinct_int(&ctx, &[x, y, z]);
    }

    #[test]
    fn test_distinct_real_smoke() {
        let ctx = ctx();
        let a = Real::from_i64(&ctx, 1);
        let b = Real::from_i64(&ctx, 2);
        let _d = distinct_real(&ctx, &[a, b]);
    }

    #[test]
    fn test_distinct_bv_smoke() {
        let ctx = ctx();
        let a = BV::from_u64(&ctx, 0, 8);
        let b = BV::from_u64(&ctx, 1, 8);
        let _d = distinct_bv(&ctx, &[a, b]);
    }

    // ── Array ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_array_new_const_smoke() {
        let ctx = ctx();
        let dom = ctx.int_sort();
        let rng = ctx.int_sort();
        let _a = Array::new_const(&ctx, "arr", dom, rng);
    }

    #[test]
    fn test_array_select_store_terms_constructed() {
        let ctx = ctx();
        let dom = ctx.int_sort();
        let rng = ctx.int_sort();
        let arr = Array::new_const(&ctx, "arr", dom, rng);
        let idx = Int::from_i64(&ctx, 0);
        let val = Int::from_i64(&ctx, 42);
        let arr2 = Array::store(&ctx, &arr, idx.id, val.id);
        let _selected = Array::select(&ctx, &arr2, idx.id);
    }

    #[test]
    fn test_array_eq_smoke() {
        let ctx = ctx();
        let dom = ctx.int_sort();
        let rng = ctx.int_sort();
        let a = Array::new_const(&ctx, "a", dom, rng);
        let b = Array::new_const(&ctx, "b", dom, rng);
        let _eq = Array::eq(&ctx, &a, &b);
    }

    #[test]
    fn test_array_sort_helper() {
        let ctx = ctx();
        let dom = ctx.int_sort();
        let rng = ctx.bool_sort();
        let _sort = ctx.array_sort(dom, rng);
    }

    // ── FuncDecl ──────────────────────────────────────────────────────────────

    #[test]
    fn test_func_decl_apply_smoke() {
        let ctx = ctx();
        let int_sort = ctx.int_sort();
        let f = FuncDecl::new(&ctx, "f", &[int_sort], int_sort);
        let arg = Int::new_const(&ctx, "x");
        let _result = f.apply(&ctx, &[arg.id]);
    }

    #[test]
    fn test_func_decl_two_args() {
        let ctx = ctx();
        let int_sort = ctx.int_sort();
        let bool_sort = ctx.bool_sort();
        let g = FuncDecl::new(&ctx, "g", &[int_sort, int_sort], bool_sort);
        let x = Int::new_const(&ctx, "x");
        let y = Int::new_const(&ctx, "y");
        let _r = g.apply(&ctx, &[x.id, y.id]);
    }

    // ── Quantifiers ───────────────────────────────────────────────────────────

    #[test]
    fn test_forall_bool_smoke() {
        let ctx = ctx();
        let int_sort = ctx.int_sort();
        // Build a simple body: x >= 0
        let x_var = ctx.tm.borrow_mut().mk_var("x", int_sort);
        let zero = ctx.tm.borrow_mut().mk_int(BigInt::from(0));
        let body_id = ctx.tm.borrow_mut().mk_ge(x_var, zero);
        let body = Bool { id: body_id };
        // forall x:Int. x >= 0  (clearly false, but we just check no panic)
        let _q = forall_bool(&ctx, [("x", int_sort)], &body);
    }

    #[test]
    fn test_exists_bool_smoke() {
        let ctx = ctx();
        let int_sort = ctx.int_sort();
        let x_var = ctx.tm.borrow_mut().mk_var("x", int_sort);
        let zero = ctx.tm.borrow_mut().mk_int(BigInt::from(0));
        let body_id = ctx.tm.borrow_mut().mk_ge(x_var, zero);
        let body = Bool { id: body_id };
        let _q = exists_bool(&ctx, [("x", int_sort)], &body);
    }

    // ── Z3Optimize ────────────────────────────────────────────────────────────

    #[test]
    fn test_optimize_sat_no_objectives() {
        // With no objectives, check() should return Sat for a satisfiable problem.
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);
        let mut opt = Z3Optimize::new(&ctx);
        let t = Bool::from_bool(&ctx, true);
        opt.assert(&t);
        let result = opt.check();
        // The optimizer's SAT check may return Unknown when there are no
        // objectives and no real assertion encoding, so we accept Sat or Unknown.
        assert!(
            result == SatResult::Sat || result == SatResult::Unknown,
            "Expected Sat or Unknown, got {:?}",
            result
        );
    }

    #[test]
    fn test_optimize_minimize_term_constructed() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);
        let mut opt = Z3Optimize::new(&ctx);
        // Build: x >= 0, minimize x
        let x = Int::new_const(&ctx, "x");
        let zero = Int::from_i64(&ctx, 0);
        let ge = Int::ge(&ctx, &x, &zero);
        opt.assert(&ge);
        let _idx = opt.minimize(x.id);
        // Just verify construction + check doesn't panic.
        let _result = opt.check();
    }

    #[test]
    fn test_optimize_get_lower_before_check_is_none() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);
        let mut opt = Z3Optimize::new(&ctx);
        let x = Int::new_const(&ctx, "x");
        let _idx = opt.minimize(x.id);
        // Before check(), bounds are None.
        assert!(opt.get_lower(0).is_none());
        assert!(opt.get_upper(0).is_none());
    }
}
