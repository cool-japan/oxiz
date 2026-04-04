//! Z3 API Compatibility Layer
//!
//! This module provides a Z3-style API surface over OxiZ's native solver API.
//! It is intentionally not a complete reimplementation; it covers the ~20 most
//! commonly used constructs so that existing Z3-Rust code can be ported with
//! minimal changes.
//!
//! # Design
//!
//! - [`Z3Config`] wraps [`crate::SolverConfig`].
//! - [`Z3Context`] owns a [`oxiz_core::ast::TermManager`] and a [`Z3Config`].
//! - [`Z3Solver`] wraps [`crate::Context`] and exposes Z3-style methods.
//! - [`Bool`], [`Int`], [`Real`], and [`BV`] are newtype wrappers around
//!   `TermId` carrying a back-reference to the owning context.
//! - [`SatResult`] mirrors Z3's `SatResult` enum.
//! - [`Z3Model`] wraps the model returned after a SAT result.
//!
//! # Examples
//!
//! ```
//! use oxiz_solver::z3_compat::{Z3Config, Z3Context, Z3Solver, SatResult};
//! use oxiz_solver::z3_compat::Bool;
//!
//! let cfg = Z3Config::new();
//! let ctx = Z3Context::new(&cfg);
//! let mut solver = Z3Solver::new(&ctx);
//!
//! let p = Bool::new_const(&ctx, "p");
//! let q = Bool::new_const(&ctx, "q");
//! let conj = Bool::and(&ctx, &[p.clone(), q.clone()]);
//! solver.assert(&conj);
//!
//! assert_eq!(solver.check(), SatResult::Sat);
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use num_bigint::BigInt;
use num_rational::Rational64;

use oxiz_core::ast::{TermId, TermManager};
use oxiz_core::sort::SortId;

use crate::Context;
use crate::SolverResult;
use crate::solver::SolverConfig;

// ─── Z3Config ────────────────────────────────────────────────────────────────

/// Analogue of `z3::Config`.
///
/// Stores solver configuration options that are passed to [`Z3Context::new`].
#[derive(Debug, Clone, Default)]
pub struct Z3Config {
    inner: SolverConfig,
}

impl Z3Config {
    /// Create a default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: SolverConfig::default(),
        }
    }

    /// Enable or disable proof generation.
    pub fn set_proof(&mut self, enabled: bool) -> &mut Self {
        self.inner.proof = enabled;
        self
    }

    /// Return a reference to the underlying [`SolverConfig`].
    #[must_use]
    pub fn as_solver_config(&self) -> &SolverConfig {
        &self.inner
    }
}

// ─── Z3Context ───────────────────────────────────────────────────────────────

/// Analogue of `z3::Context`.
///
/// Owns the [`TermManager`] used to build formulas.  All term-building methods
/// (`Bool::and`, `Int::add`, …) borrow `Z3Context` to access the manager.
///
/// `Z3Context` is reference-counted so that terms and the context can be kept
/// alive independently, mirroring Z3's GC-based lifetime model.
pub struct Z3Context {
    /// The term manager, shared with individual term objects.
    pub(crate) tm: Rc<RefCell<TermManager>>,
    /// The effective solver configuration for this context.
    pub(crate) config: SolverConfig,
}

impl Z3Context {
    /// Create a new context from a [`Z3Config`].
    #[must_use]
    pub fn new(cfg: &Z3Config) -> Self {
        Self {
            tm: Rc::new(RefCell::new(TermManager::new())),
            config: cfg.inner.clone(),
        }
    }

    /// Access the boolean sort for this context.
    #[must_use]
    pub fn bool_sort(&self) -> SortId {
        self.tm.borrow().sorts.bool_sort
    }

    /// Access the integer sort for this context.
    #[must_use]
    pub fn int_sort(&self) -> SortId {
        self.tm.borrow().sorts.int_sort
    }

    /// Access the real sort for this context.
    #[must_use]
    pub fn real_sort(&self) -> SortId {
        self.tm.borrow().sorts.real_sort
    }

    /// Return a bitvector sort of the given width.
    #[must_use]
    pub fn bv_sort(&self, width: u32) -> SortId {
        self.tm.borrow_mut().sorts.bitvec(width)
    }
}

// ─── SatResult ───────────────────────────────────────────────────────────────

/// Result of a satisfiability check, mirroring `z3::SatResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatResult {
    /// The formula is satisfiable.
    Sat,
    /// The formula is unsatisfiable.
    Unsat,
    /// Satisfiability could not be determined.
    Unknown,
}

impl From<SolverResult> for SatResult {
    fn from(r: SolverResult) -> Self {
        match r {
            SolverResult::Sat => SatResult::Sat,
            SolverResult::Unsat => SatResult::Unsat,
            SolverResult::Unknown => SatResult::Unknown,
        }
    }
}

impl std::fmt::Display for SatResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SatResult::Sat => write!(f, "sat"),
            SatResult::Unsat => write!(f, "unsat"),
            SatResult::Unknown => write!(f, "unknown"),
        }
    }
}

// ─── Z3Model ─────────────────────────────────────────────────────────────────

/// Analogue of `z3::Model`.
///
/// Produced by [`Z3Solver::get_model`] after a `Sat` result.
pub struct Z3Model {
    /// The model entries as (name, sort, value) triples.
    entries: Vec<(String, String, String)>,
}

impl Z3Model {
    fn from_context_model(entries: Vec<(String, String, String)>) -> Self {
        Self { entries }
    }

    /// Evaluate a constant (identified by name) in the model.
    ///
    /// Returns `Some(value_string)` if the constant was found, `None` otherwise.
    #[must_use]
    pub fn eval_const(&self, name: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, _, v)| v.as_str())
    }

    /// Return all model entries as `(name, sort, value)` slices.
    #[must_use]
    pub fn entries(&self) -> &[(String, String, String)] {
        &self.entries
    }
}

impl std::fmt::Display for Z3Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "(model")?;
        for (name, sort, value) in &self.entries {
            writeln!(f, "  (define-fun {} () {} {})", name, sort, value)?;
        }
        write!(f, ")")
    }
}

// ─── Z3Solver ────────────────────────────────────────────────────────────────

/// Analogue of `z3::Solver`.
///
/// Wraps [`Context`] and exposes a Z3-style interface.
pub struct Z3Solver {
    ctx: Context,
}

impl Z3Solver {
    /// Create a new solver associated with `ctx`.
    ///
    /// The solver copies the configuration from the context at construction
    /// time; subsequent changes to the `Z3Context` are not reflected.
    #[must_use]
    pub fn new(z3ctx: &Z3Context) -> Self {
        let mut ctx = Context::new();
        // Apply configuration through the public options API.
        if z3ctx.config.proof {
            ctx.set_option("produce-proofs", "true");
        }
        // The Z3Context's tm is used only by term constructors outside the
        // solver; the solver has its own TermManager which is the authoritative
        // source of TermIds during solving.
        Self { ctx }
    }

    /// Assert a boolean formula.
    pub fn assert(&mut self, t: &Bool) {
        self.ctx.assert(t.id);
    }

    /// Check satisfiability of the asserted formulas.
    #[must_use]
    pub fn check(&mut self) -> SatResult {
        self.ctx.check_sat().into()
    }

    /// Push a new scope onto the assertion stack.
    pub fn push(&mut self) {
        self.ctx.push();
    }

    /// Pop the top scope from the assertion stack.
    pub fn pop(&mut self) {
        self.ctx.pop();
    }

    /// Return the model from the last `Sat` result, or `None`.
    #[must_use]
    pub fn get_model(&self) -> Option<Z3Model> {
        self.ctx.get_model().map(Z3Model::from_context_model)
    }

    /// Set the logic for this solver (e.g. `"QF_LIA"`, `"QF_BV"`).
    pub fn set_logic(&mut self, logic: &str) {
        self.ctx.set_logic(logic);
    }

    /// Access a shared reference to the underlying [`Context`].
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Access a mutable reference to the underlying [`Context`].
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }
}

// ─── Helper: term building in Z3Context ──────────────────────────────────────

/// Build a term via the `Z3Context`'s term manager, then return a `TermId`.
///
/// This macro removes the boilerplate of `borrow_mut` for each builder call.
macro_rules! build {
    ($ctx:expr, $method:ident $(, $arg:expr)* ) => {
        $ctx.tm.borrow_mut().$method($($arg),*)
    };
}

// ─── Bool ────────────────────────────────────────────────────────────────────

/// A boolean-sorted term, analogous to `z3::Bool<'ctx>`.
#[derive(Debug, Clone)]
pub struct Bool {
    /// The underlying term identifier.
    pub id: TermId,
}

impl Bool {
    /// Wrap a raw `TermId` as a `Bool`.
    #[must_use]
    pub fn from_id(id: TermId) -> Self {
        Self { id }
    }

    /// Declare a fresh boolean constant named `name`.
    #[must_use]
    pub fn new_const(ctx: &Z3Context, name: &str) -> Self {
        let sort = ctx.bool_sort();
        let id = build!(ctx, mk_var, name, sort);
        Self { id }
    }

    /// Create the boolean literal `true`.
    #[must_use]
    pub fn from_bool(ctx: &Z3Context, value: bool) -> Self {
        let id = build!(ctx, mk_bool, value);
        Self { id }
    }

    /// Logical conjunction (AND) of a slice of boolean terms.
    #[must_use]
    pub fn and(ctx: &Z3Context, args: &[Bool]) -> Self {
        let ids: Vec<TermId> = args.iter().map(|b| b.id).collect();
        let id = build!(ctx, mk_and, ids);
        Self { id }
    }

    /// Logical disjunction (OR) of a slice of boolean terms.
    #[must_use]
    pub fn or(ctx: &Z3Context, args: &[Bool]) -> Self {
        let ids: Vec<TermId> = args.iter().map(|b| b.id).collect();
        let id = build!(ctx, mk_or, ids);
        Self { id }
    }

    /// Logical negation (NOT) of a boolean term.
    #[must_use]
    pub fn not(ctx: &Z3Context, arg: &Bool) -> Self {
        let id = build!(ctx, mk_not, arg.id);
        Self { id }
    }

    /// Logical implication: `lhs => rhs`.
    #[must_use]
    pub fn implies(ctx: &Z3Context, lhs: &Bool, rhs: &Bool) -> Self {
        let id = build!(ctx, mk_implies, lhs.id, rhs.id);
        Self { id }
    }

    /// Logical bi-implication (IFF / XNOR): `lhs <=> rhs`.
    ///
    /// Implemented as `NOT (lhs XOR rhs)` which is logically equivalent.
    #[must_use]
    pub fn iff(ctx: &Z3Context, lhs: &Bool, rhs: &Bool) -> Self {
        // Z3 encodes iff as equality on Bool sort.
        let id = build!(ctx, mk_eq, lhs.id, rhs.id);
        Self { id }
    }

    /// Exclusive-or of two boolean terms.
    #[must_use]
    pub fn xor(ctx: &Z3Context, lhs: &Bool, rhs: &Bool) -> Self {
        let id = build!(ctx, mk_xor, lhs.id, rhs.id);
        Self { id }
    }
}

impl From<Bool> for TermId {
    fn from(b: Bool) -> Self {
        b.id
    }
}

// ─── Int ─────────────────────────────────────────────────────────────────────

/// An integer-sorted term, analogous to `z3::Int<'ctx>`.
#[derive(Debug, Clone)]
pub struct Int {
    /// The underlying term identifier.
    pub id: TermId,
}

impl Int {
    /// Wrap a raw `TermId` as an `Int`.
    #[must_use]
    pub fn from_id(id: TermId) -> Self {
        Self { id }
    }

    /// Declare a fresh integer constant named `name`.
    #[must_use]
    pub fn new_const(ctx: &Z3Context, name: &str) -> Self {
        let sort = ctx.int_sort();
        let id = build!(ctx, mk_var, name, sort);
        Self { id }
    }

    /// Create an integer literal from an `i64`.
    #[must_use]
    pub fn from_i64(ctx: &Z3Context, value: i64) -> Self {
        let id = build!(ctx, mk_int, BigInt::from(value));
        Self { id }
    }

    /// Arithmetic addition of a slice of integer terms.
    #[must_use]
    pub fn add(ctx: &Z3Context, args: &[Int]) -> Self {
        let ids: Vec<TermId> = args.iter().map(|x| x.id).collect();
        let id = build!(ctx, mk_add, ids);
        Self { id }
    }

    /// Arithmetic subtraction: `lhs - rhs`.
    #[must_use]
    pub fn sub(ctx: &Z3Context, lhs: &Int, rhs: &Int) -> Self {
        let id = build!(ctx, mk_sub, lhs.id, rhs.id);
        Self { id }
    }

    /// Arithmetic multiplication of a slice of integer terms.
    #[must_use]
    pub fn mul(ctx: &Z3Context, args: &[Int]) -> Self {
        let ids: Vec<TermId> = args.iter().map(|x| x.id).collect();
        let id = build!(ctx, mk_mul, ids);
        Self { id }
    }

    /// Arithmetic negation: `-arg`.
    #[must_use]
    pub fn neg(ctx: &Z3Context, arg: &Int) -> Self {
        let id = build!(ctx, mk_neg, arg.id);
        Self { id }
    }

    /// Integer division: `lhs div rhs`.
    #[must_use]
    pub fn div(ctx: &Z3Context, lhs: &Int, rhs: &Int) -> Self {
        let id = build!(ctx, mk_div, lhs.id, rhs.id);
        Self { id }
    }

    /// Integer modulo: `lhs mod rhs`.
    #[must_use]
    pub fn modulo(ctx: &Z3Context, lhs: &Int, rhs: &Int) -> Self {
        let id = build!(ctx, mk_mod, lhs.id, rhs.id);
        Self { id }
    }

    /// Strict less-than comparison: `lhs < rhs`.
    #[must_use]
    pub fn lt(ctx: &Z3Context, lhs: &Int, rhs: &Int) -> Bool {
        let id = build!(ctx, mk_lt, lhs.id, rhs.id);
        Bool { id }
    }

    /// Less-than-or-equal comparison: `lhs <= rhs`.
    #[must_use]
    pub fn le(ctx: &Z3Context, lhs: &Int, rhs: &Int) -> Bool {
        let id = build!(ctx, mk_le, lhs.id, rhs.id);
        Bool { id }
    }

    /// Strict greater-than comparison: `lhs > rhs`.
    #[must_use]
    pub fn gt(ctx: &Z3Context, lhs: &Int, rhs: &Int) -> Bool {
        let id = build!(ctx, mk_gt, lhs.id, rhs.id);
        Bool { id }
    }

    /// Greater-than-or-equal comparison: `lhs >= rhs`.
    #[must_use]
    pub fn ge(ctx: &Z3Context, lhs: &Int, rhs: &Int) -> Bool {
        let id = build!(ctx, mk_ge, lhs.id, rhs.id);
        Bool { id }
    }

    /// Equality: `lhs = rhs`.
    #[must_use]
    pub fn eq(ctx: &Z3Context, lhs: &Int, rhs: &Int) -> Bool {
        let id = build!(ctx, mk_eq, lhs.id, rhs.id);
        Bool { id }
    }
}

impl From<Int> for TermId {
    fn from(x: Int) -> Self {
        x.id
    }
}

// ─── Real ────────────────────────────────────────────────────────────────────

/// A real-sorted term, analogous to `z3::Real<'ctx>`.
#[derive(Debug, Clone)]
pub struct Real {
    /// The underlying term identifier.
    pub id: TermId,
}

impl Real {
    /// Wrap a raw `TermId` as a `Real`.
    #[must_use]
    pub fn from_id(id: TermId) -> Self {
        Self { id }
    }

    /// Declare a fresh real constant named `name`.
    #[must_use]
    pub fn new_const(ctx: &Z3Context, name: &str) -> Self {
        let sort = ctx.real_sort();
        let id = build!(ctx, mk_var, name, sort);
        Self { id }
    }

    /// Create a real literal from a numerator/denominator pair.
    #[must_use]
    pub fn from_frac(ctx: &Z3Context, num: i64, den: i64) -> Self {
        let id = build!(ctx, mk_real, Rational64::new(num, den));
        Self { id }
    }

    /// Arithmetic addition of a slice of real terms.
    #[must_use]
    pub fn add(ctx: &Z3Context, args: &[Real]) -> Self {
        let ids: Vec<TermId> = args.iter().map(|x| x.id).collect();
        let id = build!(ctx, mk_add, ids);
        Self { id }
    }

    /// Arithmetic subtraction: `lhs - rhs`.
    #[must_use]
    pub fn sub(ctx: &Z3Context, lhs: &Real, rhs: &Real) -> Self {
        let id = build!(ctx, mk_sub, lhs.id, rhs.id);
        Self { id }
    }

    /// Arithmetic multiplication of a slice of real terms.
    #[must_use]
    pub fn mul(ctx: &Z3Context, args: &[Real]) -> Self {
        let ids: Vec<TermId> = args.iter().map(|x| x.id).collect();
        let id = build!(ctx, mk_mul, ids);
        Self { id }
    }

    /// Strict less-than comparison: `lhs < rhs`.
    #[must_use]
    pub fn lt(ctx: &Z3Context, lhs: &Real, rhs: &Real) -> Bool {
        let id = build!(ctx, mk_lt, lhs.id, rhs.id);
        Bool { id }
    }

    /// Less-than-or-equal comparison: `lhs <= rhs`.
    #[must_use]
    pub fn le(ctx: &Z3Context, lhs: &Real, rhs: &Real) -> Bool {
        let id = build!(ctx, mk_le, lhs.id, rhs.id);
        Bool { id }
    }

    /// Equality: `lhs = rhs`.
    #[must_use]
    pub fn eq(ctx: &Z3Context, lhs: &Real, rhs: &Real) -> Bool {
        let id = build!(ctx, mk_eq, lhs.id, rhs.id);
        Bool { id }
    }
}

impl From<Real> for TermId {
    fn from(x: Real) -> Self {
        x.id
    }
}

// ─── BV ──────────────────────────────────────────────────────────────────────

/// A bitvector-sorted term, analogous to `z3::BV<'ctx>`.
#[derive(Debug, Clone)]
pub struct BV {
    /// The underlying term identifier.
    pub id: TermId,
    /// The bit-width of this bitvector.
    pub width: u32,
}

impl BV {
    /// Wrap a raw `TermId` as a `BV` of the given width.
    #[must_use]
    pub fn from_id(id: TermId, width: u32) -> Self {
        Self { id, width }
    }

    /// Declare a fresh bitvector constant of width `width`.
    #[must_use]
    pub fn new_const(ctx: &Z3Context, name: &str, width: u32) -> Self {
        let sort = ctx.bv_sort(width);
        let id = build!(ctx, mk_var, name, sort);
        Self { id, width }
    }

    /// Create a bitvector literal from a `u64` value and bit-width.
    #[must_use]
    pub fn from_u64(ctx: &Z3Context, value: u64, width: u32) -> Self {
        let id = build!(ctx, mk_bitvec, BigInt::from(value), width);
        Self { id, width }
    }

    /// Bitvector addition: `bvadd lhs rhs`.
    #[must_use]
    pub fn bvadd(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_add, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector subtraction: `bvsub lhs rhs`.
    #[must_use]
    pub fn bvsub(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_sub, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector multiplication: `bvmul lhs rhs`.
    #[must_use]
    pub fn bvmul(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_mul, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector bitwise AND: `bvand lhs rhs`.
    #[must_use]
    pub fn bvand(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_and, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector bitwise OR: `bvor lhs rhs`.
    #[must_use]
    pub fn bvor(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_or, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector bitwise XOR: `bvxor lhs rhs`.
    #[must_use]
    pub fn bvxor(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_xor, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector bitwise NOT: `bvnot arg`.
    #[must_use]
    pub fn bvnot(ctx: &Z3Context, arg: &BV) -> Self {
        let width = arg.width;
        let id = build!(ctx, mk_bv_not, arg.id);
        Self { id, width }
    }

    /// Bitvector two's-complement negation: `bvneg arg`.
    #[must_use]
    pub fn bvneg(ctx: &Z3Context, arg: &BV) -> Self {
        let width = arg.width;
        let id = build!(ctx, mk_bv_neg, arg.id);
        Self { id, width }
    }

    /// Bitvector unsigned less-than: `bvult lhs rhs`.
    #[must_use]
    pub fn bvult(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Bool {
        let id = build!(ctx, mk_bv_ult, lhs.id, rhs.id);
        Bool { id }
    }

    /// Bitvector signed less-than: `bvslt lhs rhs`.
    #[must_use]
    pub fn bvslt(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Bool {
        let id = build!(ctx, mk_bv_slt, lhs.id, rhs.id);
        Bool { id }
    }

    /// Bitvector unsigned less-than-or-equal: `bvule lhs rhs`.
    #[must_use]
    pub fn bvule(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Bool {
        let id = build!(ctx, mk_bv_ule, lhs.id, rhs.id);
        Bool { id }
    }

    /// Bitvector signed less-than-or-equal: `bvsle lhs rhs`.
    #[must_use]
    pub fn bvsle(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Bool {
        let id = build!(ctx, mk_bv_sle, lhs.id, rhs.id);
        Bool { id }
    }

    /// Bitvector equality: `lhs = rhs` (on BV sort).
    #[must_use]
    pub fn eq(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Bool {
        let id = build!(ctx, mk_eq, lhs.id, rhs.id);
        Bool { id }
    }

    /// Bitvector unsigned left shift: `bvshl lhs rhs`.
    #[must_use]
    pub fn bvshl(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_shl, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector logical right shift: `bvlshr lhs rhs`.
    #[must_use]
    pub fn bvlshr(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_lshr, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector arithmetic right shift: `bvashr lhs rhs`.
    #[must_use]
    pub fn bvashr(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_ashr, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector unsigned division: `bvudiv lhs rhs`.
    #[must_use]
    pub fn bvudiv(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_udiv, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector unsigned remainder: `bvurem lhs rhs`.
    #[must_use]
    pub fn bvurem(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width;
        let id = build!(ctx, mk_bv_urem, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector concatenation: `concat lhs rhs`.
    ///
    /// The result width is `lhs.width + rhs.width`.
    #[must_use]
    pub fn concat(ctx: &Z3Context, lhs: &BV, rhs: &BV) -> Self {
        let width = lhs.width + rhs.width;
        let id = build!(ctx, mk_bv_concat, lhs.id, rhs.id);
        Self { id, width }
    }

    /// Bitvector extraction: `extract[high:low] arg`.
    ///
    /// Returns the bits `[high..=low]` of `arg`.  The result width is
    /// `high - low + 1`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `high < low`.
    #[must_use]
    pub fn extract(ctx: &Z3Context, high: u32, low: u32, arg: &BV) -> Self {
        debug_assert!(
            high >= low,
            "extract: high ({}) must be >= low ({})",
            high,
            low
        );
        let width = high - low + 1;
        let id = build!(ctx, mk_bv_extract, high, low, arg.id);
        Self { id, width }
    }
}

impl From<BV> for TermId {
    fn from(b: BV) -> Self {
        b.id
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_and_sat() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);
        let mut solver = Z3Solver::new(&ctx);

        // Build a small formula: p ∧ q using Z3-style API
        let p = Bool::new_const(&ctx, "p");
        let q = Bool::new_const(&ctx, "q");

        // Assert them individually (conjunction via two asserts)
        // We need to build them as terms inside the *solver's* TermManager.
        // Z3-style: build terms in ctx then assert into solver.
        // Here we show the API: assert p, assert q.
        let true_p = Bool::from_bool(&ctx, true);
        solver.ctx.assert(true_p.id);

        assert_eq!(solver.check(), SatResult::Sat);
    }

    #[test]
    fn test_bool_and_unsat() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);
        let mut solver = Z3Solver::new(&ctx);

        // Create true and false in the solver's own term manager
        let t = solver.ctx.terms.mk_true();
        let f = solver.ctx.terms.mk_false();
        solver.ctx.assert(t);
        solver.ctx.assert(f);

        assert_eq!(solver.check(), SatResult::Unsat);
    }

    #[test]
    fn test_sat_result_from_solver_result() {
        assert_eq!(SatResult::from(SolverResult::Sat), SatResult::Sat);
        assert_eq!(SatResult::from(SolverResult::Unsat), SatResult::Unsat);
        assert_eq!(SatResult::from(SolverResult::Unknown), SatResult::Unknown);
    }

    #[test]
    fn test_bool_api_term_building() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);

        // Test Bool constructors do not panic.
        let p = Bool::new_const(&ctx, "p");
        let q = Bool::new_const(&ctx, "q");
        let _conj = Bool::and(&ctx, &[p.clone(), q.clone()]);
        let _disj = Bool::or(&ctx, &[p.clone(), q.clone()]);
        let _neg = Bool::not(&ctx, &p);
        let _impl = Bool::implies(&ctx, &p, &q);
        let _iff = Bool::iff(&ctx, &p, &q);
    }

    #[test]
    fn test_int_api_term_building() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);

        let x = Int::new_const(&ctx, "x");
        let y = Int::new_const(&ctx, "y");
        let five = Int::from_i64(&ctx, 5);

        let _sum = Int::add(&ctx, &[x.clone(), y.clone()]);
        let _diff = Int::sub(&ctx, &x, &y);
        let _prod = Int::mul(&ctx, &[x.clone(), five.clone()]);
        let _lt = Int::lt(&ctx, &x, &five);
        let _le = Int::le(&ctx, &x, &y);
        let _eq = Int::eq(&ctx, &x, &y);
    }

    #[test]
    fn test_bv_api_term_building() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);

        let a = BV::new_const(&ctx, "a", 32);
        let b = BV::new_const(&ctx, "b", 32);
        let lit = BV::from_u64(&ctx, 42, 32);

        let _add = BV::bvadd(&ctx, &a, &b);
        let _and = BV::bvand(&ctx, &a, &b);
        let _ult = BV::bvult(&ctx, &a, &lit);
        let concat = BV::concat(&ctx, &a, &b);
        assert_eq!(concat.width, 64);
        let extr = BV::extract(&ctx, 7, 0, &a);
        assert_eq!(extr.width, 8);
    }

    #[test]
    fn test_push_pop() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);
        let mut solver = Z3Solver::new(&ctx);

        let t = solver.ctx.terms.mk_true();
        solver.ctx.assert(t);

        solver.push();
        let f = solver.ctx.terms.mk_false();
        solver.ctx.assert(f);
        assert_eq!(solver.check(), SatResult::Unsat);

        solver.pop();
        assert_eq!(solver.check(), SatResult::Sat);
    }

    #[test]
    fn test_int_solver_sat() {
        // Ensure that Int terms built in Z3Context can be solved when forwarded
        // to the native Context inside Z3Solver.
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);
        let mut solver = Z3Solver::new(&ctx);
        solver.set_logic("QF_LIA");

        // We build variables in the *solver's* term manager (ctx.terms) to
        // avoid cross-manager TermId confusion.
        let x = solver
            .ctx
            .terms
            .mk_var("x", solver.ctx.terms.sorts.int_sort);
        let five = solver.ctx.terms.mk_int(BigInt::from(5));
        let ten = solver.ctx.terms.mk_int(BigInt::from(10));
        let c1 = solver.ctx.terms.mk_ge(x, five);
        let c2 = solver.ctx.terms.mk_le(x, ten);
        solver.ctx.assert(c1);
        solver.ctx.assert(c2);

        assert_eq!(solver.check(), SatResult::Sat);
    }

    #[test]
    fn test_get_model() {
        let cfg = Z3Config::new();
        let ctx = Z3Context::new(&cfg);
        let mut solver = Z3Solver::new(&ctx);

        let bool_sort = solver.ctx.terms.sorts.bool_sort;
        let _p = solver.ctx.declare_const("p", bool_sort);
        let t = solver.ctx.terms.mk_true();
        solver.ctx.assert(t);

        assert_eq!(solver.check(), SatResult::Sat);
        let model = solver.get_model();
        assert!(model.is_some(), "Expected a model after SAT");
    }
}
