//! Shared linear real-arithmetic (LRA) primitives for quantifier elimination.
//!
//! The Ferrante–Rackoff and Loos–Weispfenning virtual-substitution eliminators
//! both operate on the same underlying object: a quantifier-free boolean
//! combination of linear comparison atoms over the reals. This module factors
//! out the machinery they share — linear-form parsing with exact rational
//! arithmetic, boundary/test-point construction, the `x → ±∞` limit rewrite,
//! and the infinitesimal `x → t + ε` virtual substitution — so each eliminator
//! only has to express its own test set.
//!
//! All internal arithmetic is carried out in exact [`BigRational`]; conversion
//! to the term representation's [`Rational64`] happens only when a constant is
//! materialised, and reports an honest `None`/`Err` on `i64` overflow rather
//! than silently truncating.
//!
//! Reference: Z3's `qe_arith.cpp` (linear real-arithmetic projection used by
//! `qe_lite`/`nlqsat`).

use crate::ast::{TermId, TermKind, TermManager};
use crate::interner::Spur;
use crate::prelude::FxHashMap;
use num_bigint::BigInt;
use num_rational::{BigRational, Rational64};
use num_traits::{One, Signed, ToPrimitive, Zero};

/// A comparison relation `expr REL 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rel {
    /// `< 0`
    Lt,
    /// `≤ 0`
    Le,
    /// `> 0`
    Gt,
    /// `≥ 0`
    Ge,
    /// `= 0`
    Eq,
    /// `≠ 0`
    Ne,
}

impl Rel {
    /// Logical negation of the relation (resolves polarity under `Not`).
    pub(crate) fn negate(self) -> Self {
        match self {
            Rel::Lt => Rel::Ge,
            Rel::Le => Rel::Gt,
            Rel::Gt => Rel::Le,
            Rel::Ge => Rel::Lt,
            Rel::Eq => Rel::Ne,
            Rel::Ne => Rel::Eq,
        }
    }

    /// Swap `<`/`>` and `≤`/`≥` (used when isolating `x` divides by a negative
    /// coefficient, which reverses the inequality direction).
    pub(crate) fn flip(self) -> Self {
        match self {
            Rel::Lt => Rel::Gt,
            Rel::Le => Rel::Ge,
            Rel::Gt => Rel::Lt,
            Rel::Ge => Rel::Le,
            Rel::Eq => Rel::Eq,
            Rel::Ne => Rel::Ne,
        }
    }
}

/// A linear form `x_coeff·x + Σ others + constant` with exact rational
/// coefficients. `others` maps an opaque, `x`-free sub-term to its coefficient.
#[derive(Debug, Clone)]
pub(crate) struct LinForm {
    /// Coefficient of the eliminated variable `x`.
    pub(crate) x_coeff: BigRational,
    /// Coefficients of the remaining (`x`-free) sub-terms.
    pub(crate) others: FxHashMap<TermId, BigRational>,
    /// Constant term.
    pub(crate) constant: BigRational,
}

impl LinForm {
    fn zero() -> Self {
        Self {
            x_coeff: BigRational::zero(),
            others: FxHashMap::default(),
            constant: BigRational::zero(),
        }
    }

    fn constant_val(c: BigRational) -> Self {
        Self {
            x_coeff: BigRational::zero(),
            others: FxHashMap::default(),
            constant: c,
        }
    }

    fn x() -> Self {
        Self {
            x_coeff: BigRational::one(),
            others: FxHashMap::default(),
            constant: BigRational::zero(),
        }
    }

    fn atom(id: TermId) -> Self {
        let mut others = FxHashMap::default();
        others.insert(id, BigRational::one());
        Self {
            x_coeff: BigRational::zero(),
            others,
            constant: BigRational::zero(),
        }
    }

    fn is_constant(&self) -> bool {
        self.x_coeff.is_zero() && self.others.values().all(BigRational::is_zero)
    }

    fn neg(mut self) -> Self {
        self.x_coeff = -self.x_coeff;
        self.constant = -self.constant;
        for v in self.others.values_mut() {
            *v = -core::mem::replace(v, BigRational::zero());
        }
        self
    }

    fn add(mut self, other: Self) -> Self {
        self.x_coeff += other.x_coeff;
        self.constant += other.constant;
        for (k, v) in other.others {
            let entry = self.others.entry(k).or_insert_with(BigRational::zero);
            *entry += v;
        }
        self
    }

    fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    fn scale(mut self, k: &BigRational) -> Self {
        self.x_coeff *= k;
        self.constant *= k;
        for v in self.others.values_mut() {
            *v *= k;
        }
        self
    }
}

/// A parsed comparison atom `x_coeff·x + Σ others + constant  REL  0` in which
/// the eliminated variable occurs with a non-zero coefficient.
#[derive(Debug, Clone)]
pub(crate) struct XAtom {
    /// Coefficient of `x` (guaranteed non-zero).
    pub(crate) x_coeff: BigRational,
    /// Coefficients of the remaining (`x`-free) sub-terms.
    pub(crate) others: FxHashMap<TermId, BigRational>,
    /// Constant term.
    pub(crate) constant: BigRational,
    /// The relation of `form REL 0`.
    pub(crate) rel: Rel,
}

/// Convert a [`Rational64`] to an exact [`BigRational`].
fn rational64_to_big(r: Rational64) -> BigRational {
    BigRational::new(BigInt::from(*r.numer()), BigInt::from(*r.denom()))
}

/// Whether `id` syntactically mentions the eliminated variable `x`.
///
/// Iterative, with a visited set. The result type `bool` has no error
/// channel, so a depth cap could only ever answer "does not mention `x`" for
/// a term it never finished inspecting — and this predicate gates the whole
/// eliminator, so a wrong `false` silently drops a constraint. The visited
/// set also stops a shared-subterm DAG from being re-walked as a tree; this
/// predicate is called at the top of three separate recursions, which made
/// the re-expansion quadratic at best.
pub(crate) fn mentions_var(id: TermId, x: Spur, tm: &TermManager) -> bool {
    let mut stack = vec![id];
    let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(term) = tm.get(current) else {
            continue;
        };
        if let TermKind::Var(s) = term.kind {
            if s == x {
                return true;
            }
            continue;
        }
        stack.extend(crate::ast::traversal::get_children(&term.kind));
    }

    false
}

/// Map every sub-term reachable from `root` to whether it mentions `x`.
///
/// The structural walkers below need this predicate at *every* node they
/// visit. Calling [`mentions_var`] per node re-walks the whole sub-tree each
/// time, which is quadratic on a deeply nested formula; one shared post-order
/// pass answers all of them in linear time. A term the manager cannot resolve
/// is recorded as `false`, exactly as [`mentions_var`] treats it.
fn mentions_map(root: TermId, x: Spur, tm: &TermManager) -> FxHashMap<TermId, bool> {
    /// Work item of the iterative post-order pass.
    enum Step {
        /// Schedule the children of a term.
        Enter(TermId),
        /// Combine the children's answers into this term's answer.
        Build(TermId),
    }

    let mut map: FxHashMap<TermId, bool> = FxHashMap::default();
    let mut stack = vec![Step::Enter(root)];

    while let Some(step) = stack.pop() {
        match step {
            Step::Enter(current) => {
                if map.contains_key(&current) {
                    continue;
                }
                let Some(term) = tm.get(current) else {
                    map.insert(current, false);
                    continue;
                };
                if let TermKind::Var(s) = term.kind {
                    map.insert(current, s == x);
                    continue;
                }
                let children = crate::ast::traversal::get_children(&term.kind);
                if children.is_empty() {
                    map.insert(current, false);
                    continue;
                }
                stack.push(Step::Build(current));
                for &c in children.iter() {
                    stack.push(Step::Enter(c));
                }
            }
            Step::Build(current) => {
                let Some(term) = tm.get(current) else {
                    map.insert(current, false);
                    continue;
                };
                let children = crate::ast::traversal::get_children(&term.kind);
                // Every child was fully processed before this `Build` popped.
                let any = children
                    .iter()
                    .any(|c| map.get(c).copied().unwrap_or(false));
                map.insert(current, any);
            }
        }
    }

    map
}

/// Look a node up in a [`mentions_map`].
///
/// A missing entry cannot happen for a node reached from the map's root, but
/// if it ever did, answering `true` keeps the walker on the path that either
/// rewrites the atom properly or reports an honest `Err` — answering `false`
/// would silently return an `x`-containing term as if it were `x`-free.
fn mentions_lookup(map: &FxHashMap<TermId, bool>, id: TermId) -> bool {
    map.get(&id).copied().unwrap_or(true)
}

/// Sort of the first syntactic occurrence of `x` in `id`, if any.
///
/// Iterative pre-order walk that visits children left-to-right, which is the
/// order in which the recursive form found "the first" occurrence. All
/// occurrences of a variable share one `TermId` (terms are hash-consed), so
/// every occurrence carries the same sort and the choice is immaterial for
/// the returned value; the order is preserved anyway.
pub(crate) fn find_var_sort(id: TermId, x: Spur, tm: &TermManager) -> Option<crate::sort::SortId> {
    let mut stack = vec![id];
    let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(term) = tm.get(current) else {
            continue;
        };
        if let TermKind::Var(s) = term.kind {
            if s == x {
                return Some(term.sort);
            }
            continue;
        }
        let children = crate::ast::traversal::get_children(&term.kind);
        stack.extend(children.iter().rev().copied());
    }

    None
}

/// Parse `id` into a [`LinForm`] over `x`, or `None` if `x` occurs non-linearly
/// (e.g. `x*x`, `x` under an uninterpreted symbol) or the term is unsupported.
pub(crate) fn to_linear(id: TermId, x: Spur, tm: &TermManager) -> Option<LinForm> {
    /// Work item of the iterative linear-form parser.
    enum Step {
        /// Classify a term and schedule its operands.
        Enter(TermId),
        /// Fold already-parsed operands into this term's linear form.
        Build(TermId),
    }

    // Explicit stack plus a memo: the recursive form had one frame per level
    // of arithmetic nesting (each holding a `LinForm` with an `FxHashMap` of
    // `BigRational`s) and re-parsed shared sub-terms once per occurrence.
    // The memo is keyed on `TermId` alone, which is exact here — the linear
    // form of a term depends only on the term and on `x`.
    let mut memo: FxHashMap<TermId, LinForm> = FxHashMap::default();
    let mut stack = vec![Step::Enter(id)];

    while let Some(step) = stack.pop() {
        match step {
            Step::Enter(current) => {
                if memo.contains_key(&current) {
                    continue;
                }
                let term = tm.get(current)?;
                match &term.kind {
                    TermKind::IntConst(n) => {
                        memo.insert(
                            current,
                            LinForm::constant_val(BigRational::from_integer(n.clone())),
                        );
                    }
                    TermKind::RealConst(r) => {
                        memo.insert(current, LinForm::constant_val(rational64_to_big(*r)));
                    }
                    TermKind::Var(s) => {
                        let form = if *s == x {
                            LinForm::x()
                        } else {
                            LinForm::atom(current)
                        };
                        memo.insert(current, form);
                    }
                    TermKind::Neg(a) => {
                        stack.push(Step::Build(current));
                        stack.push(Step::Enter(*a));
                    }
                    TermKind::Add(args) | TermKind::Mul(args) => {
                        stack.push(Step::Build(current));
                        for &a in args.iter() {
                            stack.push(Step::Enter(a));
                        }
                    }
                    TermKind::Sub(a, b) | TermKind::Div(a, b) => {
                        stack.push(Step::Build(current));
                        stack.push(Step::Enter(*a));
                        stack.push(Step::Enter(*b));
                    }
                    _ => {
                        // Any other term is acceptable only as an opaque atom
                        // that does not mention `x`; otherwise `x` occurs in
                        // an unsupported position and the caller is told so.
                        if mentions_var(current, x, tm) {
                            return None;
                        }
                        memo.insert(current, LinForm::atom(current));
                    }
                }
            }
            Step::Build(current) => {
                let term = tm.get(current)?;
                let form = match &term.kind {
                    TermKind::Neg(a) => memo.get(a)?.clone().neg(),
                    TermKind::Add(args) => {
                        let mut acc = LinForm::zero();
                        for a in args.iter() {
                            acc = acc.add(memo.get(a)?.clone());
                        }
                        acc
                    }
                    TermKind::Sub(a, b) => {
                        let la = memo.get(a)?.clone();
                        let lb = memo.get(b)?.clone();
                        la.sub(lb)
                    }
                    TermKind::Mul(args) => {
                        let mut const_prod = BigRational::one();
                        let mut nonconst: Option<LinForm> = None;
                        for a in args.iter() {
                            let lf = memo.get(a)?.clone();
                            if lf.is_constant() {
                                const_prod *= lf.constant;
                            } else if nonconst.is_none() {
                                nonconst = Some(lf);
                            } else {
                                // Product of two non-constant factors → non-linear in x.
                                return None;
                            }
                        }
                        match nonconst {
                            Some(lf) => lf.scale(&const_prod),
                            None => LinForm::constant_val(const_prod),
                        }
                    }
                    TermKind::Div(a, b) => {
                        // Real division by a non-zero constant: `a / c`.
                        let lb = memo.get(b)?;
                        if !lb.is_constant() || lb.constant.is_zero() {
                            return None;
                        }
                        let inv = lb.constant.recip();
                        memo.get(a)?.clone().scale(&inv)
                    }
                    // `Build` is only ever scheduled for the kinds above.
                    _ => return None,
                };
                memo.insert(current, form);
            }
        }
    }

    memo.remove(&id)
}

/// Materialise a rational constant as a real term, or `None` on `i64` overflow.
pub(crate) fn mk_real_const(tm: &mut TermManager, r: &BigRational) -> Option<TermId> {
    let num = r.numer().to_i64()?;
    let den = r.denom().to_i64()?;
    Some(tm.mk_real(Rational64::new(num, den)))
}

/// A real zero constant.
fn zero_real(tm: &mut TermManager) -> TermId {
    tm.mk_real(Rational64::new(0, 1))
}

/// Materialise `Σ others + constant` (guaranteed `x`-free) as a real term, or
/// `None` if any coefficient overflows `i64`.
pub(crate) fn mk_lin_term(
    others: &FxHashMap<TermId, BigRational>,
    constant: &BigRational,
    tm: &mut TermManager,
) -> Option<TermId> {
    let mut entries: Vec<(TermId, BigRational)> = others
        .iter()
        .filter(|(_, c)| !c.is_zero())
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    entries.sort_by_key(|(k, _)| k.0);

    let mut parts: Vec<TermId> = Vec::new();
    for (atom, coeff) in entries {
        if coeff.is_one() {
            parts.push(atom);
        } else {
            let c = mk_real_const(tm, &coeff)?;
            parts.push(tm.mk_mul(vec![c, atom]));
        }
    }
    if !constant.is_zero() || parts.is_empty() {
        let c = mk_real_const(tm, constant)?;
        parts.push(c);
    }
    Some(if parts.len() == 1 {
        parts[0]
    } else {
        tm.mk_add(parts)
    })
}

/// The boundary value `-（Σ others + constant) / x_coeff` of an atom (the point
/// at which its linear form equals zero), materialised as an `x`-free real term.
pub(crate) fn boundary_term(
    x_coeff: &BigRational,
    others: &FxHashMap<TermId, BigRational>,
    constant: &BigRational,
    tm: &mut TermManager,
) -> Option<TermId> {
    let factor = -x_coeff.recip(); // -1/c
    let scaled_others: FxHashMap<TermId, BigRational> =
        others.iter().map(|(k, v)| (*k, v * &factor)).collect();
    let scaled_const = constant * &factor;
    mk_lin_term(&scaled_others, &scaled_const, tm)
}

/// Emit a comparison `lhs REL rhs`.
fn emit_cmp(lhs: TermId, rhs: TermId, rel: Rel, tm: &mut TermManager) -> TermId {
    match rel {
        Rel::Lt => tm.mk_lt(lhs, rhs),
        Rel::Le => tm.mk_le(lhs, rhs),
        Rel::Gt => tm.mk_gt(lhs, rhs),
        Rel::Ge => tm.mk_ge(lhs, rhs),
        Rel::Eq => tm.mk_eq(lhs, rhs),
        Rel::Ne => {
            let e = tm.mk_eq(lhs, rhs);
            tm.mk_not(e)
        }
    }
}

/// Truth value of `form REL 0` when `form` tends to `+∞` (`to_pos_inf = true`)
/// or `-∞` (`to_pos_inf = false`).
fn inf_truth(rel: Rel, to_pos_inf: bool) -> bool {
    if to_pos_inf {
        matches!(rel, Rel::Gt | Rel::Ge | Rel::Ne)
    } else {
        matches!(rel, Rel::Lt | Rel::Le | Rel::Ne)
    }
}

/// Collect every comparison atom mentioning `x` into `out`, or return `Err` if
/// `x` appears in a position outside the supported linear fragment.
///
/// Atoms in which `x` cancels (zero net coefficient) are skipped: they are
/// `x`-free and contribute no boundary.
///
/// Iterative pre-order walk over the boolean skeleton, in the same
/// left-to-right order the recursive form used. A `visited` set stops a
/// sub-formula shared by several parents from being expanded once per path:
/// the collected atoms form a *test set*, and a repeated sub-formula yields
/// exactly the same atoms, so visiting it once loses nothing (both callers
/// sort and `dedup` the boundary terms anyway) while turning a would-be
/// exponential DAG expansion into a linear walk.
pub(crate) fn collect_x_atoms(
    formula: TermId,
    x: Spur,
    tm: &TermManager,
    out: &mut Vec<XAtom>,
) -> Result<(), String> {
    let mentions = mentions_map(formula, x, tm);
    let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();
    let mut stack = vec![formula];

    while let Some(current) = stack.pop() {
        if !mentions_lookup(&mentions, current) {
            continue;
        }
        if !visited.insert(current) {
            continue;
        }
        let Some(term) = tm.get(current) else {
            return Err("lra: term not found".to_string());
        };
        match &term.kind {
            TermKind::Not(a) => stack.push(*a),
            TermKind::And(args) | TermKind::Or(args) => {
                stack.extend(args.iter().rev().copied());
            }
            TermKind::Implies(a, b) | TermKind::Xor(a, b) => {
                stack.push(*b);
                stack.push(*a);
            }
            TermKind::Ite(c, t, e) => {
                stack.push(*e);
                stack.push(*t);
                stack.push(*c);
            }
            TermKind::Lt(a, b) => push_cmp_atom(*a, *b, Rel::Lt, x, tm, out)?,
            TermKind::Le(a, b) => push_cmp_atom(*a, *b, Rel::Le, x, tm, out)?,
            TermKind::Gt(a, b) => push_cmp_atom(*a, *b, Rel::Gt, x, tm, out)?,
            TermKind::Ge(a, b) => push_cmp_atom(*a, *b, Rel::Ge, x, tm, out)?,
            TermKind::Eq(a, b) => push_cmp_atom(*a, *b, Rel::Eq, x, tm, out)?,
            _ => {
                return Err("lra: unsupported term mentioning the eliminated variable".to_string());
            }
        }
    }

    Ok(())
}

fn push_cmp_atom(
    a: TermId,
    b: TermId,
    rel: Rel,
    x: Spur,
    tm: &TermManager,
    out: &mut Vec<XAtom>,
) -> Result<(), String> {
    let fa = to_linear(a, x, tm).ok_or("lra: non-linear comparison operand")?;
    let fb = to_linear(b, x, tm).ok_or("lra: non-linear comparison operand")?;
    let form = fa.sub(fb);
    if form.x_coeff.is_zero() {
        // `x` cancels out — the atom is effectively `x`-free.
        return Ok(());
    }
    out.push(XAtom {
        x_coeff: form.x_coeff,
        others: form.others,
        constant: form.constant,
        rel,
    });
    Ok(())
}

/// Rewrite `formula` to its truth value as `x → +∞` (`at_plus_inf = true`) or
/// `x → -∞` (`at_plus_inf = false`), producing an `x`-free equivalent of the
/// formula on the corresponding unbounded interval.
///
/// The limit commutes with the boolean connectives, so the rewrite is purely
/// structural: each comparison atom mentioning `x` collapses to a boolean
/// constant, while `x`-free atoms are preserved.
pub(crate) fn inf_rewrite(
    formula: TermId,
    x: Spur,
    at_plus_inf: bool,
    tm: &mut TermManager,
) -> Result<TermId, String> {
    structural_rewrite(formula, x, &AtomMode::Inf(at_plus_inf), tm)
}

/// How a comparison atom mentioning `x` is rewritten by [`structural_rewrite`].
///
/// Both eliminators share one structural walk over the boolean skeleton and
/// differ only in what a comparison atom becomes.
enum AtomMode {
    /// Limit `x → +∞` (`true`) or `x → -∞` (`false`).
    Inf(bool),
    /// Virtual substitution `x := s + ε`.
    Eps(TermId),
}

/// Rewrite the comparison atom `a REL b` according to `mode`.
fn atom_rewrite(
    a: TermId,
    b: TermId,
    rel: Rel,
    x: Spur,
    mode: &AtomMode,
    tm: &mut TermManager,
) -> Result<TermId, String> {
    match *mode {
        AtomMode::Inf(at_plus_inf) => atom_inf(a, b, rel, x, at_plus_inf, tm),
        AtomMode::Eps(s) => atom_eps(a, b, rel, x, s, tm),
    }
}

/// Structural rewrite of the boolean skeleton of `formula`, replacing every
/// comparison atom that mentions `x` according to `mode` and leaving `x`-free
/// sub-formulae untouched.
///
/// Explicit work stack (two phases: schedule operands, then rebuild the node
/// from the already-rewritten operands) plus a memo. Both rewrites are a
/// function of the sub-term alone once `x` and `mode` are fixed — neither
/// carries polarity or any other context down the walk — so memoising on
/// `TermId` is exact, and it keeps a sub-formula shared by several parents
/// (an `Xor`/`Ite` chain re-expands one operand under two parents per level)
/// from being rewritten once per path.
fn structural_rewrite(
    formula: TermId,
    x: Spur,
    mode: &AtomMode,
    tm: &mut TermManager,
) -> Result<TermId, String> {
    /// Work item of the iterative rewrite.
    enum Step {
        /// Classify a sub-formula and schedule its operands.
        Enter(TermId),
        /// Rebuild a node from its already-rewritten operands.
        Build(TermId),
    }

    let mentions = mentions_map(formula, x, tm);
    let mut memo: FxHashMap<TermId, TermId> = FxHashMap::default();
    let mut stack = vec![Step::Enter(formula)];

    while let Some(step) = stack.pop() {
        match step {
            Step::Enter(current) => {
                if memo.contains_key(&current) {
                    continue;
                }
                if !mentions_lookup(&mentions, current) {
                    // `x`-free sub-formula: both rewrites are the identity.
                    memo.insert(current, current);
                    continue;
                }
                let kind = tm.get(current).ok_or("lra: term not found")?.kind.clone();
                match kind {
                    TermKind::Not(a) => {
                        stack.push(Step::Build(current));
                        stack.push(Step::Enter(a));
                    }
                    TermKind::And(args) | TermKind::Or(args) => {
                        stack.push(Step::Build(current));
                        for &a in args.iter() {
                            stack.push(Step::Enter(a));
                        }
                    }
                    TermKind::Implies(a, b) | TermKind::Xor(a, b) => {
                        stack.push(Step::Build(current));
                        stack.push(Step::Enter(a));
                        stack.push(Step::Enter(b));
                    }
                    TermKind::Ite(c, t, e) => {
                        stack.push(Step::Build(current));
                        stack.push(Step::Enter(c));
                        stack.push(Step::Enter(t));
                        stack.push(Step::Enter(e));
                    }
                    TermKind::Lt(a, b) => {
                        let r = atom_rewrite(a, b, Rel::Lt, x, mode, tm)?;
                        memo.insert(current, r);
                    }
                    TermKind::Le(a, b) => {
                        let r = atom_rewrite(a, b, Rel::Le, x, mode, tm)?;
                        memo.insert(current, r);
                    }
                    TermKind::Gt(a, b) => {
                        let r = atom_rewrite(a, b, Rel::Gt, x, mode, tm)?;
                        memo.insert(current, r);
                    }
                    TermKind::Ge(a, b) => {
                        let r = atom_rewrite(a, b, Rel::Ge, x, mode, tm)?;
                        memo.insert(current, r);
                    }
                    TermKind::Eq(a, b) => {
                        let r = atom_rewrite(a, b, Rel::Eq, x, mode, tm)?;
                        memo.insert(current, r);
                    }
                    _ => {
                        return Err(
                            "lra: unsupported term mentioning the eliminated variable".to_string()
                        );
                    }
                }
            }
            Step::Build(current) => {
                let kind = tm.get(current).ok_or("lra: term not found")?.kind.clone();
                let rewritten = |id: TermId, memo: &FxHashMap<TermId, TermId>| {
                    memo.get(&id)
                        .copied()
                        .ok_or_else(|| "lra: internal error: unrewritten sub-formula".to_string())
                };
                let built = match kind {
                    TermKind::Not(a) => {
                        let a = rewritten(a, &memo)?;
                        tm.mk_not(a)
                    }
                    TermKind::And(args) => {
                        let mut out = Vec::with_capacity(args.len());
                        for &a in args.iter() {
                            out.push(rewritten(a, &memo)?);
                        }
                        tm.mk_and(out)
                    }
                    TermKind::Or(args) => {
                        let mut out = Vec::with_capacity(args.len());
                        for &a in args.iter() {
                            out.push(rewritten(a, &memo)?);
                        }
                        tm.mk_or(out)
                    }
                    TermKind::Implies(a, b) => {
                        let (a, b) = (rewritten(a, &memo)?, rewritten(b, &memo)?);
                        tm.mk_implies(a, b)
                    }
                    TermKind::Xor(a, b) => {
                        let (a, b) = (rewritten(a, &memo)?, rewritten(b, &memo)?);
                        tm.mk_xor(a, b)
                    }
                    TermKind::Ite(c, t, e) => {
                        let c = rewritten(c, &memo)?;
                        let t = rewritten(t, &memo)?;
                        let e = rewritten(e, &memo)?;
                        tm.mk_ite(c, t, e)
                    }
                    // `Build` is only ever scheduled for the kinds above.
                    _ => {
                        return Err("lra: internal error: unexpected rebuild target".to_string());
                    }
                };
                memo.insert(current, built);
            }
        }
    }

    memo.get(&formula)
        .copied()
        .ok_or_else(|| "lra: internal error: rewrite produced no result".to_string())
}

fn atom_inf(
    a: TermId,
    b: TermId,
    rel: Rel,
    x: Spur,
    at_plus_inf: bool,
    tm: &mut TermManager,
) -> Result<TermId, String> {
    let fa = to_linear(a, x, tm).ok_or("lra: non-linear comparison operand")?;
    let fb = to_linear(b, x, tm).ok_or("lra: non-linear comparison operand")?;
    let form = fa.sub(fb);
    if form.x_coeff.is_zero() {
        let rest = mk_lin_term(&form.others, &form.constant, tm)
            .ok_or("lra: coefficient too large to eliminate")?;
        let zero = zero_real(tm);
        return Ok(emit_cmp(rest, zero, rel, tm));
    }
    // `form` tends to +∞ when its `x`-coefficient sign agrees with the
    // direction in which `x` grows.
    let to_pos_inf = at_plus_inf == form.x_coeff.is_positive();
    Ok(tm.mk_bool(inf_truth(rel, to_pos_inf)))
}

/// Virtually substitute `x := s + ε` (with `ε` a positive infinitesimal) into
/// `formula`, eliminating `ε` symbolically through the Loos–Weispfenning atom
/// rewriting rules. The result is an `x`-free (and `ε`-free) formula equivalent
/// to `φ` holding on an open interval immediately above `s`.
pub(crate) fn eps_rewrite(
    formula: TermId,
    x: Spur,
    s: TermId,
    tm: &mut TermManager,
) -> Result<TermId, String> {
    structural_rewrite(formula, x, &AtomMode::Eps(s), tm)
}

fn atom_eps(
    a: TermId,
    b: TermId,
    rel: Rel,
    x: Spur,
    s: TermId,
    tm: &mut TermManager,
) -> Result<TermId, String> {
    let fa = to_linear(a, x, tm).ok_or("lra: non-linear comparison operand")?;
    let fb = to_linear(b, x, tm).ok_or("lra: non-linear comparison operand")?;
    let form = fa.sub(fb); // c·x + rest
    let c = form.x_coeff.clone();
    let rest = mk_lin_term(&form.others, &form.constant, tm)
        .ok_or("lra: coefficient too large to eliminate")?;
    let zero = zero_real(tm);
    if c.is_zero() {
        // `x`-free atom: value is `rest REL 0`, unaffected by the ε shift.
        return Ok(emit_cmp(rest, zero, rel, tm));
    }
    // q = form[x := s] = c·s + rest, so form[x := s+ε] = q + c·ε.
    let c_term = mk_real_const(tm, &c).ok_or("lra: coefficient too large to eliminate")?;
    let cs = tm.mk_mul(vec![c_term, s]);
    let q = tm.mk_add(vec![cs, rest]);
    let c_pos = c.is_positive();
    // Evaluate `q + c·ε REL 0` in the limit ε → 0⁺ (c is a known non-zero
    // rational, so the sign of the infinitesimal term is decided statically).
    let out = match (rel, c_pos) {
        (Rel::Lt, true) => emit_cmp(q, zero, Rel::Lt, tm),
        (Rel::Lt, false) => emit_cmp(q, zero, Rel::Le, tm),
        (Rel::Le, true) => emit_cmp(q, zero, Rel::Lt, tm),
        (Rel::Le, false) => emit_cmp(q, zero, Rel::Le, tm),
        (Rel::Gt, true) => emit_cmp(q, zero, Rel::Ge, tm),
        (Rel::Gt, false) => emit_cmp(q, zero, Rel::Gt, tm),
        (Rel::Ge, true) => emit_cmp(q, zero, Rel::Ge, tm),
        (Rel::Ge, false) => emit_cmp(q, zero, Rel::Gt, tm),
        // c·ε ≠ 0 is infinitesimal, so equality can never hold on an interval.
        (Rel::Eq, _) => tm.mk_false(),
        (Rel::Ne, _) => tm.mk_true(),
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_var(tm: &mut TermManager, name: &str) -> TermId {
        let real_sort = tm.sorts.real_sort;
        tm.mk_var(name, real_sort)
    }

    #[test]
    fn to_linear_parses_affine_term() {
        // 3*x - 5  over the reals.
        let mut tm = TermManager::new();
        let x = real_var(&mut tm, "x");
        let three = tm.mk_real(Rational64::new(3, 1));
        let three_x = tm.mk_mul(vec![three, x]);
        let five = tm.mk_real(Rational64::new(5, 1));
        let expr = tm.mk_sub(three_x, five);

        let x_spur = tm.intern_str("x");
        let form = to_linear(expr, x_spur, &tm).expect("linear");
        assert_eq!(form.x_coeff, BigRational::from_integer(BigInt::from(3)));
        assert_eq!(form.constant, BigRational::from_integer(BigInt::from(-5)));
        assert!(form.others.is_empty());
    }

    #[test]
    fn to_linear_rejects_nonlinear() {
        let mut tm = TermManager::new();
        let x = real_var(&mut tm, "x");
        let xx = tm.mk_mul(vec![x, x]);
        let x_spur = tm.intern_str("x");
        assert!(to_linear(xx, x_spur, &tm).is_none());
    }

    #[test]
    fn inf_truth_table() {
        assert!(inf_truth(Rel::Lt, false));
        assert!(!inf_truth(Rel::Lt, true));
        assert!(inf_truth(Rel::Gt, true));
        assert!(!inf_truth(Rel::Gt, false));
        assert!(!inf_truth(Rel::Eq, true));
        assert!(!inf_truth(Rel::Eq, false));
        assert!(inf_truth(Rel::Ne, true));
        assert!(inf_truth(Rel::Ne, false));
    }
}

#[cfg(test)]
mod deep_walk_tests {
    use super::*;

    #[test]
    fn test_mentions_var_shared_dag_is_fast() {
        // Two-strand DAG, 55 levels: 2^55 nodes without a visited set.
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let (mut a, mut b) = (x, y);
        for _ in 0..55 {
            let next_a = tm.mk_sub(a, b);
            let next_b = tm.mk_add([b, a]);
            a = next_a;
            b = next_b;
        }
        let x_spur = tm.intern_str("x");
        let z_spur = tm.intern_str("z");

        assert!(mentions_var(a, x_spur, &tm));
        assert!(!mentions_var(a, z_spur, &tm));
        assert_eq!(find_var_sort(a, x_spur, &tm), Some(int_sort));
        assert_eq!(find_var_sort(a, z_spur, &tm), None);
    }

    /// A real variable plus the atoms used by the structural tests.
    fn deep_setup(tm: &mut TermManager) -> (TermId, TermId, TermId) {
        let real_sort = tm.sorts.real_sort;
        let x = tm.mk_var("x", real_sort);
        let y = tm.mk_var("y", real_sort);
        let zero = tm.mk_real(Rational64::new(0, 1));
        let atom_x = tm.mk_lt(x, zero);
        let atom_y = tm.mk_lt(y, zero);
        (atom_x, atom_y, zero)
    }

    /// `levels` alternating `And`/`Or` nestings around an `x` atom. Alternating
    /// the connective defeats the n-ary flattening in `mk_and`/`mk_or`, so the
    /// boolean skeleton really is `levels` deep.
    fn deep_bool_formula(tm: &mut TermManager, levels: usize) -> TermId {
        let (atom_x, atom_y, _) = deep_setup(tm);
        let mut f = atom_x;
        for _ in 0..levels / 2 {
            f = tm.mk_and([f, atom_y]);
            f = tm.mk_or([f, atom_y]);
        }
        f
    }

    #[test]
    fn test_collect_x_atoms_deep_nesting_does_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut tm = TermManager::new();
                let f = deep_bool_formula(&mut tm, 6_250);
                let x_spur = tm.intern_str("x");
                let mut atoms = Vec::new();
                let outcome = collect_x_atoms(f, x_spur, &tm, &mut atoms);
                (outcome, atoms.len())
            })
            .expect("thread spawn should succeed");

        let (outcome, n) = handle.join().expect("collect_x_atoms must not overflow");
        assert!(outcome.is_ok(), "collect_x_atoms failed: {outcome:?}");
        assert_eq!(n, 1, "the single shared x atom is collected exactly once");
    }

    #[test]
    fn test_inf_rewrite_deep_nesting_does_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut tm = TermManager::new();
                let f = deep_bool_formula(&mut tm, 6_250);
                let x_spur = tm.intern_str("x");
                let neg = inf_rewrite(f, x_spur, false, &mut tm);
                let pos = inf_rewrite(f, x_spur, true, &mut tm);
                let still_mentions = neg
                    .as_ref()
                    .ok()
                    .map(|&t| mentions_var(t, x_spur, &tm))
                    .unwrap_or(true);
                (neg.is_ok(), pos.is_ok(), still_mentions)
            })
            .expect("thread spawn should succeed");

        let (neg_ok, pos_ok, still_mentions) =
            handle.join().expect("inf_rewrite must not overflow");
        assert!(neg_ok && pos_ok);
        assert!(!still_mentions, "x survived the ±∞ rewrite");
    }

    #[test]
    fn test_eps_rewrite_deep_nesting_does_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut tm = TermManager::new();
                let f = deep_bool_formula(&mut tm, 6_250);
                let s = tm.mk_real(Rational64::new(0, 1));
                let x_spur = tm.intern_str("x");
                let out = eps_rewrite(f, x_spur, s, &mut tm);
                let still_mentions = out
                    .as_ref()
                    .ok()
                    .map(|&t| mentions_var(t, x_spur, &tm))
                    .unwrap_or(true);
                (out.is_ok(), still_mentions)
            })
            .expect("thread spawn should succeed");

        let (ok, still_mentions) = handle.join().expect("eps_rewrite must not overflow");
        assert!(ok);
        assert!(!still_mentions, "x survived the ε rewrite");
    }

    #[test]
    fn test_nested_xor_walks_are_linear() {
        // 30 nested `Xor`s: each level is reachable under two parents in the
        // rewrite, so an unmemoised walk would visit 2³⁰ nodes.
        let mut tm = TermManager::new();
        let (atom_x, atom_y, s) = deep_setup(&mut tm);
        let mut f = atom_x;
        for _ in 0..30 {
            f = tm.mk_xor(f, atom_y);
        }
        let x_spur = tm.intern_str("x");

        let start = oxiz_time::Instant::now();
        let mut atoms = Vec::new();
        collect_x_atoms(f, x_spur, &tm, &mut atoms).expect("collect must succeed");
        let neg_inf = inf_rewrite(f, x_spur, false, &mut tm).expect("inf rewrite must succeed");
        let eps = eps_rewrite(f, x_spur, s, &mut tm).expect("eps rewrite must succeed");
        let elapsed = start.elapsed();

        assert_eq!(atoms.len(), 1);
        assert!(!mentions_var(neg_inf, x_spur, &tm));
        assert!(!mentions_var(eps, x_spur, &tm));
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "nested-Xor walk took {elapsed:?}: the sharing memo regressed"
        );
    }

    #[test]
    fn test_inf_rewrite_atom_limits_are_exact() {
        let mut tm = TermManager::new();
        let real_sort = tm.sorts.real_sort;
        let x = tm.mk_var("x", real_sort);
        let five = tm.mk_real(Rational64::new(5, 1));
        let lt = tm.mk_lt(x, five);
        let gt = tm.mk_gt(x, five);
        let x_spur = tm.intern_str("x");
        let (t, f) = (tm.mk_true(), tm.mk_false());

        assert_eq!(inf_rewrite(lt, x_spur, false, &mut tm), Ok(t));
        assert_eq!(inf_rewrite(lt, x_spur, true, &mut tm), Ok(f));
        assert_eq!(inf_rewrite(gt, x_spur, false, &mut tm), Ok(f));
        assert_eq!(inf_rewrite(gt, x_spur, true, &mut tm), Ok(t));
    }

    #[test]
    fn test_rewrites_leave_x_free_subformulae_untouched() {
        let mut tm = TermManager::new();
        let real_sort = tm.sorts.real_sort;
        let y = tm.mk_var("y", real_sort);
        let zero = tm.mk_real(Rational64::new(0, 1));
        let atom_y = tm.mk_lt(y, zero);
        let x_spur = tm.intern_str("x");

        assert_eq!(inf_rewrite(atom_y, x_spur, true, &mut tm), Ok(atom_y));
        assert_eq!(eps_rewrite(atom_y, x_spur, zero, &mut tm), Ok(atom_y));
    }

    #[test]
    fn test_eps_rewrite_equality_cannot_hold_on_an_interval() {
        let mut tm = TermManager::new();
        let real_sort = tm.sorts.real_sort;
        let x = tm.mk_var("x", real_sort);
        let three = tm.mk_real(Rational64::new(3, 1));
        let eq = tm.mk_eq(x, three);
        let ne = tm.mk_not(eq);
        let x_spur = tm.intern_str("x");
        let (t, f) = (tm.mk_true(), tm.mk_false());

        assert_eq!(eps_rewrite(eq, x_spur, three, &mut tm), Ok(f));
        // ¬(x = 3) rewrites through `Not` of the same atom.
        assert_eq!(eps_rewrite(ne, x_spur, three, &mut tm), Ok(t));
    }

    #[test]
    fn test_lra_walks_deep_nesting_do_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut tm = TermManager::new();
                let int_sort = tm.sorts.int_sort;
                let x = tm.mk_var("x", int_sort);
                let one = tm.mk_int(1);
                let mut term = x;
                for _ in 0..7_500 {
                    term = tm.mk_add([term, one]);
                }
                let x_spur = tm.intern_str("x");

                let mentions = mentions_var(term, x_spur, &tm);
                let linear = to_linear(term, x_spur, &tm);
                (mentions, linear.map(|f| f.x_coeff))
            })
            .expect("thread spawn should succeed");

        let (mentions, x_coeff) = handle.join().expect("deep walks must not overflow");
        assert!(mentions);
        // `x + 1 + 1 + ...` keeps a unit coefficient on `x`.
        assert_eq!(x_coeff, Some(BigRational::one()));
    }
}
