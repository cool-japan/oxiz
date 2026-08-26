// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Linearisation: an arithmetic conjunction becomes a linear problem plus a
//! set of monics.
//!
//! Each product of two or more non-constant factors is named by a fresh
//! variable and recorded as a [`Monic`]. What survives is entirely linear, so
//! the existing Simplex core can solve it without knowing that multiplication
//! exists; the nonlinear content lives in the monic list, for the interval
//! propagator and the lemma constructors to spend.
//!
//! # Grammar, and what falls outside it
//!
//! Accepted: top-level `And`-nesting of `Eq` / `Le` / `Lt` / `Ge` / `Gt` over
//! `Add` / `Sub` / `Neg` / `Mul` / `IntConst` / integral `RealConst` / `Var`,
//! with every leaf of `Int` or `Real` sort.
//!
//! Everything else — `Or`, `Ite`, `Distinct`, `Apply`, `Div`, `Mod`,
//! non-integral `RealConst`, quantifiers, Boolean structure under a negation —
//! causes the *whole containing conjunct* to be dropped and
//! [`Linearization::incomplete`] to be set. Dropping a conjunct of a
//! conjunction only weakens the problem, so `unsat` on the result still
//! refutes the input; `sat` must be gated on `!incomplete`.
//!
//! Aux-var definitions emitted while translating a conjunct are kept even when
//! that conjunct is subsequently dropped. They are conservative: each defines
//! a *fresh* variable that occurs nowhere else, so they are a definitional
//! extension and cannot change satisfiability.

use super::super::simplex::{LinExpr, VarId};
use super::{checked_add_r64, checked_mul_r64, checked_neg_r64};
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::{One, ToPrimitive, Zero};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use smallvec::SmallVec;

/// A monomial named by a variable: `product = prod_i factors[i].0 ^ factors[i].1`.
///
/// `factors` is sorted by [`VarId`] and its exponents sum to at least two — a
/// degree-one "monomial" would be a plain variable and is never recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Monic {
    /// The fresh variable standing for the product.
    pub(crate) product: VarId,
    /// `(factor, exponent)` pairs, sorted by factor.
    pub(crate) factors: SmallVec<[(VarId, u32); 2]>,
}

impl Monic {
    /// Total degree: the sum of the exponents. Always `>= 2`.
    pub(crate) fn degree(&self) -> u32 {
        self.factors.iter().map(|(_, e)| *e).sum()
    }
}

/// The relation an atom asserts between its expression and zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LinAtomKind {
    /// `expr <= 0`
    Le,
    /// `expr >= 0`
    Ge,
    /// `expr == 0`
    Eq,
    /// `expr < 0`
    Lt,
    /// `expr > 0`
    Gt,
}

/// A linear atom, read as `expr ⋈ 0`.
#[derive(Debug, Clone)]
pub(crate) struct LinAtom {
    /// The left-hand side.
    pub(crate) expr: LinExpr,
    /// The relation to zero.
    pub(crate) kind: LinAtomKind,
}

/// The linear relaxation of an arithmetic conjunction.
#[derive(Debug, Clone, Default)]
pub(crate) struct Linearization {
    /// Linear atoms, implicitly conjoined. Includes aux-var definitions.
    pub(crate) atoms: Vec<LinAtom>,
    /// The nonlinear content that `atoms` deliberately forgets.
    pub(crate) monics: Vec<Monic>,
    /// Number of allocated variables; ids are `0 .. num_vars`.
    pub(crate) num_vars: u32,
    /// Source term for every variable that has one.
    pub(crate) term_of_var: FxHashMap<VarId, TermId>,
    /// Set of variables known to be integer-sorted.
    pub(crate) int_vars: FxHashSet<VarId>,
    /// At least one conjunct was outside the grammar and was dropped. A `sat`
    /// verdict derived from this relaxation is not sound for the input.
    pub(crate) incomplete: bool,
}

/// Linearise `assertions`, read as a conjunction.
///
/// Returns `None` when nothing arithmetic survived translation, which is the
/// caller's signal that this engine has no business with the goal.
pub(crate) fn linearize(assertions: &[TermId], manager: &TermManager) -> Option<Linearization> {
    let mut b = Builder::new(manager);

    // Flatten the top-level `And`-nesting iteratively: a deeply left-nested
    // conjunction is a real shape in SMT-LIB input and must not consume stack.
    let mut stack: Vec<TermId> = assertions.iter().rev().copied().collect();
    while let Some(id) = stack.pop() {
        let Some(node) = manager.get(id) else {
            b.incomplete = true;
            continue;
        };
        match &node.kind {
            TermKind::And(args) => stack.extend(args.iter().rev().copied()),
            TermKind::True => {}
            _ => {
                if b.conjunct(id).is_none() {
                    b.incomplete = true;
                }
            }
        }
    }

    if b.atoms.is_empty() {
        return None;
    }
    Some(Linearization {
        atoms: b.atoms,
        monics: b.monics,
        num_vars: b.next_var,
        term_of_var: b.term_of_var,
        int_vars: b.int_vars,
        incomplete: b.incomplete,
    })
}

// --- checked LinExpr algebra ------------------------------------------------

/// Merge `coef * var` into `e`, or `None` on overflow. Never leaves `e`
/// half-updated: the checked add happens before the write.
fn lin_add_term(e: &mut LinExpr, var: VarId, coef: Rational64) -> Option<()> {
    if coef.is_zero() {
        return Some(());
    }
    for (v, c) in &mut e.terms {
        if *v == var {
            *c = checked_add_r64(*c, coef)?;
            if c.is_zero() {
                e.terms.retain(|(w, _)| *w != var);
            }
            return Some(());
        }
    }
    e.terms.push((var, coef));
    Some(())
}

/// `a + b`, or `None` on overflow.
fn lin_add(a: &LinExpr, b: &LinExpr) -> Option<LinExpr> {
    let mut out = a.clone();
    for (v, c) in &b.terms {
        lin_add_term(&mut out, *v, *c)?;
    }
    out.constant = checked_add_r64(out.constant, b.constant)?;
    Some(out)
}

/// `-e`, or `None` on overflow.
fn lin_neg(e: &LinExpr) -> Option<LinExpr> {
    let mut out = LinExpr::new();
    for (v, c) in &e.terms {
        out.terms.push((*v, checked_neg_r64(*c)?));
    }
    out.constant = checked_neg_r64(e.constant)?;
    Some(out)
}

/// `k * e`, or `None` on overflow.
fn lin_scale(e: &LinExpr, k: Rational64) -> Option<LinExpr> {
    if k.is_zero() {
        return Some(LinExpr::new());
    }
    let mut out = LinExpr::new();
    for (v, c) in &e.terms {
        out.terms.push((*v, checked_mul_r64(*c, k)?));
    }
    out.constant = checked_mul_r64(e.constant, k)?;
    Some(out)
}

/// The single variable `e` denotes when it is exactly `1 * v`, else `None`.
fn as_bare_var(e: &LinExpr) -> Option<VarId> {
    if e.terms.len() == 1 && e.constant.is_zero() && e.terms[0].1.is_one() {
        Some(e.terms[0].0)
    } else {
        None
    }
}

// --- the walker -------------------------------------------------------------

/// One step of the explicit work stack that replaces native recursion.
enum Task {
    /// Translate this term.
    Visit(TermId),
    /// Its children are on the value stack; fold them.
    Combine(TermId),
}

struct Builder<'m> {
    manager: &'m TermManager,
    next_var: u32,
    /// Variable standing for a term: real `Var`s and aux-defined subterms.
    var_of_term: FxHashMap<TermId, VarId>,
    term_of_var: FxHashMap<VarId, TermId>,
    int_vars: FxHashSet<VarId>,
    /// Product variable for an already-seen sorted factor key.
    monic_of_key: FxHashMap<Vec<(VarId, u32)>, VarId>,
    /// Index into `monics` for each product variable, for factor splicing.
    monic_of_var: FxHashMap<VarId, usize>,
    monics: Vec<Monic>,
    atoms: Vec<LinAtom>,
    incomplete: bool,
}

impl<'m> Builder<'m> {
    fn new(manager: &'m TermManager) -> Self {
        Self {
            manager,
            next_var: 0,
            var_of_term: FxHashMap::default(),
            term_of_var: FxHashMap::default(),
            int_vars: FxHashSet::default(),
            monic_of_key: FxHashMap::default(),
            monic_of_var: FxHashMap::default(),
            monics: Vec::new(),
            atoms: Vec::new(),
            incomplete: false,
        }
    }

    /// `Some(true)` for `Int`, `Some(false)` for `Real`, `None` otherwise.
    fn arith_is_int(&self, id: TermId) -> Option<bool> {
        let sort = self.manager.get(id)?.sort;
        if sort == self.manager.sorts.int_sort {
            Some(true)
        } else if sort == self.manager.sorts.real_sort {
            Some(false)
        } else {
            None
        }
    }

    fn fresh(&mut self, term: Option<TermId>, is_int: bool) -> VarId {
        let v = self.next_var;
        self.next_var += 1;
        if is_int {
            self.int_vars.insert(v);
        }
        if let Some(t) = term {
            self.term_of_var.entry(v).or_insert(t);
        }
        v
    }

    /// The variable for a term that must be represented atomically.
    fn var_for(&mut self, id: TermId, is_int: bool) -> VarId {
        if let Some(v) = self.var_of_term.get(&id) {
            return *v;
        }
        let v = self.fresh(Some(id), is_int);
        self.var_of_term.insert(id, v);
        v
    }

    /// Translate one conjunct into an atom. `None` means "outside the
    /// grammar, or unrepresentable": the caller drops it.
    fn conjunct(&mut self, id: TermId) -> Option<()> {
        let kind = self.manager.get(id)?.kind.clone();
        let (lhs, rhs, rel) = match &kind {
            TermKind::Eq(a, b) => (*a, *b, LinAtomKind::Eq),
            TermKind::Le(a, b) => (*a, *b, LinAtomKind::Le),
            TermKind::Lt(a, b) => (*a, *b, LinAtomKind::Lt),
            TermKind::Ge(a, b) => (*a, *b, LinAtomKind::Ge),
            TermKind::Gt(a, b) => (*a, *b, LinAtomKind::Gt),
            _ => return None,
        };
        // `Eq` is also Boolean/array/BV equality; only arithmetic operands are
        // ours to translate.
        let int_lhs = self.arith_is_int(lhs)?;
        let int_rhs = self.arith_is_int(rhs)?;
        let le = self.translate(lhs)?;
        let re = self.translate(rhs)?;
        let expr = lin_add(&le, &lin_neg(&re)?)?;
        let atom = self.tighten_strict(LinAtom { expr, kind: rel }, int_lhs && int_rhs);
        self.atoms.push(atom);
        Some(())
    }

    /// Turn `expr < 0` over integer-valued data into `expr' + 1 <= 0` (and
    /// dually for `>`), which is a strictly stronger *equivalent* over the
    /// integers and something Simplex can enforce without a delta.
    ///
    /// Applies only when every variable is integer-sorted; the expression is
    /// first scaled by the lcm of all denominators so the `±1` step is valid.
    /// If any step overflows we keep the strict atom: weaker, still sound.
    fn tighten_strict(&self, atom: LinAtom, all_int_operands: bool) -> LinAtom {
        if !matches!(atom.kind, LinAtomKind::Lt | LinAtomKind::Gt) || !all_int_operands {
            return atom;
        }
        if !atom
            .expr
            .terms
            .iter()
            .all(|(v, _)| self.int_vars.contains(v))
        {
            return atom;
        }
        let Some(scaled) = self.clear_denominators(&atom.expr) else {
            return atom;
        };
        let mut out = scaled;
        let step = Rational64::one();
        match atom.kind {
            LinAtomKind::Lt => match checked_add_r64(out.constant, step) {
                Some(c) => {
                    out.constant = c;
                    LinAtom {
                        expr: out,
                        kind: LinAtomKind::Le,
                    }
                }
                None => atom,
            },
            LinAtomKind::Gt => match checked_add_r64(out.constant, -step) {
                Some(c) => {
                    out.constant = c;
                    LinAtom {
                        expr: out,
                        kind: LinAtomKind::Ge,
                    }
                }
                None => atom,
            },
            _ => atom,
        }
    }

    /// Scale by the lcm of all denominators so every coefficient and the
    /// constant become integers. `None` on overflow.
    fn clear_denominators(&self, e: &LinExpr) -> Option<LinExpr> {
        let mut lcm: i64 = 1;
        for d in e
            .terms
            .iter()
            .map(|(_, c)| *c.denom())
            .chain(core::iter::once(*e.constant.denom()))
        {
            let g = gcd_i64(lcm, d);
            if g == 0 {
                return None;
            }
            lcm = lcm.checked_mul(d / g)?;
        }
        if lcm <= 0 {
            return None;
        }
        lin_scale(e, Rational64::from_integer(lcm))
    }

    /// Translate an arithmetic term to a linear expression, allocating aux
    /// variables and monics as needed. Iterative: no native recursion.
    fn translate(&mut self, root: TermId) -> Option<LinExpr> {
        let mut tasks = vec![Task::Visit(root)];
        let mut values: Vec<LinExpr> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Visit(id) => {
                    let kind = self.manager.get(id)?.kind.clone();
                    match &kind {
                        TermKind::IntConst(n) => {
                            let v = n.to_i64()?;
                            values.push(LinExpr::constant(Rational64::from_integer(v)));
                        }
                        TermKind::RealConst(r) => {
                            // A non-integral rational constant is outside the
                            // declared grammar; drop the conjunct.
                            if !r.is_integer() {
                                return None;
                            }
                            values.push(LinExpr::constant(*r));
                        }
                        TermKind::Var(_) => {
                            let is_int = self.arith_is_int(id)?;
                            let v = self.var_for(id, is_int);
                            values.push(LinExpr::var(v));
                        }
                        TermKind::Neg(a) => {
                            tasks.push(Task::Combine(id));
                            tasks.push(Task::Visit(*a));
                        }
                        TermKind::Sub(a, b) => {
                            tasks.push(Task::Combine(id));
                            tasks.push(Task::Visit(*b));
                            tasks.push(Task::Visit(*a));
                        }
                        TermKind::Add(args) | TermKind::Mul(args) => {
                            if args.is_empty() {
                                return None;
                            }
                            tasks.push(Task::Combine(id));
                            for a in args.iter().rev() {
                                tasks.push(Task::Visit(*a));
                            }
                        }
                        _ => return None,
                    }
                }
                Task::Combine(id) => {
                    let kind = self.manager.get(id)?.kind.clone();
                    match &kind {
                        TermKind::Neg(_) => {
                            let a = values.pop()?;
                            values.push(lin_neg(&a)?);
                        }
                        TermKind::Sub(_, _) => {
                            let b = values.pop()?;
                            let a = values.pop()?;
                            values.push(lin_add(&a, &lin_neg(&b)?)?);
                        }
                        TermKind::Add(args) => {
                            let at = values.len().checked_sub(args.len())?;
                            let kids: Vec<LinExpr> = values.split_off(at);
                            let mut acc = LinExpr::new();
                            for k in &kids {
                                acc = lin_add(&acc, k)?;
                            }
                            values.push(acc);
                        }
                        TermKind::Mul(args) => {
                            let at = values.len().checked_sub(args.len())?;
                            let kids: Vec<LinExpr> = values.split_off(at);
                            let is_int = self.arith_is_int(id)?;
                            values.push(self.combine_mul(id, args, &kids, is_int)?);
                        }
                        _ => return None,
                    }
                }
            }
        }
        values.pop()
    }

    /// Fold a `Mul` node: split off the constant part, then name the product
    /// of what remains.
    fn combine_mul(
        &mut self,
        id: TermId,
        args: &[TermId],
        kids: &[LinExpr],
        is_int: bool,
    ) -> Option<LinExpr> {
        let mut coef = Rational64::one();
        // Multiplicity by *source term*, so repeated occurrences of the same
        // subterm collapse into an exponent rather than distinct factors.
        let mut nonconst: Vec<(TermId, &LinExpr)> = Vec::new();
        for (a, k) in args.iter().zip(kids.iter()) {
            if k.terms.is_empty() {
                coef = checked_mul_r64(coef, k.constant)?;
            } else {
                nonconst.push((*a, k));
            }
        }
        if coef.is_zero() || nonconst.is_empty() {
            return Some(LinExpr::constant(coef));
        }
        if nonconst.len() == 1 {
            return lin_scale(nonconst[0].1, coef);
        }

        // Each non-constant factor must be an atomic variable; anything with
        // real linear structure gets a fresh aux variable and a defining
        // equation `expr - aux = 0`.
        let mut exps: Vec<(VarId, u32)> = Vec::new();
        for (term, expr) in &nonconst {
            let base = match as_bare_var(expr) {
                Some(v) => v,
                None => self.define_aux(*term, expr)?,
            };
            // Splice a nested product's factors in, so `(* (* x y) z)` becomes
            // the single monic `x*y*z` rather than a monic over a monic.
            let spliced: SmallVec<[(VarId, u32); 2]> = match self.monic_of_var.get(&base) {
                Some(i) => self.monics.get(*i)?.factors.clone(),
                None => smallvec::smallvec![(base, 1)],
            };
            for (v, e) in spliced {
                match exps.iter_mut().find(|(w, _)| *w == v) {
                    Some(slot) => slot.1 = slot.1.checked_add(e)?,
                    None => exps.push((v, e)),
                }
            }
        }
        exps.sort_unstable_by_key(|(v, _)| *v);

        let existing = self.monic_of_key.get(&exps).copied();
        let product = match existing {
            Some(v) => v,
            None => {
                // The product variable stands for the *bare* monomial, while
                // `id` denotes `coef * monomial`. Claiming `term_of_var[p] = id`
                // when `coef != 1` would be off by that factor, and since
                // `(* x y)` and `(* 2 x y)` dedupe to the same key it would also
                // be ambiguous. Record the correspondence only when it is exact.
                let term = if coef.is_one() { Some(id) } else { None };
                let p = self.fresh(term, is_int);
                self.monic_of_key.insert(exps.clone(), p);
                self.monic_of_var.insert(p, self.monics.len());
                self.monics.push(Monic {
                    product: p,
                    factors: exps.iter().copied().collect(),
                });
                p
            }
        };
        lin_scale(&LinExpr::var(product), coef)
    }

    /// Name a compound factor with a fresh variable and commit its definition.
    fn define_aux(&mut self, term: TermId, expr: &LinExpr) -> Option<VarId> {
        if let Some(v) = self.var_of_term.get(&term) {
            return Some(*v);
        }
        let is_int = self.arith_is_int(term)?;
        let v = self.fresh(Some(term), is_int);
        self.var_of_term.insert(term, v);
        let mut def = expr.clone();
        lin_add_term(&mut def, v, -Rational64::one())?;
        self.atoms.push(LinAtom {
            expr: def,
            kind: LinAtomKind::Eq,
        });
        Some(v)
    }
}

/// GCD of two `i64`s, for denominator clearing.
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[cfg(test)]
mod tests;
