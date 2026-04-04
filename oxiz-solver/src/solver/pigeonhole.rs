//! Pigeonhole principle encoding and integer domain clause generation.
//!
//! This module provides SAT-level encodings for:
//! - Pigeonhole exclusion clauses for integer-domain terms
//! - Select equality splits for array theory reasoning
//! - Integer domain enumeration clauses for bounded variables

use crate::prelude::*;
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use oxiz_core::ast::{TermId, TermKind, TermManager};

use super::Solver;

impl Solver {
    /// Add pigeonhole exclusion clauses from pre-collected domains and disequalities.
    pub(super) fn add_pigeonhole_exclusions_from(
        &mut self,
        domains: &FxHashMap<TermId, (i64, i64)>,
        diseq_pairs: &[(TermId, TermId)],
        manager: &mut TermManager,
    ) {
        for &(x, y) in diseq_pairs {
            let x_domain = domains.get(&x).copied();
            let y_domain = domains.get(&y).copied();
            if let (Some((x_lo, x_hi)), Some((y_lo, y_hi))) = (x_domain, y_domain) {
                let lo = x_lo.max(y_lo);
                let hi = x_hi.min(y_hi);
                if hi >= lo && (hi - lo) <= 20 {
                    for v in lo..=hi {
                        let val = manager.mk_int(BigInt::from(v));
                        let eq_x = manager.mk_eq(x, val);
                        let eq_y = manager.mk_eq(y, val);
                        let lit_x = self.encode(eq_x, manager);
                        let lit_y = self.encode(eq_y, manager);
                        // At most one of x and y can equal k
                        let _ = self.sat.add_clause([lit_x.negate(), lit_y.negate()]);
                    }
                }
            }
        }
    }

    /// Add pigeonhole exclusion clauses for integer-domain terms.
    ///
    /// For every pair of terms (x, y) where we have an active disequality
    /// `not(= x y)` and both have bounded integer domains [L, U],
    /// add `Not(Eq(x, k)) OR Not(Eq(y, k))` for each value k in the domain.
    /// This SAT-level encoding directly captures the pigeonhole principle.
    pub(super) fn add_pigeonhole_exclusions(&mut self, manager: &mut TermManager) {
        // Collect domain information: term -> (lo, hi)
        let mut domains: FxHashMap<TermId, (i64, i64)> = FxHashMap::default();
        // Collect disequality pairs
        let mut diseq_pairs: Vec<(TermId, TermId)> = Vec::new();

        // Scan all encoded terms for domain bounds and disequalities
        for &tid in self.arith_terms.iter() {
            // Already tracked -- skip
            let _ = tid;
        }

        // Scan assertions for the patterns we need
        for &aterm in &self.assertions {
            self.scan_for_pigeonhole(aterm, manager, &mut domains, &mut diseq_pairs);
        }

        // Also scan SAT clause implications -- check unit-propagated terms
        // by scanning the term->var mapping for known domain/diseq patterns
        for (&tid, _) in self.term_to_var.iter() {
            self.scan_for_pigeonhole(tid, manager, &mut domains, &mut diseq_pairs);
        }

        // For each disequality pair where both have domains, add exclusion
        for &(x, y) in &diseq_pairs {
            let x_domain = domains.get(&x).copied();
            let y_domain = domains.get(&y).copied();
            if let (Some((x_lo, x_hi)), Some((y_lo, y_hi))) = (x_domain, y_domain) {
                let lo = x_lo.max(y_lo);
                let hi = x_hi.min(y_hi);
                if hi >= lo && (hi - lo) <= 20 {
                    for v in lo..=hi {
                        let val = manager.mk_int(BigInt::from(v));
                        let eq_x = manager.mk_eq(x, val);
                        let eq_y = manager.mk_eq(y, val);
                        let lit_x = self.encode(eq_x, manager);
                        let lit_y = self.encode(eq_y, manager);
                        // Not(Eq(x, k)) OR Not(Eq(y, k))
                        let _ = self.sat.add_clause([lit_x.negate(), lit_y.negate()]);
                    }
                }
            }
        }
    }

    pub(super) fn scan_for_pigeonhole(
        &self,
        term: TermId,
        manager: &TermManager,
        domains: &mut FxHashMap<TermId, (i64, i64)>,
        diseq_pairs: &mut Vec<(TermId, TermId)>,
    ) {
        let Some(t) = manager.get(term) else { return };
        match &t.kind {
            // Recurse into Implies -- scan both guard and consequent
            TermKind::Implies(_guard, consequent) => {
                // The consequent typically has the constraint after guard filtering.
                // Scan it for disequalities and domain bounds.
                self.scan_for_pigeonhole(*consequent, manager, domains, diseq_pairs);
            }
            // And(Ge(x, L), Le(x, U)) -> domain for x
            // Also recurse into And elements for nested patterns
            TermKind::And(args) => {
                let mut lower: Option<(TermId, i64)> = None;
                let mut upper: Option<(TermId, i64)> = None;
                for &a in args.iter() {
                    if let Some(at) = manager.get(a) {
                        match &at.kind {
                            TermKind::Ge(lhs, rhs) => {
                                if let Some(rt) = manager.get(*rhs) {
                                    if let TermKind::IntConst(n) = &rt.kind {
                                        if let Some(v) = n.to_i64() {
                                            lower = Some((*lhs, v));
                                        }
                                    }
                                }
                                // Also check Ge(IntConst, x) -> upper bound
                                if let Some(lt) = manager.get(*lhs) {
                                    if let TermKind::IntConst(n) = &lt.kind {
                                        if let Some(v) = n.to_i64() {
                                            upper = Some((*rhs, v));
                                        }
                                    }
                                }
                            }
                            TermKind::Le(lhs, rhs) => {
                                if let Some(rt) = manager.get(*rhs) {
                                    if let TermKind::IntConst(n) = &rt.kind {
                                        if let Some(v) = n.to_i64() {
                                            upper = Some((*lhs, v));
                                        }
                                    }
                                }
                                // Also check Le(IntConst, x) -> lower bound
                                if let Some(lt) = manager.get(*lhs) {
                                    if let TermKind::IntConst(n) = &lt.kind {
                                        if let Some(v) = n.to_i64() {
                                            lower = Some((*rhs, v));
                                        }
                                    }
                                }
                            }
                            _ => {
                                // Recurse into sub-elements
                                self.scan_for_pigeonhole(a, manager, domains, diseq_pairs);
                            }
                        }
                    }
                }
                if let (Some((lx, lo)), Some((ux, hi))) = (lower, upper) {
                    if lx == ux {
                        domains.insert(lx, (lo, hi));
                    }
                }
            }
            // Not(Eq(x, y)) -> disequality pair
            TermKind::Not(inner) => {
                if let Some(it) = manager.get(*inner) {
                    if let TermKind::Eq(lhs, rhs) = &it.kind {
                        diseq_pairs.push((*lhs, *rhs));
                    }
                }
            }
            _ => {}
        }
    }

    /// Add explicit pairwise equality decisions for all select terms
    /// tracked by the arithmetic solver.  For each pair of select terms
    /// `select(a, i)` and `select(a, j)` with the same array, add the
    /// tautological clause `Eq(s_i, s_j) OR Not(Eq(s_i, s_j))`.  This
    /// forces the SAT solver to decide the equality, enabling theory
    /// propagation for pigeonhole-style contradictions.
    pub(super) fn add_select_equality_splits(&mut self, manager: &mut TermManager) {
        // Collect all select terms from the arith terms set
        let select_terms: Vec<(TermId, TermId, TermId)> = self
            .arith_terms
            .iter()
            .filter_map(|&tid| {
                let t = manager.get(tid)?;
                if let TermKind::Select(array, index) = &t.kind {
                    Some((tid, *array, *index))
                } else {
                    None
                }
            })
            .collect();

        // For each pair of selects on the same array, add equality split
        for i in 0..select_terms.len() {
            for j in (i + 1)..select_terms.len() {
                let (s_i, arr_i, _) = select_terms[i];
                let (s_j, arr_j, _) = select_terms[j];
                if arr_i != arr_j {
                    continue;
                }
                // Add: Eq(s_i, s_j) OR Not(Eq(s_i, s_j))
                // This is a tautology, but it forces the SAT solver to
                // assign a truth value to Eq(s_i, s_j), enabling the
                // theory solver to detect conflicts.
                let eq = manager.mk_eq(s_i, s_j);
                let eq_lit = self.encode(eq, manager);
                // The tautological clause is always satisfied, but the
                // important side effect is that Eq(s_i, s_j) now has a
                // SAT variable. The SAT solver must decide it.
                let _ = self.sat.add_clause([eq_lit, eq_lit.negate()]);

                // Also add the disequality split: if they're unequal,
                // they must be ordered.
                let lt = manager.mk_lt(s_i, s_j);
                let gt = manager.mk_gt(s_i, s_j);
                let lt_lit = self.encode(lt, manager);
                let gt_lit = self.encode(gt, manager);
                let neq_lit = eq_lit.negate();
                // Not(Eq(s_i, s_j)) => Lt(s_i, s_j) OR Gt(s_i, s_j)
                let _ = self.sat.add_clause([eq_lit, lt_lit, gt_lit]);
                let _ = neq_lit;
            }
        }
    }

    /// For a conjunction `And(Ge(x, L), Le(x, U))` on integer terms,
    /// add the clause `Eq(x, L) OR Eq(x, L+1) OR ... OR Eq(x, U)`.
    ///
    /// This forces the SAT solver to pick a concrete integer value for x,
    /// which is required for pigeonhole reasoning (the simplex over rationals
    /// cannot detect integer pigeonhole violations).
    pub(super) fn add_int_domain_clauses(&mut self, term: TermId, manager: &mut TermManager) {
        let Some(t) = manager.get(term).cloned() else {
            return;
        };
        if let TermKind::And(args) = &t.kind {
            // Look for Ge(x, IntConst(L)) / Le(IntConst(L), x) and
            //          Le(x, IntConst(U)) / Ge(IntConst(U), x) pairs.
            // deep_simplify may convert Ge(a,b) -> Le(b,a), so both forms
            // must be recognized.
            let mut lower: Option<(TermId, i64)> = None;
            let mut upper: Option<(TermId, i64)> = None;
            for &a in args.iter() {
                if let Some(at) = manager.get(a).cloned() {
                    match &at.kind {
                        // Ge(x, IntConst(L)) -> lower bound L for x
                        TermKind::Ge(lhs, rhs) => {
                            if let Some(rt) = manager.get(*rhs) {
                                if let TermKind::IntConst(n) = &rt.kind {
                                    if let Some(v) = n.to_i64() {
                                        lower = Some((*lhs, v));
                                    }
                                }
                            }
                            // Ge(IntConst(U), x) -> upper bound U for x
                            if let Some(lt) = manager.get(*lhs) {
                                if let TermKind::IntConst(n) = &lt.kind {
                                    if let Some(v) = n.to_i64() {
                                        upper = Some((*rhs, v));
                                    }
                                }
                            }
                        }
                        TermKind::Le(lhs, rhs) => {
                            // Le(x, IntConst(U)) -> upper bound U for x
                            if let Some(rt) = manager.get(*rhs) {
                                if let TermKind::IntConst(n) = &rt.kind {
                                    if let Some(v) = n.to_i64() {
                                        upper = Some((*lhs, v));
                                    }
                                }
                            }
                            // Le(IntConst(L), x) -> lower bound L for x
                            if let Some(lt) = manager.get(*lhs) {
                                if let TermKind::IntConst(n) = &lt.kind {
                                    if let Some(v) = n.to_i64() {
                                        lower = Some((*rhs, v));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            if let (Some((lx, lo)), Some((ux, hi))) = (lower, upper) {
                if lx == ux && hi >= lo && (hi - lo) <= 10 {
                    // Add: Eq(x, lo) OR Eq(x, lo+1) OR ... OR Eq(x, hi)
                    let mut domain_lits = Vec::new();
                    for v in lo..=hi {
                        let val = manager.mk_int(BigInt::from(v));
                        let eq = manager.mk_eq(lx, val);
                        let lit = self.encode(eq, manager);
                        domain_lits.push(lit);
                    }
                    if !domain_lits.is_empty() {
                        self.sat.add_clause(domain_lits);
                    }
                }
            }
        }
    }
}
