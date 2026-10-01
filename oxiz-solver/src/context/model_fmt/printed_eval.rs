//! The query-free half of the published-model certificate (re-fix pass 15,
//! decisions (67), (68)): the goal, closed by the printed model, decided by
//! exact evaluation.
//!
//! Once every declared constant is replaced by its printed value and every
//! uninterpreted function by its printed table (`printed_check`), the only
//! symbols an assertion still has are bound variables.  The certificate's
//! queries (`solver::array_completion_certify::printed`) decide most goals;
//! this module decides the ones they leave open — a vacuous binder over a
//! datatype, an `exists` whose witness no named point gives, a negated
//! universal, a nonlinear body over a bounded integer box, a small
//! bit-vector domain — by evaluation alone, three-valued:
//!
//! * every maximal closed sub-term (no bound variable in it) is folded to its
//!   value first, so a body's array reads and table lookups are literals;
//! * a quantifier over `Bool`, a bit-vector sort of at most 2^12 values, or
//!   an `Int` a universal's guard bounds to a finite box is enumerated
//!   exhaustively — decided `true` or `false`;
//! * an `exists` over `Int` / `Real` is witnessed at the body's own literals
//!   and their neighbours — decided `true` when one holds, open otherwise;
//! * a universal over an unbounded domain is decided `true` only by
//!   `mbqi::model_certify`'s region enumeration, which checks its own
//!   fragment first, and is open otherwise;
//! * a variable the body does not mention drops out (every sort is
//!   non-empty).
//!
//! Only `Some(true)` for every assertion certifies; `None` never does.
//! Sound by construction: every answer is an evaluation of the printed model.

use super::*;
use crate::prelude::{FxHashMap, FxHashSet};

mod datatype_reads;
mod ground_values;

/// Evaluations one certification may spend.
const MAX_DECIDE_STEPS: usize = 20_000;

/// Points one quantifier may enumerate.
const MAX_ENUMERATED_POINTS: usize = 4_096;

/// Widest bit-vector sort a quantifier is enumerated over.
const MAX_ENUMERATED_BV_WIDTH: u32 = 12;

/// `Real` literals beyond which a witness search takes no pairwise
/// midpoints.
const MAX_REAL_WITNESS_LITERALS: usize = 12;

/// One bound variable's domain: its points, and whether they are every value
/// the variable can take where the body can hold.
struct Domain {
    points: Vec<TermId>,
    exhaustive: bool,
}

impl Context {
    /// The query-free certificate: `true` when every assertion of `goal`,
    /// closed by `printed`, evaluates to `true` (see the module docs).
    pub(in crate::context) fn certified_by_evaluation(
        &mut self,
        goal: &[TermId],
        printed: &FxHashMap<TermId, TermId>,
    ) -> bool {
        // A selector or tester over a substituted constructor value folds
        // (`TermManager::fold_constructor_accessors`, decision (72)(d)).
        let closed: Vec<TermId> = goal
            .iter()
            .map(|&assertion| {
                let closed = self.terms.substitute(assertion, printed);
                self.terms.fold_constructor_accessors(closed)
            })
            .collect();
        if closed.iter().any(|&assertion| {
            !self
                .terms
                .free_vars_including_patterns(assertion)
                .is_empty()
        }) {
            return false;
        }
        if crate::mbqi::model_certify::certify(&closed, &FxHashMap::default(), &self.terms) {
            return true;
        }
        let mut budget = MAX_DECIDE_STEPS;
        closed
            .iter()
            .all(|&assertion| self.decide_closed(assertion, &mut budget) == Some(true))
    }

    /// One assertion of the goal, closed by `printed`, shown true by exact
    /// evaluation — the per-assertion form of
    /// [`Self::certified_by_evaluation`].
    pub(in crate::context) fn assertion_certified_by_evaluation(
        &mut self,
        assertion: TermId,
        printed: &FxHashMap<TermId, TermId>,
    ) -> bool {
        let closed = self.terms.substitute(assertion, printed);
        let closed = self.terms.fold_constructor_accessors(closed);
        if !self.terms.free_vars_including_patterns(closed).is_empty() {
            return false;
        }
        let mut budget = MAX_DECIDE_STEPS;
        self.decide_closed(closed, &mut budget) == Some(true)
            || crate::mbqi::model_certify::certify(&[closed], &FxHashMap::default(), &self.terms)
    }

    /// The truth value of a closed Boolean term, where evaluation decides it.
    fn decide_closed(&mut self, term: TermId, budget: &mut usize) -> Option<bool> {
        if *budget == 0 {
            return None;
        }
        *budget -= 1;
        let term = self.terms.fold_constructor_accessors(term);
        let term = self.fold_datatype_reads(term);
        let term = self.fold_closed_subterms(term);
        if !self.has_quantifier(term) {
            return self.evaluate_bool(term);
        }
        let kind = self.terms.get(term)?.kind.clone();
        match kind {
            TermKind::Forall { vars, body, .. } => {
                self.decide_quantifier(true, &vars, body, budget)
            }
            TermKind::Exists { vars, body, .. } => {
                self.decide_quantifier(false, &vars, body, budget)
            }
            TermKind::Not(inner) => self.decide_closed(inner, budget).map(|v| !v),
            TermKind::And(args) => {
                let mut open = false;
                for arg in args {
                    match self.decide_closed(arg, budget) {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => open = true,
                    }
                }
                (!open).then_some(true)
            }
            TermKind::Or(args) => {
                let mut open = false;
                for arg in args {
                    match self.decide_closed(arg, budget) {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => open = true,
                    }
                }
                (!open).then_some(false)
            }
            TermKind::Implies(premise, conclusion) => match self.decide_closed(premise, budget) {
                Some(false) => Some(true),
                Some(true) => self.decide_closed(conclusion, budget),
                None => (self.decide_closed(conclusion, budget) == Some(true)).then_some(true),
            },
            TermKind::Ite(condition, then_branch, else_branch) => {
                match self.decide_closed(condition, budget) {
                    Some(true) => self.decide_closed(then_branch, budget),
                    Some(false) => self.decide_closed(else_branch, budget),
                    None => {
                        let a = self.decide_closed(then_branch, budget)?;
                        let b = self.decide_closed(else_branch, budget)?;
                        (a == b).then_some(a)
                    }
                }
            }
            TermKind::Eq(left, right) if self.is_bool(left) => {
                let a = self.decide_closed(left, budget)?;
                let b = self.decide_closed(right, budget)?;
                Some(a == b)
            }
            TermKind::Xor(left, right) => {
                let a = self.decide_closed(left, budget)?;
                let b = self.decide_closed(right, budget)?;
                Some(a != b)
            }
            _ => None,
        }
    }

    /// `∀vars. body` (`universal`) or `∃vars. body`, decided by enumerating
    /// the variables' domains (see the module docs).
    fn decide_quantifier(
        &mut self,
        universal: bool,
        vars: &[(oxiz_core::interner::Spur, SortId)],
        body: TermId,
        budget: &mut usize,
    ) -> Option<bool> {
        let mut bound: Vec<TermId> = Vec::with_capacity(vars.len());
        let mut domains: Vec<Domain> = Vec::with_capacity(vars.len());
        let free = self.terms.free_vars_including_patterns(body);
        for &(name, sort) in vars {
            let text = self.terms.resolve_str(name).to_string();
            let var = self.terms.mk_var(&text, sort);
            if !free.contains(&var) {
                // Not mentioned: every sort is non-empty, so it drops out.
                continue;
            }
            let domain = self.domain_of(var, sort, body, universal)?;
            bound.push(var);
            domains.push(domain);
        }
        if bound.is_empty() {
            return self.decide_closed(body, budget);
        }
        if domains.iter().any(|d| d.exhaustive && d.points.is_empty()) {
            // A guard box with no integer in it: the universal is vacuous.
            return Some(universal);
        }
        let mut total: usize = 1;
        for domain in &domains {
            if domain.points.is_empty() {
                return None;
            }
            total = total
                .checked_mul(domain.points.len())
                .filter(|&t| t <= MAX_ENUMERATED_POINTS)?;
        }
        let exhaustive = domains.iter().all(|d| d.exhaustive);
        let mut open = false;
        let mut position: Vec<usize> = vec![0; domains.len()];
        for _ in 0..total {
            let map: FxHashMap<TermId, TermId> = bound
                .iter()
                .zip(&position)
                .zip(&domains)
                .map(|((&var, &i), domain)| (var, domain.points[i]))
                .collect();
            let instance = self.terms.substitute(body, &map);
            match (universal, self.decide_closed(instance, budget)) {
                (true, Some(false)) => return Some(false),
                (false, Some(true)) => return Some(true),
                (_, None) => open = true,
                _ => {}
            }
            for (slot, domain) in position.iter_mut().zip(&domains).rev() {
                *slot += 1;
                if *slot < domain.points.len() {
                    break;
                }
                *slot = 0;
            }
        }
        if exhaustive && !open {
            return Some(universal);
        }
        if universal {
            // An unbounded universal: only the region argument decides it.
            let quantifier = self.rebuild_forall(vars, body);
            return crate::mbqi::model_certify::certify(
                &[quantifier],
                &FxHashMap::default(),
                &self.terms,
            )
            .then_some(true);
        }
        None
    }

    /// The domain a bound variable is enumerated over, or `None` when no
    /// enumeration can speak for it.
    fn domain_of(
        &mut self,
        var: TermId,
        sort: SortId,
        body: TermId,
        universal: bool,
    ) -> Option<Domain> {
        match self.terms.sorts.get(sort).map(|s| s.kind.clone())? {
            SortKind::Bool => Some(Domain {
                points: vec![self.terms.mk_false(), self.terms.mk_true()],
                exhaustive: true,
            }),
            SortKind::BitVec(width) if width <= MAX_ENUMERATED_BV_WIDTH => Some(Domain {
                points: (0u32..(1u32 << width))
                    .map(|v| self.terms.mk_bitvec(v, width))
                    .collect(),
                exhaustive: true,
            }),
            SortKind::Int => {
                if universal
                    && let Some(TermKind::Implies(guard, _)) =
                        self.terms.get(body).map(|t| t.kind.clone())
                    && let Some((lo, hi)) = int_bounds_of(guard, var, &self.terms)
                {
                    if hi < lo {
                        return Some(Domain {
                            points: Vec::new(),
                            exhaustive: true,
                        });
                    }
                    let width = usize::try_from(&hi - &lo + 1u8).ok()?;
                    if width > MAX_ENUMERATED_POINTS {
                        return None;
                    }
                    let mut points = Vec::with_capacity(width);
                    let mut value = lo;
                    while value <= hi {
                        points.push(self.terms.mk_int(value.clone()));
                        value += 1u8;
                    }
                    return Some(Domain {
                        points,
                        exhaustive: true,
                    });
                }
                // The body's literals, their sums and differences (a boundary
                // `c₁ = x + c₂` sits at `c₁ - c₂`), and the neighbours of each.
                let mut literals: Vec<BigInt> = vec![BigInt::zero()];
                for literal in self.literals_of(body) {
                    if let Some(TermKind::IntConst(n)) =
                        self.terms.get(literal).map(|t| t.kind.clone())
                    {
                        literals.push(n);
                    }
                }
                literals.sort();
                literals.dedup();
                let mut centres: Vec<BigInt> = literals.clone();
                if literals.len() <= MAX_REAL_WITNESS_LITERALS {
                    for a in &literals {
                        for b in &literals {
                            centres.push(a - b);
                            centres.push(a + b);
                        }
                    }
                }
                let mut values: Vec<BigInt> = Vec::with_capacity(centres.len() * 3);
                for c in centres {
                    values.extend([&c - 1u8, c.clone(), c + 1u8]);
                }
                values.sort();
                values.dedup();
                Some(Domain {
                    points: values.into_iter().map(|v| self.terms.mk_int(v)).collect(),
                    exhaustive: false,
                })
            }
            SortKind::Real => {
                let one = num_rational::Rational64::from_integer(1);
                let mut literals: Vec<num_rational::Rational64> = self
                    .literals_of(body)
                    .into_iter()
                    .filter_map(|t| match self.terms.get(t).map(|d| &d.kind) {
                        Some(TermKind::RealConst(r)) => Some(*r),
                        Some(TermKind::IntConst(n)) => i64::try_from(n.clone())
                            .ok()
                            .map(num_rational::Rational64::from_integer),
                        _ => None,
                    })
                    .collect();
                literals.sort();
                literals.dedup();
                let mut values: Vec<num_rational::Rational64> =
                    vec![num_rational::Rational64::from_integer(0)];
                for &r in &literals {
                    values.push(r);
                    values.extend(num_traits::CheckedSub::checked_sub(&r, &one));
                    values.extend(num_traits::CheckedAdd::checked_add(&r, &one));
                }
                if literals.len() <= MAX_REAL_WITNESS_LITERALS {
                    for window in literals.windows(2) {
                        if let [a, b] = window {
                            values.extend(
                                num_traits::CheckedAdd::checked_add(a, b)
                                    .map(|s| s / num_rational::Rational64::from_integer(2)),
                            );
                        }
                    }
                }
                values.sort();
                values.dedup();
                Some(Domain {
                    points: values.into_iter().map(|v| self.terms.mk_real(v)).collect(),
                    exhaustive: false,
                })
            }
            SortKind::Datatype(_) => {
                // An enumeration is exactly its constructors; over a datatype
                // with a field, the constructor values the body names are
                // witness candidates only.
                if let Some(constructors) = self.enumeration_values(sort) {
                    return Some(Domain {
                        points: constructors,
                        exhaustive: true,
                    });
                }
                let points: Vec<TermId> = self
                    .constructor_values_of(body)
                    .into_iter()
                    .filter(|&t| self.terms.get(t).is_some_and(|d| d.sort == sort))
                    .collect();
                Some(Domain {
                    points,
                    exhaustive: false,
                })
            }
            _ => None,
        }
    }

    /// Every constructor of `sort` as a term, when every constructor is
    /// nullary.
    fn enumeration_values(&mut self, sort: SortId) -> Option<Vec<TermId>> {
        let name = self.terms.sorts.datatype_name(sort)?.to_string();
        let names: Vec<String> = {
            let def = self.terms.sorts.get_datatype(&name)?;
            if def.constructors.is_empty()
                || def.constructors.iter().any(|c| !c.selectors.is_empty())
            {
                return None;
            }
            def.constructors
                .iter()
                .map(|c| self.terms.resolve_str(c.name).to_string())
                .collect()
        };
        Some(
            names
                .iter()
                .map(|n| self.terms.mk_dt_constructor(n, [], sort))
                .collect(),
        )
    }

    /// Every ground constructor value `term` spells.
    fn constructor_values_of(&self, term: TermId) -> Vec<TermId> {
        let mut out: Vec<TermId> = Vec::new();
        let mut stack: Vec<TermId> = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if super::array_model::is_constructor_value(current, &self.terms) {
                out.push(current);
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        }
        out.sort_unstable_by_key(|t| t.raw());
        out
    }

    /// Every numeric literal of `term`.
    fn literals_of(&self, term: TermId) -> Vec<TermId> {
        let mut out: Vec<TermId> = Vec::new();
        let mut stack: Vec<TermId> = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if matches!(data.kind, TermKind::IntConst(_) | TermKind::RealConst(_)) {
                out.push(current);
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        }
        out.sort_unstable_by_key(|t| t.raw());
        out
    }

    /// `term` with every maximal closed, quantifier-free, scalar sub-term that
    /// is not already a literal replaced by its value.
    fn fold_closed_subterms(&mut self, term: TermId) -> TermId {
        let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut stack: Vec<TermId> = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current).cloned() else {
                continue;
            };
            let foldable = !is_literal_term(&data.kind)
                && !matches!(data.kind, TermKind::Forall { .. } | TermKind::Exists { .. })
                && self.is_scalar_sort(data.sort)
                && !self.has_quantifier(current)
                && self.terms.free_vars_including_patterns(current).is_empty();
            if foldable && let Some(value) = self.value_term_of(current) {
                map.insert(current, value);
                continue;
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        }
        if map.is_empty() {
            term
        } else {
            self.terms.substitute(term, &map)
        }
    }

    /// The literal a closed quantifier-free scalar term evaluates to.
    fn value_term_of(&mut self, term: TermId) -> Option<TermId> {
        let sort = self.terms.get(term)?.sort;
        let value = match crate::solver::array_completion_certify::evaluate_closed_value(
            &self.solver,
            term,
            &mut self.terms,
        ) {
            Some(value) => value,
            None => {
                let simplified = self.terms.simplify(term);
                crate::solver::array_completion_certify::evaluate_closed_value(
                    &self.solver,
                    simplified,
                    &mut self.terms,
                )?
            }
        };
        Some(match value {
            crate::solver::EvalVal::Bool(flag) => self.terms.mk_bool(flag),
            crate::solver::EvalVal::Num(number) => {
                if sort == self.terms.sorts.int_sort {
                    if *number.denom() != 1 {
                        return None;
                    }
                    self.terms.mk_int(*number.numer())
                } else {
                    self.terms.mk_real(number)
                }
            }
            crate::solver::EvalVal::Bv { value, width } => self.terms.mk_bitvec(value, width),
        })
    }

    /// A closed quantifier-free Boolean term's value.
    fn evaluate_bool(&mut self, term: TermId) -> Option<bool> {
        match crate::solver::array_completion_certify::evaluate_closed(
            &self.solver,
            term,
            &mut self.terms,
        ) {
            Some(value) => Some(value),
            None => {
                let simplified = self.terms.simplify(term);
                crate::solver::array_completion_certify::evaluate_closed(
                    &self.solver,
                    simplified,
                    &mut self.terms,
                )
            }
        }
    }

    /// `∀vars. body` rebuilt.
    fn rebuild_forall(
        &mut self,
        vars: &[(oxiz_core::interner::Spur, SortId)],
        body: TermId,
    ) -> TermId {
        let names: Vec<(String, SortId)> = vars
            .iter()
            .map(|&(name, sort)| (self.terms.resolve_str(name).to_string(), sort))
            .collect();
        let binders: Vec<(&str, SortId)> = names.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        self.terms.mk_forall(binders, body)
    }

    /// Whether `term` holds a quantifier.
    fn has_quantifier(&self, term: TermId) -> bool {
        let mut stack: Vec<TermId> = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = self.terms.get(current) else {
                continue;
            };
            if matches!(data.kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
                return true;
            }
            stack.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        }
        false
    }

    /// Whether `term` is `Bool`-sorted.
    fn is_bool(&self, term: TermId) -> bool {
        self.terms
            .get(term)
            .is_some_and(|t| t.sort == self.terms.sorts.bool_sort)
    }

    /// Whether `sort` is `Bool`, `Int`, `Real` or a bit-vector sort.
    fn is_scalar_sort(&self, sort: SortId) -> bool {
        matches!(
            self.terms.sorts.get(sort).map(|s| &s.kind),
            Some(SortKind::Bool | SortKind::Int | SortKind::Real | SortKind::BitVec(_))
        )
    }
}

/// Whether a term kind is a literal of a scalar sort.
fn is_literal_term(kind: &TermKind) -> bool {
    matches!(
        kind,
        TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
    )
}

/// The integer interval `[lo, hi]` a guard conjunction confines `var` to,
/// read from its top-level conjuncts `var ⊕ c` / `c ⊕ var` with integer
/// literals `c`; `None` when a side is unbounded.
fn int_bounds_of(
    guard: TermId,
    var: TermId,
    terms: &oxiz_core::ast::TermManager,
) -> Option<(BigInt, BigInt)> {
    let conjuncts: Vec<TermId> = match terms.get(guard).map(|t| &t.kind) {
        Some(TermKind::And(args)) => args.to_vec(),
        _ => vec![guard],
    };
    let literal = |term: TermId| match terms.get(term).map(|t| &t.kind) {
        Some(TermKind::IntConst(n)) => Some(n.clone()),
        _ => None,
    };
    let mut lo: Option<BigInt> = None;
    let mut hi: Option<BigInt> = None;
    for conjunct in conjuncts {
        let Some(kind) = terms.get(conjunct).map(|t| t.kind.clone()) else {
            continue;
        };
        let (l, r, op) = match kind {
            TermKind::Le(l, r) => (l, r, "<="),
            TermKind::Lt(l, r) => (l, r, "<"),
            TermKind::Ge(l, r) => (l, r, ">="),
            TermKind::Gt(l, r) => (l, r, ">"),
            // `var = c` is the one-point box `[c, c]`.
            TermKind::Eq(l, r) => {
                let pinned = if l == var {
                    literal(r)
                } else if r == var {
                    literal(l)
                } else {
                    None
                };
                if let Some(c) = pinned {
                    if lo.as_ref().is_none_or(|x| c > *x) {
                        lo = Some(c.clone());
                    }
                    if hi.as_ref().is_none_or(|x| c < *x) {
                        hi = Some(c);
                    }
                }
                continue;
            }
            _ => continue,
        };
        let (op, c) = if l == var {
            let Some(c) = literal(r) else { continue };
            (op, c)
        } else if r == var {
            let Some(c) = literal(l) else { continue };
            let mirrored = match op {
                "<=" => ">=",
                "<" => ">",
                ">=" => "<=",
                _ => "<",
            };
            (mirrored, c)
        } else {
            continue;
        };
        let (bound, is_upper) = match op {
            "<=" => (c, true),
            "<" => (c - 1u8, true),
            ">=" => (c, false),
            _ => (c + 1u8, false),
        };
        if is_upper {
            if hi.as_ref().is_none_or(|h| bound < *h) {
                hi = Some(bound);
            }
        } else if lo.as_ref().is_none_or(|l| bound > *l) {
            lo = Some(bound);
        }
    }
    Some((lo?, hi?))
}
