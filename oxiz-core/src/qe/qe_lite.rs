//! QE Lite - Fast Approximate Quantifier Elimination
//!
//! Provides fast, always-equivalence-preserving quantifier elimination for the
//! cheap patterns that dominate real inputs, falling back to an honest
//! "unchanged" (or partially simplified) result when no cheap rule applies.
//!
//! The rules implemented are the standard QE-lite ones:
//!
//! * **Unused-variable dropping** — `∃x.φ ≡ φ` and `∀x.φ ≡ φ` when `x` does not
//!   occur free in `φ`.
//! * **Definitional-equality substitution** (destructive equality resolution) —
//!   `∃x.(x = t ∧ φ) ≡ φ[x := t]` when `x ∉ t`, and dually
//!   `∀x.(x = t → φ) ≡ φ[x := t]`.
//! * **Distribution** — `∃x.(A ∧ B) ≡ A ∧ ∃x.B` and `∀x.(A ∨ B) ≡ A ∨ ∀x.B`
//!   when `x ∉ A`, applied to pull `x`-free operands out of the quantifier and
//!   recurse on the `x`-dependent remainder.
//!
//! Every rule is a logically equivalent transformation; a quantifier is only
//! reported as `Eliminated` once the eliminated variable is provably absent
//! from the result.

use crate::ast::{TermId, TermKind, TermManager};
use crate::interner::Spur;
use crate::prelude::FxHashMap;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;

/// Configuration for QE Lite
#[derive(Debug, Clone)]
pub struct QeLiteConfig {
    /// Maximum formula size to process
    pub max_formula_size: usize,
    /// Enable equality substitution
    pub equality_substitution: bool,
    /// Enable simple bound elimination
    pub bound_elimination: bool,
    /// Enable divisibility handling
    pub divisibility_handling: bool,
}

impl Default for QeLiteConfig {
    fn default() -> Self {
        Self {
            max_formula_size: 1000,
            equality_substitution: true,
            bound_elimination: true,
            divisibility_handling: true,
        }
    }
}

/// Result of QE Lite processing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QeLiteResult {
    /// Quantifier was eliminated
    Eliminated(TermId),
    /// Quantifier was simplified but not fully eliminated
    Simplified(TermId),
    /// No elimination possible
    Unchanged,
    /// Error during processing
    Error(String),
}

impl QeLiteResult {
    /// Check if elimination was successful
    pub fn is_eliminated(&self) -> bool {
        matches!(self, QeLiteResult::Eliminated(_))
    }

    /// Get the result term if available
    pub fn result_term(&self) -> Option<TermId> {
        match self {
            QeLiteResult::Eliminated(t) | QeLiteResult::Simplified(t) => Some(*t),
            _ => None,
        }
    }
}

/// Statistics for QE Lite
#[derive(Debug, Clone, Default)]
pub struct QeLiteStats {
    /// Number of elimination attempts
    pub attempts: u64,
    /// Number of successful eliminations
    pub successes: u64,
    /// Number of simplifications
    pub simplifications: u64,
    /// Number of equality substitutions
    pub equality_subs: u64,
    /// Number of bound eliminations
    pub bound_elims: u64,
}

/// QE Lite solver for fast approximate quantifier elimination
#[derive(Debug)]
pub struct QeLiteSolver {
    /// Configuration
    config: QeLiteConfig,
    /// Statistics
    stats: QeLiteStats,
}

impl QeLiteSolver {
    /// Create a new QE Lite solver
    pub fn new() -> Self {
        Self {
            config: QeLiteConfig::default(),
            stats: QeLiteStats::default(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: QeLiteConfig) -> Self {
        Self {
            config,
            stats: QeLiteStats::default(),
        }
    }

    /// Try to eliminate a quantifier from a formula
    pub fn eliminate(&mut self, formula: TermId, manager: &mut TermManager) -> QeLiteResult {
        self.stats.attempts += 1;

        // Check if the formula is a quantifier
        let kind = match manager.get(formula) {
            Some(t) => t.kind.clone(),
            None => return QeLiteResult::Error("Unknown formula".to_string()),
        };

        match kind {
            TermKind::Forall { vars, body, .. } => {
                self.eliminate_forall_qvar(vars.as_slice(), body, manager)
            }
            TermKind::Exists { vars, body, .. } => {
                self.eliminate_exists_qvar(vars.as_slice(), body, manager)
            }
            _ => QeLiteResult::Unchanged,
        }
    }

    /// Eliminate an existential quantifier `∃vars. body`, one variable at a time.
    fn eliminate_exists_qvar(
        &mut self,
        vars: &[(Spur, SortId)],
        body: TermId,
        manager: &mut TermManager,
    ) -> QeLiteResult {
        let mut current = body;
        let mut remaining: Vec<(Spur, SortId)> = Vec::new();
        for &(spur, sort) in vars {
            let name = manager.resolve_str(spur).to_string();
            let var_id = manager.mk_var(&name, sort);
            match self.try_eliminate_one_exists(var_id, current, manager) {
                Some(new_body) => current = new_body,
                None => remaining.push((spur, sort)),
            }
        }

        if remaining.is_empty() {
            self.stats.successes += 1;
            QeLiteResult::Eliminated(current)
        } else if remaining.len() < vars.len() {
            self.stats.simplifications = self.stats.simplifications.saturating_add(1);
            let rebuilt = self.rebuild_exists(&remaining, current, manager);
            QeLiteResult::Simplified(rebuilt)
        } else {
            QeLiteResult::Unchanged
        }
    }

    /// Eliminate a universal quantifier `∀vars. body`, one variable at a time.
    fn eliminate_forall_qvar(
        &mut self,
        vars: &[(Spur, SortId)],
        body: TermId,
        manager: &mut TermManager,
    ) -> QeLiteResult {
        let mut current = body;
        let mut remaining: Vec<(Spur, SortId)> = Vec::new();
        for &(spur, sort) in vars {
            let name = manager.resolve_str(spur).to_string();
            let var_id = manager.mk_var(&name, sort);
            match self.try_eliminate_one_forall(var_id, current, manager) {
                Some(new_body) => current = new_body,
                None => remaining.push((spur, sort)),
            }
        }

        if remaining.is_empty() {
            self.stats.successes += 1;
            QeLiteResult::Eliminated(current)
        } else if remaining.len() < vars.len() {
            self.stats.simplifications = self.stats.simplifications.saturating_add(1);
            let rebuilt = self.rebuild_forall(&remaining, current, manager);
            QeLiteResult::Simplified(rebuilt)
        } else {
            QeLiteResult::Unchanged
        }
    }

    /// Attempt to eliminate a single existentially-quantified variable, returning
    /// the equivalent `var`-free body on success.
    ///
    /// Rule 3 below recurses on the
    /// `x`-dependent conjuncts, so its depth is the number of conjuncts of
    /// the body — caller-controlled and unbounded. The descent is therefore
    /// written as a loop that stacks the pulled-out `x`-free operands and
    /// re-assembles them afterwards, which is exactly what the recursive
    /// unwinding did.
    fn try_eliminate_one_exists(
        &mut self,
        var: TermId,
        body: TermId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let mut pulled_out: Vec<Vec<TermId>> = Vec::new();
        let mut current = body;

        let innermost = loop {
            // Rule 1: `x` does not occur — ∃x.φ ≡ φ.
            if !contains_var(current, var, manager) {
                break current;
            }

            // Rule 2: definitional equality — ∃x.(x = t ∧ φ) ≡ φ[x := t]  (DER).
            let mut substituted = None;
            if self.config.equality_substitution
                && let Some(t) = find_equality_substitution(var, current, manager)
            {
                let mut subst = FxHashMap::default();
                subst.insert(var, t);
                let new_body = manager.substitute(current, &subst);
                // Only accept when the variable is genuinely gone.
                if !contains_var(new_body, var, manager) {
                    self.stats.equality_subs += 1;
                    substituted = Some(new_body);
                }
            }
            if let Some(new_body) = substituted {
                break new_body;
            }

            // Rule 3: distribute over conjunction — ∃x.(A ∧ B) ≡ A ∧ ∃x.B, x ∉ A.
            let (free, dep) = self.split_conjunction(var, current, manager)?;
            pulled_out.push(free);
            current = manager.mk_and(dep);
        };

        Some(self.reassemble(innermost, pulled_out, true, manager))
    }

    /// Split an `and`-body into its `x`-free and `x`-dependent operands,
    /// returning `None` when no progress is possible.
    fn split_conjunction(
        &self,
        var: TermId,
        body: TermId,
        manager: &TermManager,
    ) -> Option<(Vec<TermId>, Vec<TermId>)> {
        let args = match &manager.get(body)?.kind {
            TermKind::And(args) => args.clone(),
            _ => return None,
        };
        let (free, dep) = split_by_var(&args, var, manager);
        if free.is_empty() || dep.is_empty() {
            // No `x`-free operand to pull out — no progress.
            return None;
        }
        Some((free, dep))
    }

    /// Split an `or`-body into its `x`-free and `x`-dependent operands,
    /// returning `None` when no progress is possible.
    fn split_disjunction(
        &self,
        var: TermId,
        body: TermId,
        manager: &TermManager,
    ) -> Option<(Vec<TermId>, Vec<TermId>)> {
        let args = match &manager.get(body)?.kind {
            TermKind::Or(args) => args.clone(),
            _ => return None,
        };
        let (free, dep) = split_by_var(&args, var, manager);
        if free.is_empty() || dep.is_empty() {
            return None;
        }
        Some((free, dep))
    }

    /// Re-assemble the operands pulled out on the way down, outermost last —
    /// the iterative equivalent of returning through the recursive frames.
    fn reassemble(
        &self,
        innermost: TermId,
        pulled_out: Vec<Vec<TermId>>,
        conjunctive: bool,
        manager: &mut TermManager,
    ) -> TermId {
        let mut acc = innermost;
        for mut parts in pulled_out.into_iter().rev() {
            parts.push(acc);
            acc = if conjunctive {
                manager.mk_and(parts)
            } else {
                manager.mk_or(parts)
            };
        }
        acc
    }

    /// Attempt to eliminate a single universally-quantified variable.
    /// Iterative counterpart of [`Self::try_eliminate_one_exists`] for `∀`,
    /// distributing over disjunction instead of conjunction.
    fn try_eliminate_one_forall(
        &mut self,
        var: TermId,
        body: TermId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let mut pulled_out: Vec<Vec<TermId>> = Vec::new();
        let mut current = body;

        let innermost = loop {
            // Rule 1: `x` does not occur — ∀x.φ ≡ φ.
            if !contains_var(current, var, manager) {
                break current;
            }

            // Rule 2: definitional equality — ∀x.(x = t → φ) ≡ φ[x := t].
            let mut substituted = None;
            if self.config.equality_substitution
                && let Some(t) = find_forall_equality(var, current, manager)
            {
                let mut subst = FxHashMap::default();
                subst.insert(var, t);
                let new_body = manager.substitute(current, &subst);
                if !contains_var(new_body, var, manager) {
                    self.stats.equality_subs += 1;
                    substituted = Some(new_body);
                }
            }
            if let Some(new_body) = substituted {
                break new_body;
            }

            // Rule 3: distribute over disjunction — ∀x.(A ∨ B) ≡ A ∨ ∀x.B, x ∉ A.
            let (free, dep) = self.split_disjunction(var, current, manager)?;
            pulled_out.push(free);
            current = manager.mk_or(dep);
        };

        Some(self.reassemble(innermost, pulled_out, false, manager))
    }

    /// Rebuild `∃remaining. body` for the variables that could not be eliminated.
    fn rebuild_exists(
        &self,
        vars: &[(Spur, SortId)],
        body: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        let names: Vec<(String, SortId)> = vars
            .iter()
            .map(|&(s, sort)| (manager.resolve_str(s).to_string(), sort))
            .collect();
        manager.mk_exists(names.iter().map(|(n, s)| (n.as_str(), *s)), body)
    }

    /// Rebuild `∀remaining. body` for the variables that could not be eliminated.
    fn rebuild_forall(
        &self,
        vars: &[(Spur, SortId)],
        body: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        let names: Vec<(String, SortId)> = vars
            .iter()
            .map(|&(s, sort)| (manager.resolve_str(s).to_string(), sort))
            .collect();
        manager.mk_forall(names.iter().map(|(n, s)| (n.as_str(), *s)), body)
    }

    /// Get statistics
    pub fn stats(&self) -> &QeLiteStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = QeLiteStats::default();
    }
}

/// Partition `args` into the operands that are `x`-free and those that mention
/// `x`, preserving order.
fn split_by_var(args: &[TermId], var: TermId, manager: &TermManager) -> (Vec<TermId>, Vec<TermId>) {
    let mut free = Vec::new();
    let mut dep = Vec::new();
    for &arg in args {
        if contains_var(arg, var, manager) {
            dep.push(arg);
        } else {
            free.push(arg);
        }
    }
    (free, dep)
}

/// Find `t` such that `body` conjunctively entails `var = t` with `var ∉ t`,
/// suitable for destructive equality resolution under `∃`.
fn find_equality_substitution(var: TermId, body: TermId, manager: &TermManager) -> Option<TermId> {
    let t = manager.get(body)?;
    match &t.kind {
        // x = t (where x doesn't appear in t)
        TermKind::Eq(a, b) => equality_side(*a, *b, var, manager),
        // (x = t) as a top-level conjunct.
        TermKind::And(args) => {
            for arg in args.iter() {
                if let Some(inner) = manager.get(*arg)
                    && let TermKind::Eq(a, b) = &inner.kind
                    && let Some(sol) = equality_side(*a, *b, var, manager)
                {
                    return Some(sol);
                }
            }
            None
        }
        _ => None,
    }
}

/// Find `t` for `∀x.(x = t → φ)`: an equality antecedent solving for `var`.
fn find_forall_equality(var: TermId, body: TermId, manager: &TermManager) -> Option<TermId> {
    let t = manager.get(body)?;
    if let TermKind::Implies(antecedent, _consequent) = &t.kind
        && let Some(ant_t) = manager.get(*antecedent)
        && let TermKind::Eq(a, b) = &ant_t.kind
    {
        return equality_side(*a, *b, var, manager);
    }
    None
}

/// If one side of `a = b` is exactly `var` and the other is `var`-free, return
/// the other side.
fn equality_side(a: TermId, b: TermId, var: TermId, manager: &TermManager) -> Option<TermId> {
    if a == var && !contains_var(b, var, manager) {
        return Some(b);
    }
    if b == var && !contains_var(a, var, manager) {
        return Some(a);
    }
    None
}

/// Whether `term` syntactically contains `var`.
///
/// Complete over every `TermKind` (it walks through
/// [`crate::ast::traversal::get_children`]), so a "does not occur" answer is
/// trustworthy — an unsound "eliminated" result can never be produced from a
/// missed occurrence.
///
/// Uses an explicit heap stack rather than native recursion: this used to
/// recurse once per level of term nesting with no depth guard, so a
/// pathologically deep (but valid) term could overflow the call stack. There
/// is no error channel to bail out through either (`bool` has no "unknown"
/// value), and a depth cap would only ever have been able to return a
/// silently *wrong* answer past the cap — worse than the crash it replaces,
/// since [`QeLiteSolver`] treats "does not occur" as license to eliminate the
/// quantifier outright. A `Vec`-backed stack has no such limit (bounded by
/// memory, not the fixed native stack).
///
/// `visited` memoizes subterms already walked without finding `var`: whether
/// a subterm's tree contains `var` never depends on where it is reached from
/// (unlike free-variable collection, which is binder-scope-sensitive), so —
/// unlike `TermManager::free_vars` — memoizing globally here is always sound;
/// it just avoids re-walking a shared subterm once it is known not to
/// contain `var`.
fn contains_var(term: TermId, var: TermId, manager: &TermManager) -> bool {
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = vec![term];
    while let Some(id) = stack.pop() {
        if id == var {
            return true;
        }
        if !visited.insert(id) {
            continue;
        }
        let Some(t) = manager.get(id) else {
            continue;
        };
        for &child in &crate::ast::traversal::get_children(&t.kind) {
            stack.push(child);
        }
    }
    false
}

impl Default for QeLiteSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qe_lite_config() {
        let config = QeLiteConfig::default();
        assert_eq!(config.max_formula_size, 1000);
        assert!(config.equality_substitution);
        assert!(config.bound_elimination);
    }

    #[test]
    fn test_qe_lite_result() {
        let t = TermId::from(1u32);
        let elim = QeLiteResult::Eliminated(t);
        assert!(elim.is_eliminated());
        assert_eq!(elim.result_term(), Some(t));

        let unchanged = QeLiteResult::Unchanged;
        assert!(!unchanged.is_eliminated());
        assert_eq!(unchanged.result_term(), None);
    }

    #[test]
    fn test_qe_lite_creation() {
        let solver = QeLiteSolver::new();
        assert_eq!(solver.stats().attempts, 0);
    }

    #[test]
    fn test_qe_lite_stats() {
        let stats = QeLiteStats::default();
        assert_eq!(stats.attempts, 0);
        assert_eq!(stats.successes, 0);
    }

    #[test]
    fn test_qe_lite_with_config() {
        let config = QeLiteConfig {
            max_formula_size: 500,
            equality_substitution: false,
            bound_elimination: true,
            divisibility_handling: false,
        };
        let solver = QeLiteSolver::with_config(config.clone());
        assert_eq!(solver.config.max_formula_size, 500);
        assert!(!solver.config.equality_substitution);
    }

    #[test]
    fn test_reset_stats() {
        let mut solver = QeLiteSolver::new();
        solver.stats.attempts = 10;
        solver.reset_stats();
        assert_eq!(solver.stats().attempts, 0);
    }

    fn int_var(tm: &mut TermManager, name: &str) -> TermId {
        let int_sort = tm.sorts.int_sort;
        tm.mk_var(name, int_sort)
    }

    #[test]
    fn exists_unused_variable_is_dropped() {
        // ∃x. (y > 0) ≡ (y > 0).
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let y = int_var(&mut tm, "y");
        let zero = tm.mk_int(0);
        let body = tm.mk_gt(y, zero);
        let quant = tm.mk_exists([("x", int_sort)], body);

        let mut solver = QeLiteSolver::new();
        let result = solver.eliminate(quant, &mut tm);
        assert_eq!(result, QeLiteResult::Eliminated(body));
    }

    #[test]
    fn exists_equality_substitution() {
        // ∃x. (x = y ∧ z > x) ≡ (z > y), with x eliminated by DER.
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = int_var(&mut tm, "x");
        let y = int_var(&mut tm, "y");
        let z = int_var(&mut tm, "z");
        let eq = tm.mk_eq(x, y);
        let gt = tm.mk_gt(z, x);
        let body = tm.mk_and(vec![eq, gt]);
        let quant = tm.mk_exists([("x", int_sort)], body);

        let mut solver = QeLiteSolver::new();
        let result = solver.eliminate(quant, &mut tm);
        assert!(result.is_eliminated());
        let term = result.result_term().expect("has term");
        let x_spur = tm.intern_str("x");
        assert!(!mentions(term, x_spur, &tm), "x still present");
        // The eliminated body must still constrain z relative to y (z > y).
        let z = int_var(&mut tm, "z");
        let y = int_var(&mut tm, "y");
        let expected = tm.mk_gt(z, y);
        let expected_s = tm.simplify(expected);
        let simplified = tm.simplify(term);
        assert_eq!(simplified, expected_s);
    }

    #[test]
    fn exists_distribution_over_conjunction() {
        // ∃x. (y > 0 ∧ x = z ∧ w < x) — x-free conjunct pulled out, DER on rest.
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = int_var(&mut tm, "x");
        let y = int_var(&mut tm, "y");
        let z = int_var(&mut tm, "z");
        let w = int_var(&mut tm, "w");
        let zero = tm.mk_int(0);
        let a = tm.mk_gt(y, zero);
        let b = tm.mk_eq(x, z);
        let c = tm.mk_lt(w, x);
        let body = tm.mk_and(vec![a, b, c]);
        let quant = tm.mk_exists([("x", int_sort)], body);

        let mut solver = QeLiteSolver::new();
        let result = solver.eliminate(quant, &mut tm);
        assert!(result.is_eliminated());
        let term = result.result_term().expect("has term");
        let x_spur = tm.intern_str("x");
        assert!(!mentions(term, x_spur, &tm), "x still present");
    }

    #[test]
    fn exists_without_cheap_rule_is_unchanged() {
        // ∃x. (x > 0 ∧ x < y) has no cheap rule → honest Unchanged.
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = int_var(&mut tm, "x");
        let y = int_var(&mut tm, "y");
        let zero = tm.mk_int(0);
        let a = tm.mk_gt(x, zero);
        let b = tm.mk_lt(x, y);
        let body = tm.mk_and(vec![a, b]);
        let quant = tm.mk_exists([("x", int_sort)], body);

        let mut solver = QeLiteSolver::new();
        let result = solver.eliminate(quant, &mut tm);
        assert_eq!(result, QeLiteResult::Unchanged);
    }

    #[test]
    fn forall_unused_variable_is_dropped() {
        // ∀x. (y > 0) ≡ (y > 0).
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let y = int_var(&mut tm, "y");
        let zero = tm.mk_int(0);
        let body = tm.mk_gt(y, zero);
        let quant = tm.mk_forall([("x", int_sort)], body);

        let mut solver = QeLiteSolver::new();
        let result = solver.eliminate(quant, &mut tm);
        assert_eq!(result, QeLiteResult::Eliminated(body));
    }

    #[test]
    fn forall_equality_substitution() {
        // ∀x. (x = 5 → x > 3) ≡ (5 > 3) ≡ true.
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = int_var(&mut tm, "x");
        let five = tm.mk_int(5);
        let three = tm.mk_int(3);
        let eq = tm.mk_eq(x, five);
        let gt = tm.mk_gt(x, three);
        let body = tm.mk_implies(eq, gt);
        let quant = tm.mk_forall([("x", int_sort)], body);

        let mut solver = QeLiteSolver::new();
        let result = solver.eliminate(quant, &mut tm);
        assert!(result.is_eliminated());
        let term = result.result_term().expect("has term");
        let x_spur = tm.intern_str("x");
        assert!(!mentions(term, x_spur, &tm), "x still present");
        let simplified = tm.simplify(term);
        assert!(matches!(
            tm.get(simplified).map(|t| &t.kind),
            Some(TermKind::True)
        ));
    }

    /// Local occurrence check mirroring the module's internal one.
    fn mentions(id: TermId, name: Spur, tm: &TermManager) -> bool {
        let Some(term) = tm.get(id) else {
            return false;
        };
        if let TermKind::Var(s) = term.kind {
            return s == name;
        }
        crate::ast::traversal::get_children(&term.kind)
            .iter()
            .any(|&c| mentions(c, name, tm))
    }

    #[test]
    fn contains_var_shallow_true_and_false() {
        // Direct behaviour-preservation pin for the iterative rewrite: `var`
        // present deep inside one operand (true), absent from an unrelated
        // term built from different variables (false).
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = int_var(&mut tm, "x");
        let y = int_var(&mut tm, "y");
        let z = int_var(&mut tm, "z");
        let zero = tm.mk_int(0);

        let inner = tm.mk_add(vec![y, x]);
        let haystack = tm.mk_gt(inner, zero);
        assert!(
            contains_var(haystack, x, &tm),
            "x occurs inside (y + x) > 0"
        );
        assert!(
            contains_var(haystack, y, &tm),
            "y occurs inside (y + x) > 0"
        );
        assert!(
            !contains_var(haystack, z, &tm),
            "z does not occur anywhere in (y + x) > 0"
        );

        let _ = int_sort;
    }

    #[test]
    fn contains_var_deep_chain_on_tiny_stack_finds_and_misses_correctly() {
        // Regression: `contains_var` used to recurse natively once per level
        // of term nesting with no depth guard at all. Built iteratively
        // (never recursively, which would overflow before the assertion
        // runs) and run inside a thread with a deliberately small 128 KiB
        // stack: the call returning at all is part of the assertion, but the
        // boolean answer must also be exactly correct in both directions —
        // `var` present at the very bottom of the chain (true), and a
        // different variable that never occurs anywhere in it (false).
        const DEPTH: usize = 12_500;

        let (found_x, found_absent_y) = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(move || {
                let mut tm = TermManager::new();
                let int_sort = tm.sorts.int_sort;
                let bool_sort = tm.sorts.bool_sort;
                let x = tm.mk_var("x", int_sort);
                let y = tm.mk_var("y", int_sort);
                let zero = tm.mk_int(0);
                // `x > 0`, then wrapped in a chain of `not`s built via
                // `intern` directly (bypassing `mk_not`'s double-negation
                // simplification, which would otherwise collapse the chain
                // instead of building it), guaranteeing the chain actually
                // reaches the requested depth.
                let mut term = tm.mk_gt(x, zero);
                for _ in 0..DEPTH {
                    term = tm.intern(TermKind::Not(term), bool_sort);
                }
                (contains_var(term, x, &tm), contains_var(term, y, &tm))
            })
            .expect("spawning the constrained-stack test thread should succeed")
            .join()
            .expect("the constrained-stack thread must not panic");

        assert!(
            found_x,
            "x sits at the bottom of the chain and must be found"
        );
        assert!(
            !found_absent_y,
            "y never occurs anywhere in the chain and must not be reported as found"
        );
    }
}

#[cfg(test)]
mod deep_walk_tests {
    use super::*;
    use crate::ast::TermManager;

    /// Build `x = k ∧ p_1 ∧ ... ∧ p_n` nested so that each level of
    /// [`QeLiteSolver::try_eliminate_one_exists`]'s distribution rule peels
    /// off exactly one `x`-free conjunct.
    fn nested_conjunction(manager: &mut TermManager, levels: usize) -> (TermId, TermId) {
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let x = manager.mk_var("x", int_sort);
        let k = manager.mk_int(7);
        let mut body = manager.mk_eq(x, k);
        for i in 0..levels {
            let free = manager.mk_var(&format!("p{i}"), bool_sort);
            body = manager.mk_and([free, body]);
        }
        (x, body)
    }

    #[test]
    fn test_nested_distribution_pin() {
        let mut manager = TermManager::new();
        let (x, body) = nested_conjunction(&mut manager, 3);
        let mut solver = QeLiteSolver::new();
        let eliminated = solver
            .try_eliminate_one_exists(x, body, &mut manager)
            .expect("the variable is definitionally determined");
        assert!(
            !contains_var(eliminated, x, &manager),
            "the eliminated variable must be gone"
        );
    }

    #[test]
    fn test_nested_distribution_deep_does_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let mut manager = TermManager::new();
                let (x, body) = nested_conjunction(&mut manager, 20_000);
                let mut solver = QeLiteSolver::new();
                let eliminated = solver.try_eliminate_one_exists(x, body, &mut manager);
                eliminated.map(|t| contains_var(t, x, &manager))
            })
            .expect("thread spawn should succeed");

        assert_eq!(
            handle.join().expect("deep elimination must not overflow"),
            Some(false)
        );
    }
}
