//! QE Lite - Fast Approximate Quantifier Elimination
//!
//! Provides fast approximate quantifier elimination for common patterns.
//! Falls back to full QE when approximation is not possible.

use crate::ast::{TermId, TermKind, TermManager};
use crate::sort::SortId;
use lasso::Spur;
use std::collections::HashMap;

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
    /// Variable substitutions
    substitutions: HashMap<TermId, TermId>,
}

impl QeLiteSolver {
    /// Create a new QE Lite solver
    pub fn new() -> Self {
        Self {
            config: QeLiteConfig::default(),
            stats: QeLiteStats::default(),
            substitutions: HashMap::new(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: QeLiteConfig) -> Self {
        Self {
            config,
            stats: QeLiteStats::default(),
            substitutions: HashMap::new(),
        }
    }

    /// Try to eliminate a quantifier from a formula
    pub fn eliminate(&mut self, formula: TermId, manager: &mut TermManager) -> QeLiteResult {
        self.stats.attempts += 1;
        self.substitutions.clear();

        // Check if the formula is a quantifier
        let t = match manager.get(formula) {
            Some(t) => t.clone(),
            None => return QeLiteResult::Error("Unknown formula".to_string()),
        };

        match &t.kind {
            TermKind::Forall { vars, body, .. } => {
                self.eliminate_forall_qvar(vars.as_slice(), *body, manager)
            }
            TermKind::Exists { vars, body, .. } => {
                self.eliminate_exists_qvar(vars.as_slice(), *body, manager)
            }
            _ => QeLiteResult::Unchanged,
        }
    }

    /// Try to eliminate an existential quantifier (with quantifier variables)
    fn eliminate_exists_qvar(
        &mut self,
        vars: &[(Spur, SortId)],
        body: TermId,
        manager: &mut TermManager,
    ) -> QeLiteResult {
        // For now, we can't easily do substitution without TermIds for variables
        // This is a simplified version that returns the body as simplified

        if vars.len() == 1 {
            // Try simple cases
            if self.is_trivially_satisfiable(body, manager) {
                self.stats.successes += 1;
                return QeLiteResult::Eliminated(manager.mk_true());
            }
        }

        // Cannot eliminate, return unchanged
        QeLiteResult::Unchanged
    }

    /// Try to eliminate a universal quantifier (with quantifier variables)
    fn eliminate_forall_qvar(
        &mut self,
        vars: &[(Spur, SortId)],
        body: TermId,
        manager: &mut TermManager,
    ) -> QeLiteResult {
        if vars.len() == 1 {
            // Try simple cases
            if self.is_trivially_valid(body, manager) {
                self.stats.successes += 1;
                return QeLiteResult::Eliminated(manager.mk_true());
            }
        }

        // Cannot eliminate, return unchanged
        QeLiteResult::Unchanged
    }

    /// Check if a formula is trivially satisfiable
    fn is_trivially_satisfiable(&self, body: TermId, manager: &TermManager) -> bool {
        let t = match manager.get(body) {
            Some(t) => t,
            None => return false,
        };
        matches!(t.kind, TermKind::True)
    }

    /// Check if a formula is trivially valid
    fn is_trivially_valid(&self, body: TermId, manager: &TermManager) -> bool {
        let t = match manager.get(body) {
            Some(t) => t,
            None => return false,
        };
        matches!(t.kind, TermKind::True)
    }

    /// Try to eliminate an existential quantifier with TermId vars
    #[allow(dead_code)]
    fn eliminate_exists(
        &mut self,
        vars: &[TermId],
        body: TermId,
        manager: &mut TermManager,
    ) -> QeLiteResult {
        // Try equality substitution first
        if self.config.equality_substitution {
            for &var in vars {
                if let Some(subst) = self.find_equality_substitution(var, body, manager) {
                    // Apply substitution
                    let new_body = self.apply_substitution(body, var, subst, manager);
                    self.stats.equality_subs += 1;

                    // If only one variable, we're done
                    if vars.len() == 1 {
                        self.stats.successes += 1;
                        return QeLiteResult::Eliminated(new_body);
                    }

                    // Otherwise, continue with remaining variables
                    let remaining: Vec<_> = vars.iter().filter(|&&v| v != var).copied().collect();
                    return self.wrap_exists(&remaining, new_body, manager);
                }
            }
        }

        // Try bound elimination
        if self.config.bound_elimination {
            for &var in vars {
                if let Some(result) = self.try_bound_elimination(var, body, manager) {
                    self.stats.bound_elims += 1;

                    if vars.len() == 1 {
                        self.stats.successes += 1;
                        return QeLiteResult::Eliminated(result);
                    }

                    let remaining: Vec<_> = vars.iter().filter(|&&v| v != var).copied().collect();
                    return self.wrap_exists(&remaining, result, manager);
                }
            }
        }

        QeLiteResult::Unchanged
    }

    /// Try to eliminate a universal quantifier with TermId vars
    #[allow(dead_code)]
    fn eliminate_forall(
        &mut self,
        vars: &[TermId],
        body: TermId,
        manager: &mut TermManager,
    ) -> QeLiteResult {
        // ∀x.φ(x) is equivalent to ¬∃x.¬φ(x)
        // So we can use similar techniques

        // Try equality substitution
        if self.config.equality_substitution {
            for &var in vars {
                // For forall, we need an equality that holds for all values
                if let Some(subst) = self.find_forall_equality(var, body, manager) {
                    let new_body = self.apply_substitution(body, var, subst, manager);
                    self.stats.equality_subs += 1;

                    if vars.len() == 1 {
                        self.stats.successes += 1;
                        return QeLiteResult::Eliminated(new_body);
                    }

                    let remaining: Vec<_> = vars.iter().filter(|&&v| v != var).copied().collect();
                    return self.wrap_forall(&remaining, new_body, manager);
                }
            }
        }

        QeLiteResult::Unchanged
    }

    /// Find an equality that can be used for substitution
    fn find_equality_substitution(
        &self,
        var: TermId,
        body: TermId,
        manager: &TermManager,
    ) -> Option<TermId> {
        let t = manager.get(body)?;

        match &t.kind {
            // x = t (where x doesn't appear in t)
            TermKind::Eq(a, b) => {
                if *a == var && !self.contains_var(*b, var, manager) {
                    return Some(*b);
                }
                if *b == var && !self.contains_var(*a, var, manager) {
                    return Some(*a);
                }
                None
            }
            // φ ∧ (x = t) or (x = t) ∧ φ
            TermKind::And(args) => {
                for arg in args.iter() {
                    if let Some(t) = manager.get(*arg)
                        && let TermKind::Eq(a, b) = &t.kind
                    {
                        if *a == var && !self.contains_var(*b, var, manager) {
                            return Some(*b);
                        }
                        if *b == var && !self.contains_var(*a, var, manager) {
                            return Some(*a);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Find an equality for forall elimination
    fn find_forall_equality(
        &self,
        var: TermId,
        body: TermId,
        manager: &TermManager,
    ) -> Option<TermId> {
        let t = manager.get(body)?;

        // For ∀x. (x = t → φ), we can substitute t for x in φ
        if let TermKind::Implies(antecedent, _consequent) = &t.kind
            && let Some(ant_t) = manager.get(*antecedent)
            && let TermKind::Eq(a, b) = &ant_t.kind
        {
            if *a == var && !self.contains_var(*b, var, manager) {
                return Some(*b);
            }
            if *b == var && !self.contains_var(*a, var, manager) {
                return Some(*a);
            }
        }

        None
    }

    /// Check if a term contains a variable
    #[allow(clippy::only_used_in_recursion)]
    fn contains_var(&self, term: TermId, var: TermId, manager: &TermManager) -> bool {
        if term == var {
            return true;
        }

        let t = match manager.get(term) {
            Some(t) => t,
            None => return false,
        };

        match &t.kind {
            TermKind::Not(a) | TermKind::Neg(a) => self.contains_var(*a, var, manager),
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => args.iter().any(|&arg| self.contains_var(arg, var, manager)),
            TermKind::Xor(a, b)
            | TermKind::Implies(a, b)
            | TermKind::Eq(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b) => {
                self.contains_var(*a, var, manager) || self.contains_var(*b, var, manager)
            }
            TermKind::Ite(a, b, c) => {
                self.contains_var(*a, var, manager)
                    || self.contains_var(*b, var, manager)
                    || self.contains_var(*c, var, manager)
            }
            _ => false,
        }
    }

    /// Apply a substitution to a term
    #[allow(clippy::only_used_in_recursion)]
    fn apply_substitution(
        &self,
        term: TermId,
        var: TermId,
        subst: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        if term == var {
            return subst;
        }

        let t = match manager.get(term) {
            Some(t) => t.clone(),
            None => return term,
        };

        match &t.kind {
            TermKind::Not(a) => {
                let new_a = self.apply_substitution(*a, var, subst, manager);
                manager.mk_not(new_a)
            }
            TermKind::And(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| self.apply_substitution(arg, var, subst, manager))
                    .collect();
                manager.mk_and(new_args)
            }
            TermKind::Or(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| self.apply_substitution(arg, var, subst, manager))
                    .collect();
                manager.mk_or(new_args)
            }
            TermKind::Eq(a, b) => {
                let new_a = self.apply_substitution(*a, var, subst, manager);
                let new_b = self.apply_substitution(*b, var, subst, manager);
                manager.mk_eq(new_a, new_b)
            }
            TermKind::Lt(a, b) => {
                let new_a = self.apply_substitution(*a, var, subst, manager);
                let new_b = self.apply_substitution(*b, var, subst, manager);
                manager.mk_lt(new_a, new_b)
            }
            TermKind::Le(a, b) => {
                let new_a = self.apply_substitution(*a, var, subst, manager);
                let new_b = self.apply_substitution(*b, var, subst, manager);
                manager.mk_le(new_a, new_b)
            }
            _ => term, // No substitution needed for other cases
        }
    }

    /// Try bound elimination for a variable
    fn try_bound_elimination(
        &mut self,
        var: TermId,
        body: TermId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        // Collect bounds for the variable
        let bounds = self.collect_bounds(var, body, manager);

        if bounds.lower.is_empty() && bounds.upper.is_empty() {
            return None;
        }

        // If we have both lower and upper bounds, check if they're trivially satisfied
        if !bounds.lower.is_empty() && !bounds.upper.is_empty() {
            // Generate: ∨_{l ∈ lower, u ∈ upper} (l < u ∧ φ[x/l] ∧ φ[x/u])
            // Simplified: just check if any bound pair works
            // For now, return the body without the quantifier if we found bounds
            self.stats.simplifications = self.stats.simplifications.saturating_add(1);
            return Some(body);
        }

        None
    }

    /// Collect bounds for a variable
    fn collect_bounds(&self, var: TermId, body: TermId, manager: &TermManager) -> VarBounds {
        let mut bounds = VarBounds::default();

        let t = match manager.get(body) {
            Some(t) => t,
            None => return bounds,
        };

        match &t.kind {
            // x < t or x <= t
            TermKind::Lt(a, b) | TermKind::Le(a, b) => {
                if *a == var && !self.contains_var(*b, var, manager) {
                    bounds.upper.push(*b);
                } else if *b == var && !self.contains_var(*a, var, manager) {
                    bounds.lower.push(*a);
                }
            }
            // x > t or x >= t
            TermKind::Gt(a, b) | TermKind::Ge(a, b) => {
                if *a == var && !self.contains_var(*b, var, manager) {
                    bounds.lower.push(*b);
                } else if *b == var && !self.contains_var(*a, var, manager) {
                    bounds.upper.push(*a);
                }
            }
            TermKind::And(args) => {
                for &arg in args.iter() {
                    let arg_bounds = self.collect_bounds(var, arg, manager);
                    bounds.lower.extend(arg_bounds.lower);
                    bounds.upper.extend(arg_bounds.upper);
                }
            }
            _ => {}
        }

        bounds
    }

    /// Wrap a body with an existential quantifier
    fn wrap_exists(
        &mut self,
        vars: &[TermId],
        body: TermId,
        _manager: &mut TermManager,
    ) -> QeLiteResult {
        if vars.is_empty() {
            QeLiteResult::Eliminated(body)
        } else {
            // Cannot fully eliminate, but we simplified
            // The body still contains quantified variables
            self.stats.simplifications = self.stats.simplifications.saturating_add(1);
            QeLiteResult::Simplified(body)
        }
    }

    /// Wrap a body with a universal quantifier
    fn wrap_forall(
        &mut self,
        vars: &[TermId],
        body: TermId,
        _manager: &mut TermManager,
    ) -> QeLiteResult {
        if vars.is_empty() {
            QeLiteResult::Eliminated(body)
        } else {
            // Cannot fully eliminate, simplified only
            self.stats.simplifications = self.stats.simplifications.saturating_add(1);
            QeLiteResult::Simplified(body)
        }
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

/// Variable bounds
#[derive(Debug, Default)]
struct VarBounds {
    lower: Vec<TermId>,
    upper: Vec<TermId>,
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
}
