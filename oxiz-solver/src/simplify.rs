//! Formula simplification and preprocessing
//!
//! This module provides simplification passes that run before the main solver
//! to reduce problem size and improve solving performance.

// Allow these clippy lints for simplification code patterns
#![allow(clippy::map_entry)] // contains_key + insert pattern used for clarity
#![allow(clippy::only_used_in_recursion)] // recursive simplification intentional
#![allow(clippy::for_kv_map)] // iterating map keys with values pattern

use oxiz_core::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashMap;

/// Simplification statistics
#[derive(Debug, Clone, Default)]
pub struct SimplifyStats {
    /// Number of constant propagations performed
    pub const_propagations: usize,
    /// Number of terms eliminated
    pub terms_eliminated: usize,
    /// Number of trivial equations detected
    pub trivial_equalities: usize,
    /// Number of contradictions found
    pub contradictions_found: usize,
    /// Number of nested operations flattened
    pub operations_flattened: usize,
    /// Number of duplicate literals eliminated
    pub duplicates_eliminated: usize,
    /// Number of tautologies detected
    pub tautologies_detected: usize,
}

/// Context-aware formula simplifier
///
/// Performs simplification passes including:
/// - Constant propagation
/// - Boolean simplification
/// - Trivial equality elimination
/// - Contradiction detection
#[derive(Debug)]
pub struct Simplifier {
    /// Cache of simplified terms
    cache: FxHashMap<TermId, TermId>,
    /// Statistics
    stats: SimplifyStats,
}

impl Simplifier {
    /// Create a new simplifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            stats: SimplifyStats::default(),
        }
    }

    /// Get simplification statistics
    #[must_use]
    #[allow(dead_code)]
    pub fn stats(&self) -> &SimplifyStats {
        &self.stats
    }

    /// Reset the simplifier state
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.cache.clear();
        self.stats = SimplifyStats::default();
    }

    /// Simplify a term
    ///
    /// Returns a simplified version of the term, or the original if no simplification applies
    pub fn simplify(&mut self, term: TermId, manager: &mut TermManager) -> TermId {
        // Check cache first
        if let Some(&simplified) = self.cache.get(&term) {
            return simplified;
        }

        let result = self.simplify_impl(term, manager);
        self.cache.insert(term, result);
        result
    }

    fn simplify_impl(&mut self, term: TermId, manager: &mut TermManager) -> TermId {
        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        match &t.kind {
            // Boolean simplifications
            TermKind::True | TermKind::False => term,

            TermKind::Not(arg) => {
                let arg_simplified = self.simplify(*arg, manager);

                // not(true) => false
                if let Some(arg_term) = manager.get(arg_simplified) {
                    if matches!(arg_term.kind, TermKind::True) {
                        self.stats.const_propagations += 1;
                        return manager.mk_false();
                    }
                    // not(false) => true
                    if matches!(arg_term.kind, TermKind::False) {
                        self.stats.const_propagations += 1;
                        return manager.mk_true();
                    }
                    // not(not(x)) => x
                    if let TermKind::Not(inner) = arg_term.kind {
                        self.stats.terms_eliminated += 1;
                        return inner;
                    }
                }

                if arg_simplified == *arg {
                    term
                } else {
                    manager.mk_not(arg_simplified)
                }
            }

            TermKind::And(args) => {
                let mut simplified_args = Vec::new();
                let mut seen = FxHashMap::default();

                for &arg in args.iter() {
                    let simplified = self.simplify(arg, manager);

                    // and(..., false, ...) => false
                    if let Some(arg_term) = manager.get(simplified) {
                        if matches!(arg_term.kind, TermKind::False) {
                            self.stats.const_propagations += 1;
                            return manager.mk_false();
                        }
                        // Skip true literals
                        if matches!(arg_term.kind, TermKind::True) {
                            self.stats.terms_eliminated += 1;
                            continue;
                        }

                        // Flatten nested ANDs: and(and(a, b), c) => and(a, b, c)
                        if let TermKind::And(nested_args) = &arg_term.kind {
                            self.stats.operations_flattened += 1;
                            let nested_args_cloned = nested_args.clone();
                            for &nested_arg in &nested_args_cloned {
                                let nested_simplified = self.simplify(nested_arg, manager);
                                if !seen.contains_key(&nested_simplified) {
                                    seen.insert(nested_simplified, ());
                                    simplified_args.push(nested_simplified);
                                } else {
                                    self.stats.duplicates_eliminated += 1;
                                }
                            }
                            continue;
                        }
                    }

                    // Check for contradictions: and(x, not(x)) => false
                    if let Some(arg_term) = manager.get(simplified)
                        && let TermKind::Not(inner) = arg_term.kind
                        && (simplified_args.contains(&inner) || seen.contains_key(&inner))
                    {
                        self.stats.contradictions_found += 1;
                        return manager.mk_false();
                    }
                    // Check if we already have not(arg) in the list
                    let neg = manager.mk_not(simplified);
                    if simplified_args.contains(&neg) || seen.contains_key(&neg) {
                        self.stats.contradictions_found += 1;
                        return manager.mk_false();
                    }

                    // Eliminate duplicates
                    if !seen.contains_key(&simplified) {
                        seen.insert(simplified, ());
                        simplified_args.push(simplified);
                    } else {
                        self.stats.duplicates_eliminated += 1;
                    }
                }

                match simplified_args.len() {
                    0 => {
                        self.stats.const_propagations += 1;
                        manager.mk_true()
                    }
                    1 => {
                        self.stats.terms_eliminated += 1;
                        simplified_args[0]
                    }
                    _ => manager.mk_and(simplified_args),
                }
            }

            TermKind::Or(args) => {
                let mut simplified_args = Vec::new();
                let mut seen = FxHashMap::default();

                for &arg in args.iter() {
                    let simplified = self.simplify(arg, manager);

                    // or(..., true, ...) => true
                    if let Some(arg_term) = manager.get(simplified) {
                        if matches!(arg_term.kind, TermKind::True) {
                            self.stats.const_propagations += 1;
                            return manager.mk_true();
                        }
                        // Skip false literals
                        if matches!(arg_term.kind, TermKind::False) {
                            self.stats.terms_eliminated += 1;
                            continue;
                        }

                        // Flatten nested ORs: or(or(a, b), c) => or(a, b, c)
                        if let TermKind::Or(nested_args) = &arg_term.kind {
                            self.stats.operations_flattened += 1;
                            let nested_args_cloned = nested_args.clone();
                            for &nested_arg in &nested_args_cloned {
                                let nested_simplified = self.simplify(nested_arg, manager);
                                if !seen.contains_key(&nested_simplified) {
                                    seen.insert(nested_simplified, ());
                                    simplified_args.push(nested_simplified);
                                } else {
                                    self.stats.duplicates_eliminated += 1;
                                }
                            }
                            continue;
                        }
                    }

                    // Check for tautologies: or(x, not(x)) => true
                    if let Some(arg_term) = manager.get(simplified)
                        && let TermKind::Not(inner) = arg_term.kind
                        && (simplified_args.contains(&inner) || seen.contains_key(&inner))
                    {
                        self.stats.tautologies_detected += 1;
                        return manager.mk_true();
                    }
                    // Check if we already have not(arg) in the list
                    let neg = manager.mk_not(simplified);
                    if simplified_args.contains(&neg) || seen.contains_key(&neg) {
                        self.stats.tautologies_detected += 1;
                        return manager.mk_true();
                    }

                    // Eliminate duplicates
                    if !seen.contains_key(&simplified) {
                        seen.insert(simplified, ());
                        simplified_args.push(simplified);
                    } else {
                        self.stats.duplicates_eliminated += 1;
                    }
                }

                match simplified_args.len() {
                    0 => {
                        self.stats.const_propagations += 1;
                        manager.mk_false()
                    }
                    1 => {
                        self.stats.terms_eliminated += 1;
                        simplified_args[0]
                    }
                    _ => manager.mk_or(simplified_args),
                }
            }

            TermKind::Implies(lhs, rhs) => {
                let lhs_simplified = self.simplify(*lhs, manager);
                let rhs_simplified = self.simplify(*rhs, manager);

                // false => x  =  true
                if let Some(lhs_term) = manager.get(lhs_simplified)
                    && matches!(lhs_term.kind, TermKind::False)
                {
                    self.stats.const_propagations += 1;
                    return manager.mk_true();
                }

                // true => x  =  x
                if let Some(lhs_term) = manager.get(lhs_simplified)
                    && matches!(lhs_term.kind, TermKind::True)
                {
                    self.stats.terms_eliminated += 1;
                    return rhs_simplified;
                }

                // x => true  =  true
                if let Some(rhs_term) = manager.get(rhs_simplified)
                    && matches!(rhs_term.kind, TermKind::True)
                {
                    self.stats.const_propagations += 1;
                    return manager.mk_true();
                }

                // x => false  =  not(x)
                if let Some(rhs_term) = manager.get(rhs_simplified)
                    && matches!(rhs_term.kind, TermKind::False)
                {
                    self.stats.terms_eliminated += 1;
                    return manager.mk_not(lhs_simplified);
                }

                if lhs_simplified == *lhs && rhs_simplified == *rhs {
                    term
                } else {
                    manager.mk_implies(lhs_simplified, rhs_simplified)
                }
            }

            TermKind::Ite(cond, then_br, else_br) => {
                let cond_simplified = self.simplify(*cond, manager);
                let then_simplified = self.simplify(*then_br, manager);
                let else_simplified = self.simplify(*else_br, manager);

                // ite(true, x, y) => x
                if let Some(cond_term) = manager.get(cond_simplified)
                    && matches!(cond_term.kind, TermKind::True)
                {
                    self.stats.const_propagations += 1;
                    return then_simplified;
                }

                // ite(false, x, y) => y
                if let Some(cond_term) = manager.get(cond_simplified)
                    && matches!(cond_term.kind, TermKind::False)
                {
                    self.stats.const_propagations += 1;
                    return else_simplified;
                }

                // ite(c, x, x) => x
                if then_simplified == else_simplified {
                    self.stats.terms_eliminated += 1;
                    return then_simplified;
                }

                if cond_simplified == *cond
                    && then_simplified == *then_br
                    && else_simplified == *else_br
                {
                    term
                } else {
                    manager.mk_ite(cond_simplified, then_simplified, else_simplified)
                }
            }

            TermKind::Eq(lhs, rhs) => {
                let lhs_simplified = self.simplify(*lhs, manager);
                let rhs_simplified = self.simplify(*rhs, manager);

                // x = x  =>  true
                if lhs_simplified == rhs_simplified {
                    self.stats.trivial_equalities += 1;
                    return manager.mk_true();
                }

                // Check for contradictory constants
                if let (Some(lhs_term), Some(rhs_term)) =
                    (manager.get(lhs_simplified), manager.get(rhs_simplified))
                {
                    match (&lhs_term.kind, &rhs_term.kind) {
                        (TermKind::True, TermKind::False) | (TermKind::False, TermKind::True) => {
                            self.stats.contradictions_found += 1;
                            return manager.mk_false();
                        }
                        (TermKind::True, TermKind::True) | (TermKind::False, TermKind::False) => {
                            self.stats.trivial_equalities += 1;
                            return manager.mk_true();
                        }
                        _ => {}
                    }
                }

                if lhs_simplified == *lhs && rhs_simplified == *rhs {
                    term
                } else {
                    manager.mk_eq(lhs_simplified, rhs_simplified)
                }
            }

            // For other term kinds, just return the original
            _ => term,
        }
    }

    /// Simplify multiple assertions
    ///
    /// Returns simplified versions of all assertions and a flag indicating
    /// if a contradiction was found
    #[allow(dead_code)]
    pub fn simplify_assertions(
        &mut self,
        assertions: &[TermId],
        manager: &mut TermManager,
    ) -> (Vec<TermId>, bool) {
        let mut simplified = Vec::new();
        let mut found_false = false;

        for &assertion in assertions {
            let simp = self.simplify(assertion, manager);

            // Check if we found false
            if let Some(term) = manager.get(simp) {
                if matches!(term.kind, TermKind::False) {
                    found_false = true;
                }
                // Skip true assertions (they don't constrain anything)
                if matches!(term.kind, TermKind::True) {
                    continue;
                }
            }

            simplified.push(simp);
        }

        (simplified, found_false)
    }

    /// Apply unit propagation at preprocessing level
    /// Returns simplified assertions after propagating unit clauses
    #[allow(dead_code)]
    pub fn unit_propagation(
        &mut self,
        assertions: &[TermId],
        manager: &mut TermManager,
    ) -> Vec<TermId> {
        let mut units = FxHashMap::default(); // Map from term to its assigned value (true/false)
        let mut result = Vec::new();

        // First pass: collect unit clauses (single literals)
        for &assertion in assertions {
            if let Some(term) = manager.get(assertion) {
                match &term.kind {
                    TermKind::True | TermKind::False => {
                        // Already handled by simplification
                        result.push(assertion);
                    }
                    TermKind::Not(inner) => {
                        // Unit clause: not(x)
                        units.insert(*inner, false);
                        result.push(assertion);
                    }
                    _ => {
                        // Check if it's a variable (also a unit clause)
                        if matches!(term.kind, TermKind::Var(_)) {
                            units.insert(assertion, true);
                        }
                        result.push(assertion);
                    }
                }
            } else {
                result.push(assertion);
            }
        }

        // If we found unit clauses, propagate them
        if !units.is_empty() {
            self.stats.const_propagations += units.len();
            result = result
                .into_iter()
                .map(|term| self.substitute_units(term, &units, manager))
                .collect();
        }

        result
    }

    /// Substitute unit assignments in a term
    fn substitute_units(
        &mut self,
        term: TermId,
        units: &FxHashMap<TermId, bool>,
        manager: &mut TermManager,
    ) -> TermId {
        // Check if this term has a unit assignment
        if let Some(&value) = units.get(&term) {
            return if value {
                manager.mk_true()
            } else {
                manager.mk_false()
            };
        }

        // Recursively substitute in subterms
        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        match &t.kind {
            TermKind::Not(arg) => {
                let arg_subst = self.substitute_units(*arg, units, manager);
                if arg_subst == *arg {
                    term
                } else {
                    manager.mk_not(arg_subst)
                }
            }
            TermKind::And(args) => {
                let mut changed = false;
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| {
                        let subst = self.substitute_units(arg, units, manager);
                        if subst != arg {
                            changed = true;
                        }
                        subst
                    })
                    .collect();
                if changed {
                    manager.mk_and(new_args)
                } else {
                    term
                }
            }
            TermKind::Or(args) => {
                let mut changed = false;
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| {
                        let subst = self.substitute_units(arg, units, manager);
                        if subst != arg {
                            changed = true;
                        }
                        subst
                    })
                    .collect();
                if changed {
                    manager.mk_or(new_args)
                } else {
                    term
                }
            }
            _ => term,
        }
    }

    /// Detect pure literals (literals that appear only in one polarity)
    /// Returns a map from pure literals to their polarity (true = positive, false = negative)
    #[allow(dead_code)]
    pub fn detect_pure_literals(
        &self,
        assertions: &[TermId],
        manager: &TermManager,
    ) -> FxHashMap<TermId, bool> {
        let mut positive = FxHashMap::default();
        let mut negative = FxHashMap::default();

        // Collect all literal occurrences
        for &assertion in assertions {
            self.collect_literals(assertion, true, &mut positive, &mut negative, manager);
        }

        // Find pure literals (appear only in one polarity)
        let mut pure_literals = FxHashMap::default();
        for (&lit, _) in &positive {
            if !negative.contains_key(&lit) {
                pure_literals.insert(lit, true);
            }
        }
        for (&lit, _) in &negative {
            if !positive.contains_key(&lit) {
                pure_literals.insert(lit, false);
            }
        }

        pure_literals
    }

    /// Collect literal occurrences with their polarities
    fn collect_literals(
        &self,
        term: TermId,
        polarity: bool,
        positive: &mut FxHashMap<TermId, ()>,
        negative: &mut FxHashMap<TermId, ()>,
        manager: &TermManager,
    ) {
        let Some(t) = manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::Var(_) => {
                if polarity {
                    positive.insert(term, ());
                } else {
                    negative.insert(term, ());
                }
            }
            TermKind::Not(arg) => {
                self.collect_literals(*arg, !polarity, positive, negative, manager);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    self.collect_literals(arg, polarity, positive, negative, manager);
                }
            }
            TermKind::Implies(lhs, rhs) => {
                self.collect_literals(*lhs, !polarity, positive, negative, manager);
                self.collect_literals(*rhs, polarity, positive, negative, manager);
            }
            TermKind::Ite(cond, then_br, else_br) => {
                // For ITE, both branches can be reached
                self.collect_literals(*cond, true, positive, negative, manager);
                self.collect_literals(*cond, false, positive, negative, manager);
                self.collect_literals(*then_br, polarity, positive, negative, manager);
                self.collect_literals(*else_br, polarity, positive, negative, manager);
            }
            _ => {}
        }
    }
}

impl Default for Simplifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_not() {
        let mut manager = TermManager::new();
        let mut simplifier = Simplifier::new();

        // not(true) => false
        let t = manager.mk_true();
        let not_t = manager.mk_not(t);
        let result = simplifier.simplify(not_t, &mut manager);
        assert!(matches!(manager.get(result).unwrap().kind, TermKind::False));

        // not(false) => true
        let f = manager.mk_false();
        let not_f = manager.mk_not(f);
        let result = simplifier.simplify(not_f, &mut manager);
        assert!(matches!(manager.get(result).unwrap().kind, TermKind::True));
    }

    #[test]
    fn test_simplify_and() {
        let mut manager = TermManager::new();
        let mut simplifier = Simplifier::new();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let f = manager.mk_false();

        // and(x, true) => x
        let and_x_true = manager.mk_and([x, t]);
        let result = simplifier.simplify(and_x_true, &mut manager);
        assert_eq!(result, x);

        // and(x, false) => false
        let and_x_false = manager.mk_and([x, f]);
        let result = simplifier.simplify(and_x_false, &mut manager);
        assert!(matches!(manager.get(result).unwrap().kind, TermKind::False));
    }

    #[test]
    fn test_simplify_or() {
        let mut manager = TermManager::new();
        let mut simplifier = Simplifier::new();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let f = manager.mk_false();

        // or(x, false) => x
        let or_x_false = manager.mk_or([x, f]);
        let result = simplifier.simplify(or_x_false, &mut manager);
        assert_eq!(result, x);

        // or(x, true) => true
        let or_x_true = manager.mk_or([x, t]);
        let result = simplifier.simplify(or_x_true, &mut manager);
        assert!(matches!(manager.get(result).unwrap().kind, TermKind::True));
    }

    #[test]
    fn test_simplify_implies() {
        let mut manager = TermManager::new();
        let mut simplifier = Simplifier::new();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let f = manager.mk_false();

        // false => x  =  true
        let imp = manager.mk_implies(f, x);
        let result = simplifier.simplify(imp, &mut manager);
        assert!(matches!(manager.get(result).unwrap().kind, TermKind::True));

        // true => x  =  x
        let imp = manager.mk_implies(t, x);
        let result = simplifier.simplify(imp, &mut manager);
        assert_eq!(result, x);
    }

    #[test]
    fn test_simplify_ite() {
        let mut manager = TermManager::new();
        let mut simplifier = Simplifier::new();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let f = manager.mk_false();

        // ite(true, x, y) => x
        let ite = manager.mk_ite(t, x, y);
        let result = simplifier.simplify(ite, &mut manager);
        assert_eq!(result, x);

        // ite(false, x, y) => y
        let ite = manager.mk_ite(f, x, y);
        let result = simplifier.simplify(ite, &mut manager);
        assert_eq!(result, y);

        // ite(cond, x, x) => x
        let ite = manager.mk_ite(x, y, y);
        let result = simplifier.simplify(ite, &mut manager);
        assert_eq!(result, y);
    }

    #[test]
    fn test_simplify_eq() {
        let mut manager = TermManager::new();
        let mut simplifier = Simplifier::new();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let f = manager.mk_false();

        // x = x  =>  true
        let eq = manager.mk_eq(x, x);
        let result = simplifier.simplify(eq, &mut manager);
        assert!(matches!(manager.get(result).unwrap().kind, TermKind::True));

        // true = false  =>  false
        let eq = manager.mk_eq(t, f);
        let result = simplifier.simplify(eq, &mut manager);
        assert!(matches!(manager.get(result).unwrap().kind, TermKind::False));
    }

    #[test]
    fn test_simplify_assertions() {
        let mut manager = TermManager::new();
        let mut simplifier = Simplifier::new();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let f = manager.mk_false();

        // Simplify a list of assertions
        let assertions = vec![manager.mk_and([x, t]), manager.mk_or([x, f])];
        let (simplified, found_false) = simplifier.simplify_assertions(&assertions, &mut manager);

        assert!(!found_false);
        assert_eq!(simplified.len(), 2);
        assert_eq!(simplified[0], x); // and(x, true) => x
        assert_eq!(simplified[1], x); // or(x, false) => x

        // Test with a false assertion
        let assertions_with_false = vec![x, f];
        let (_, found_false) = simplifier.simplify_assertions(&assertions_with_false, &mut manager);
        assert!(found_false);
    }

    #[test]
    fn test_simplifier_reset() {
        let mut manager = TermManager::new();
        let mut simplifier = Simplifier::new();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);

        // Perform a simplification to populate the cache
        let eq = manager.mk_eq(x, x);
        let result1 = simplifier.simplify(eq, &mut manager);
        assert!(matches!(manager.get(result1).unwrap().kind, TermKind::True));

        // Create another term that would be cached
        let eq2 = manager.mk_eq(y, y);
        let result2 = simplifier.simplify(eq2, &mut manager);
        assert!(matches!(manager.get(result2).unwrap().kind, TermKind::True));

        // Reset the simplifier
        simplifier.reset();

        // Verify stats are cleared
        let stats_after_reset = simplifier.stats();
        assert_eq!(stats_after_reset.const_propagations, 0);
        assert_eq!(stats_after_reset.terms_eliminated, 0);
        assert_eq!(stats_after_reset.trivial_equalities, 0);
        assert_eq!(stats_after_reset.contradictions_found, 0);
    }
}
