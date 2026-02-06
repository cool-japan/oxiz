//! Efficient substitution with structural sharing
//!
//! This module provides optimized term substitution for E-matching instantiations.
//! Substitutions map bound variables to ground terms, and are applied when
//! instantiating quantified formulas.
//!
//! # Design Principles
//!
//! - **Structural Sharing**: Reuse unchanged subterms to minimize allocations
//! - **Caching**: Cache substitution results to avoid redundant work
//! - **Incremental**: Support incremental substitution updates
//!
//! # Algorithm
//!
//! Based on Z3's substitution implementation in src/ast/substitution.cpp

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::{OxizError, Result};
use lasso::Spur;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::fmt;

/// A substitution mapping variables to terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Substitution {
    /// Variable name to term mapping
    bindings: FxHashMap<Spur, TermId>,
}

impl Substitution {
    /// Create an empty substitution
    pub fn new() -> Self {
        Self {
            bindings: FxHashMap::default(),
        }
    }

    /// Create a substitution with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bindings: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Insert a binding
    pub fn insert(&mut self, var: Spur, term: TermId) {
        self.bindings.insert(var, term);
    }

    /// Get a binding
    pub fn get(&self, var: &Spur) -> Option<TermId> {
        self.bindings.get(var).copied()
    }

    /// Check if a variable is bound
    pub fn contains(&self, var: &Spur) -> bool {
        self.bindings.contains_key(var)
    }

    /// Get the number of bindings
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if the substitution is empty
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Iterate over bindings
    pub fn iter(&self) -> impl Iterator<Item = (&Spur, &TermId)> {
        self.bindings.iter()
    }

    /// Clear all bindings
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Apply this substitution to a term
    pub fn apply(&self, term: TermId, manager: &mut TermManager) -> Result<TermId> {
        self.apply_recursive(term, manager)
    }

    /// Recursive substitution implementation
    fn apply_recursive(&self, term: TermId, manager: &mut TermManager) -> Result<TermId> {
        let (kind, sort) = {
            let Some(t) = manager.get(term) else {
                return Err(OxizError::EmatchError(format!(
                    "Term {:?} not found in manager",
                    term
                )));
            };
            (t.kind.clone(), t.sort)
        };

        match &kind {
            // Variable: check if it's in our substitution
            TermKind::Var(name) => {
                if let Some(replacement) = self.bindings.get(name) {
                    Ok(*replacement)
                } else {
                    Ok(term) // Not bound, return as-is
                }
            }

            // Ground terms: return as-is (no substitution needed)
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => Ok(term),

            // Recursive cases: apply substitution to children
            TermKind::Apply { func, args } => {
                let mut changed = false;
                let new_args: Result<SmallVec<[TermId; 4]>> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.apply_recursive(arg, manager)?;
                        if new_arg != arg {
                            changed = true;
                        }
                        Ok(new_arg)
                    })
                    .collect();
                let new_args = new_args?;

                if changed {
                    let func_name = manager.resolve_str(*func).to_string();
                    Ok(manager.mk_apply(&func_name, new_args, sort))
                } else {
                    Ok(term)
                }
            }

            TermKind::Eq(lhs, rhs) => {
                let new_lhs = self.apply_recursive(*lhs, manager)?;
                let new_rhs = self.apply_recursive(*rhs, manager)?;
                if new_lhs != *lhs || new_rhs != *rhs {
                    Ok(manager.mk_eq(new_lhs, new_rhs))
                } else {
                    Ok(term)
                }
            }

            TermKind::Lt(lhs, rhs) => {
                let new_lhs = self.apply_recursive(*lhs, manager)?;
                let new_rhs = self.apply_recursive(*rhs, manager)?;
                if new_lhs != *lhs || new_rhs != *rhs {
                    Ok(manager.mk_lt(new_lhs, new_rhs))
                } else {
                    Ok(term)
                }
            }

            TermKind::Le(lhs, rhs) => {
                let new_lhs = self.apply_recursive(*lhs, manager)?;
                let new_rhs = self.apply_recursive(*rhs, manager)?;
                if new_lhs != *lhs || new_rhs != *rhs {
                    Ok(manager.mk_le(new_lhs, new_rhs))
                } else {
                    Ok(term)
                }
            }

            TermKind::Gt(lhs, rhs) => {
                let new_lhs = self.apply_recursive(*lhs, manager)?;
                let new_rhs = self.apply_recursive(*rhs, manager)?;
                if new_lhs != *lhs || new_rhs != *rhs {
                    Ok(manager.mk_gt(new_lhs, new_rhs))
                } else {
                    Ok(term)
                }
            }

            TermKind::Ge(lhs, rhs) => {
                let new_lhs = self.apply_recursive(*lhs, manager)?;
                let new_rhs = self.apply_recursive(*rhs, manager)?;
                if new_lhs != *lhs || new_rhs != *rhs {
                    Ok(manager.mk_ge(new_lhs, new_rhs))
                } else {
                    Ok(term)
                }
            }

            TermKind::Sub(lhs, rhs) => {
                let new_lhs = self.apply_recursive(*lhs, manager)?;
                let new_rhs = self.apply_recursive(*rhs, manager)?;
                if new_lhs != *lhs || new_rhs != *rhs {
                    Ok(manager.mk_sub(new_lhs, new_rhs))
                } else {
                    Ok(term)
                }
            }

            TermKind::Div(lhs, rhs) => {
                let new_lhs = self.apply_recursive(*lhs, manager)?;
                let new_rhs = self.apply_recursive(*rhs, manager)?;
                if new_lhs != *lhs || new_rhs != *rhs {
                    Ok(manager.mk_div(new_lhs, new_rhs))
                } else {
                    Ok(term)
                }
            }

            TermKind::Add(args) => {
                let mut changed = false;
                let new_args: Result<SmallVec<[TermId; 4]>> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.apply_recursive(arg, manager)?;
                        if new_arg != arg {
                            changed = true;
                        }
                        Ok(new_arg)
                    })
                    .collect();
                let new_args = new_args?;

                if changed {
                    Ok(manager.mk_add(new_args))
                } else {
                    Ok(term)
                }
            }

            TermKind::Mul(args) => {
                let mut changed = false;
                let new_args: Result<SmallVec<[TermId; 4]>> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.apply_recursive(arg, manager)?;
                        if new_arg != arg {
                            changed = true;
                        }
                        Ok(new_arg)
                    })
                    .collect();
                let new_args = new_args?;

                if changed {
                    Ok(manager.mk_mul(new_args))
                } else {
                    Ok(term)
                }
            }

            TermKind::And(args) => {
                let mut changed = false;
                let new_args: Result<SmallVec<[TermId; 4]>> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.apply_recursive(arg, manager)?;
                        if new_arg != arg {
                            changed = true;
                        }
                        Ok(new_arg)
                    })
                    .collect();
                let new_args = new_args?;

                if changed {
                    Ok(manager.mk_and(new_args))
                } else {
                    Ok(term)
                }
            }

            TermKind::Or(args) => {
                let mut changed = false;
                let new_args: Result<SmallVec<[TermId; 4]>> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.apply_recursive(arg, manager)?;
                        if new_arg != arg {
                            changed = true;
                        }
                        Ok(new_arg)
                    })
                    .collect();
                let new_args = new_args?;

                if changed {
                    Ok(manager.mk_or(new_args))
                } else {
                    Ok(term)
                }
            }

            TermKind::Not(inner) => {
                let new_inner = self.apply_recursive(*inner, manager)?;
                if new_inner != *inner {
                    Ok(manager.mk_not(new_inner))
                } else {
                    Ok(term)
                }
            }

            TermKind::Neg(inner) => {
                let new_inner = self.apply_recursive(*inner, manager)?;
                if new_inner != *inner {
                    Ok(manager.mk_neg(new_inner))
                } else {
                    Ok(term)
                }
            }

            TermKind::Implies(lhs, rhs) => {
                let new_lhs = self.apply_recursive(*lhs, manager)?;
                let new_rhs = self.apply_recursive(*rhs, manager)?;
                if new_lhs != *lhs || new_rhs != *rhs {
                    Ok(manager.mk_implies(new_lhs, new_rhs))
                } else {
                    Ok(term)
                }
            }

            TermKind::Ite(c, t_branch, e_branch) => {
                let new_c = self.apply_recursive(*c, manager)?;
                let new_t = self.apply_recursive(*t_branch, manager)?;
                let new_e = self.apply_recursive(*e_branch, manager)?;
                if new_c != *c || new_t != *t_branch || new_e != *e_branch {
                    Ok(manager.mk_ite(new_c, new_t, new_e))
                } else {
                    Ok(term)
                }
            }

            TermKind::Select(arr, idx) => {
                let new_arr = self.apply_recursive(*arr, manager)?;
                let new_idx = self.apply_recursive(*idx, manager)?;
                if new_arr != *arr || new_idx != *idx {
                    Ok(manager.mk_select(new_arr, new_idx))
                } else {
                    Ok(term)
                }
            }

            TermKind::Store(arr, idx, val) => {
                let new_arr = self.apply_recursive(*arr, manager)?;
                let new_idx = self.apply_recursive(*idx, manager)?;
                let new_val = self.apply_recursive(*val, manager)?;
                if new_arr != *arr || new_idx != *idx || new_val != *val {
                    Ok(manager.mk_store(new_arr, new_idx, new_val))
                } else {
                    Ok(term)
                }
            }

            // For quantifiers, be careful about variable capture
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                // Check if any of the bound variables shadow our substitution
                let shadowed: FxHashMap<Spur, TermId> = self
                    .bindings
                    .iter()
                    .filter(|(name, _)| !vars.iter().any(|(v, _)| v == *name))
                    .map(|(&k, &v)| (k, v))
                    .collect();

                if shadowed.len() < self.bindings.len() {
                    // Some variables are shadowed, create a new substitution
                    let sub = Substitution { bindings: shadowed };
                    let new_body = sub.apply_recursive(*body, manager)?;
                    let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                        .iter()
                        .map(|pattern| {
                            pattern
                                .iter()
                                .map(|&p| sub.apply_recursive(p, manager))
                                .collect::<Result<SmallVec<[TermId; 2]>>>()
                        })
                        .collect::<Result<_>>()?;

                    let var_names: Vec<_> = vars
                        .iter()
                        .map(|(n, s)| (manager.resolve_str(*n).to_string(), *s))
                        .collect();
                    let var_strs: Vec<_> = var_names
                        .iter()
                        .map(|(name, sort)| (name.as_str(), *sort))
                        .collect();

                    Ok(manager.mk_forall_with_patterns(var_strs, new_body, new_patterns))
                } else {
                    // No shadowing
                    let new_body = self.apply_recursive(*body, manager)?;
                    if new_body != *body {
                        let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                            .iter()
                            .map(|pattern| {
                                pattern
                                    .iter()
                                    .map(|&p| self.apply_recursive(p, manager))
                                    .collect::<Result<SmallVec<[TermId; 2]>>>()
                            })
                            .collect::<Result<_>>()?;

                        let var_names: Vec<_> = vars
                            .iter()
                            .map(|(n, s)| (manager.resolve_str(*n).to_string(), *s))
                            .collect();
                        let var_strs: Vec<_> = var_names
                            .iter()
                            .map(|(name, sort)| (name.as_str(), *sort))
                            .collect();

                        Ok(manager.mk_forall_with_patterns(var_strs, new_body, new_patterns))
                    } else {
                        Ok(term)
                    }
                }
            }

            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                // Similar handling for exists
                let shadowed: FxHashMap<Spur, TermId> = self
                    .bindings
                    .iter()
                    .filter(|(name, _)| !vars.iter().any(|(v, _)| v == *name))
                    .map(|(&k, &v)| (k, v))
                    .collect();

                if shadowed.len() < self.bindings.len() {
                    let sub = Substitution { bindings: shadowed };
                    let new_body = sub.apply_recursive(*body, manager)?;
                    let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                        .iter()
                        .map(|pattern| {
                            pattern
                                .iter()
                                .map(|&p| sub.apply_recursive(p, manager))
                                .collect::<Result<SmallVec<[TermId; 2]>>>()
                        })
                        .collect::<Result<_>>()?;

                    let var_names: Vec<_> = vars
                        .iter()
                        .map(|(n, s)| (manager.resolve_str(*n).to_string(), *s))
                        .collect();
                    let var_strs: Vec<_> = var_names
                        .iter()
                        .map(|(name, sort)| (name.as_str(), *sort))
                        .collect();

                    Ok(manager.mk_exists_with_patterns(var_strs, new_body, new_patterns))
                } else {
                    let new_body = self.apply_recursive(*body, manager)?;
                    if new_body != *body {
                        let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                            .iter()
                            .map(|pattern| {
                                pattern
                                    .iter()
                                    .map(|&p| self.apply_recursive(p, manager))
                                    .collect::<Result<SmallVec<[TermId; 2]>>>()
                            })
                            .collect::<Result<_>>()?;

                        let var_names: Vec<_> = vars
                            .iter()
                            .map(|(n, s)| (manager.resolve_str(*n).to_string(), *s))
                            .collect();
                        let var_strs: Vec<_> = var_names
                            .iter()
                            .map(|(name, sort)| (name.as_str(), *sort))
                            .collect();

                        Ok(manager.mk_exists_with_patterns(var_strs, new_body, new_patterns))
                    } else {
                        Ok(term)
                    }
                }
            }

            // For other terms, return as-is (may need to add more cases later)
            _ => Ok(term),
        }
    }
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Substitution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (var, term)) in self.bindings.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?} -> {:?}", var, term)?;
        }
        write!(f, "}}")
    }
}

/// Builder for constructing substitutions
#[derive(Debug)]
pub struct SubstitutionBuilder {
    subst: Substitution,
}

impl SubstitutionBuilder {
    /// Create a new substitution builder
    pub fn new() -> Self {
        Self {
            subst: Substitution::new(),
        }
    }

    /// Create with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            subst: Substitution::with_capacity(capacity),
        }
    }

    /// Add a binding
    pub fn bind(mut self, var: Spur, term: TermId) -> Self {
        self.subst.insert(var, term);
        self
    }

    /// Build the substitution
    pub fn build(self) -> Substitution {
        self.subst
    }
}

impl Default for SubstitutionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for substitution
#[derive(Debug, Clone)]
pub struct SubstitutionConfig {
    /// Whether to enable caching of substitution results
    pub enable_cache: bool,
    /// Maximum cache size (0 = unlimited)
    pub max_cache_size: usize,
}

impl Default for SubstitutionConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            max_cache_size: 10000,
        }
    }
}

/// Cache for substitution results
#[derive(Debug)]
pub struct SubstitutionCache {
    /// Configuration
    config: SubstitutionConfig,
    /// Cache mapping (term, subst_hash) to result
    cache: FxHashMap<(TermId, u64), TermId>,
    /// Statistics
    stats: SubstitutionStats,
}

/// Statistics about substitutions
#[derive(Debug, Clone, Default)]
pub struct SubstitutionStats {
    /// Number of substitutions applied
    pub substitutions_applied: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of terms substituted
    pub terms_substituted: usize,
}

impl SubstitutionCache {
    /// Create a new substitution cache
    pub fn new(config: SubstitutionConfig) -> Self {
        Self {
            config,
            cache: FxHashMap::default(),
            stats: SubstitutionStats::default(),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(SubstitutionConfig::default())
    }

    /// Apply a substitution with caching
    pub fn apply(
        &mut self,
        term: TermId,
        subst: &Substitution,
        manager: &mut TermManager,
    ) -> Result<TermId> {
        self.stats.substitutions_applied += 1;

        if !self.config.enable_cache {
            return subst.apply(term, manager);
        }

        // Compute a hash of the substitution for caching
        let subst_hash = self.hash_substitution(subst);
        let key = (term, subst_hash);

        // Check cache
        if let Some(&result) = self.cache.get(&key) {
            self.stats.cache_hits += 1;
            return Ok(result);
        }

        self.stats.cache_misses += 1;

        // Apply substitution
        let result = subst.apply(term, manager)?;

        // Cache result if within limits
        if self.config.max_cache_size == 0 || self.cache.len() < self.config.max_cache_size {
            self.cache.insert(key, result);
        }

        Ok(result)
    }

    /// Hash a substitution for caching
    fn hash_substitution(&self, subst: &Substitution) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Sort bindings for consistent hashing
        let mut bindings: Vec<_> = subst.iter().map(|(&k, &v)| (k.into_inner(), v)).collect();
        bindings.sort_by_key(|(k, _)| *k);

        for (k, v) in bindings {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.stats = SubstitutionStats::default();
    }

    /// Get statistics
    pub fn stats(&self) -> &SubstitutionStats {
        &self.stats
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.stats.cache_hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    fn setup() -> TermManager {
        TermManager::new()
    }

    #[test]
    fn test_substitution_creation() {
        let subst = Substitution::new();
        assert!(subst.is_empty());
        assert_eq!(subst.len(), 0);
    }

    #[test]
    fn test_substitution_insert_get() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let x_name = manager.intern_str("x");
        let five = manager.mk_int(5);

        subst.insert(x_name, five);

        assert_eq!(subst.len(), 1);
        assert_eq!(subst.get(&x_name), Some(five));
        assert!(subst.contains(&x_name));
    }

    #[test]
    fn test_substitution_apply_variable() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        subst.insert(x_name, five);

        let result = subst.apply(x, &mut manager).unwrap();
        assert_eq!(result, five);
    }

    #[test]
    fn test_substitution_apply_non_bound_variable() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let int_sort = manager.sorts.int_sort;
        let y = manager.mk_var("y", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        subst.insert(x_name, five);

        // y is not bound, should remain unchanged
        let result = subst.apply(y, &mut manager).unwrap();
        assert_eq!(result, y);
    }

    #[test]
    fn test_substitution_apply_complex_term() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let sum = manager.mk_add([x, y]);

        let five = manager.mk_int(5);
        let x_name = manager.intern_str("x");
        subst.insert(x_name, five);

        // sum is (x + y), after substitution should be (5 + y)
        let result = subst.apply(sum, &mut manager).unwrap();

        // Verify result is not the same term
        assert_ne!(result, sum);

        // Verify it's structurally (5 + y)
        if let Some(result_term) = manager.get(result) {
            if let TermKind::Add(args) = &result_term.kind {
                assert!(args.contains(&five));
                assert!(args.contains(&y));
            } else {
                panic!("Expected Add term");
            }
        } else {
            panic!("Result term not found");
        }
    }

    #[test]
    fn test_substitution_builder() {
        let mut manager = setup();
        let x_name = manager.intern_str("x");
        let y_name = manager.intern_str("y");
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let subst = SubstitutionBuilder::new()
            .bind(x_name, five)
            .bind(y_name, ten)
            .build();

        assert_eq!(subst.len(), 2);
        assert_eq!(subst.get(&x_name), Some(five));
        assert_eq!(subst.get(&y_name), Some(ten));
    }

    #[test]
    fn test_substitution_config_default() {
        let config = SubstitutionConfig::default();
        assert!(config.enable_cache);
        assert_eq!(config.max_cache_size, 10000);
    }

    #[test]
    fn test_substitution_cache() {
        let mut manager = setup();
        let mut cache = SubstitutionCache::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        let subst = SubstitutionBuilder::new().bind(x_name, five).build();

        // First application - cache miss
        let result1 = cache.apply(x, &subst, &mut manager).unwrap();
        assert_eq!(cache.stats.cache_misses, 1);
        assert_eq!(cache.stats.cache_hits, 0);

        // Second application - cache hit
        let result2 = cache.apply(x, &subst, &mut manager).unwrap();
        assert_eq!(cache.stats.cache_hits, 1);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_substitution_structural_sharing() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let f_y = manager.mk_apply("f", [y], int_sort);
        let sum = manager.mk_add([x, f_y]);

        let five = manager.mk_int(5);
        let x_name = manager.intern_str("x");
        subst.insert(x_name, five);

        // Apply substitution: (x + f(y)) becomes (5 + f(y))
        let result = subst.apply(sum, &mut manager).unwrap();

        // f(y) should be shared (not re-created)
        if let Some(result_term) = manager.get(result)
            && let TermKind::Add(args) = &result_term.kind
        {
            // One of the args should be f(y) unchanged
            assert!(args.contains(&f_y));
        }
    }

    #[test]
    fn test_substitution_clear() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let x_name = manager.intern_str("x");
        let five = manager.mk_int(5);

        subst.insert(x_name, five);
        assert_eq!(subst.len(), 1);

        subst.clear();
        assert_eq!(subst.len(), 0);
        assert!(subst.is_empty());
    }

    #[test]
    fn test_substitution_ground_term_unchanged() {
        let mut manager = setup();
        let subst = Substitution::new();

        let five = manager.mk_int(5);
        let result = subst.apply(five, &mut manager).unwrap();

        // Ground term should remain unchanged
        assert_eq!(result, five);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut manager = setup();
        let mut cache = SubstitutionCache::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        let subst = SubstitutionBuilder::new().bind(x_name, five).build();

        // 1 miss, 2 hits
        cache.apply(x, &subst, &mut manager).unwrap();
        cache.apply(x, &subst, &mut manager).unwrap();
        cache.apply(x, &subst, &mut manager).unwrap();

        let hit_rate = cache.hit_rate();
        assert!((hit_rate - 2.0 / 3.0).abs() < 0.01);
    }
}
