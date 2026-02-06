//! Macro Support for MBQI
//!
//! This module provides utility macros and helper functions for MBQI implementation.
//! It includes macros for common patterns, debugging, and code generation.

/// Macro for creating a quantified formula with default parameters
#[macro_export]
macro_rules! quantifier {
    ($term:expr, $vars:expr, $body:expr, universal) => {
        $crate::mbqi::QuantifiedFormula::new($term, $vars, $body, true)
    };
    ($term:expr, $vars:expr, $body:expr, existential) => {
        $crate::mbqi::QuantifiedFormula::new($term, $vars, $body, false)
    };
}

/// Macro for creating an instantiation
#[macro_export]
macro_rules! instantiation {
    ($quantifier:expr, $subst:expr, $result:expr, $gen:expr) => {
        $crate::mbqi::Instantiation::new($quantifier, $subst, $result, $gen)
    };
}

/// Macro for debugging MBQI state
#[macro_export]
macro_rules! mbqi_debug {
    ($($arg:tt)*) => {
        #[cfg(feature = "mbqi-debug")]
        {
            eprintln!("[MBQI DEBUG] {}", format!($($arg)*));
        }
    };
}

/// Macro for MBQI tracing
#[macro_export]
macro_rules! mbqi_trace {
    ($($arg:tt)*) => {
        #[cfg(feature = "mbqi-trace")]
        {
            eprintln!("[MBQI TRACE] {}", format!($($arg)*));
        }
    };
}

/// Macro for timing MBQI operations
#[macro_export]
macro_rules! mbqi_time {
    ($name:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let elapsed = start.elapsed();
        mbqi_debug!("{} took {:?}", $name, elapsed);
        result
    }};
}

/// Macro for creating a model completion error
#[macro_export]
macro_rules! completion_error {
    ($msg:expr) => {
        $crate::mbqi::model_completion::CompletionError::CompletionFailed($msg.to_string())
    };
}

/// Macro for creating a finder error
#[macro_export]
macro_rules! finder_error {
    (unsat) => {
        $crate::mbqi::finite_model::FinderError::UnsatAtBound
    };
    (max_bound) => {
        $crate::mbqi::finite_model::FinderError::ExceededMaxBound
    };
    (resource) => {
        $crate::mbqi::finite_model::FinderError::ResourceLimit
    };
}

/// Utility functions for MBQI
pub mod utils {
    use lasso::Spur;
    use oxiz_core::ast::{TermId, TermKind, TermManager};
    use rustc_hash::{FxHashMap, FxHashSet};

    /// Check if a term is ground (no free variables)
    pub fn is_ground(term: TermId, manager: &TermManager) -> bool {
        let mut visited = FxHashSet::default();
        is_ground_rec(term, manager, &mut visited)
    }

    fn is_ground_rec(term: TermId, manager: &TermManager, visited: &mut FxHashSet<TermId>) -> bool {
        if visited.contains(&term) {
            return true;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return true;
        };

        if matches!(t.kind, TermKind::Var(_)) {
            return false;
        }

        match &t.kind {
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    if !is_ground_rec(arg, manager, visited) {
                        return false;
                    }
                }
                true
            }
            TermKind::Not(arg) | TermKind::Neg(arg) => is_ground_rec(*arg, manager, visited),
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    if !is_ground_rec(arg, manager, visited) {
                        return false;
                    }
                }
                true
            }
            _ => true,
        }
    }

    /// Collect all free variables in a term
    pub fn free_vars(term: TermId, manager: &TermManager) -> FxHashSet<Spur> {
        let mut vars = FxHashSet::default();
        let mut visited = FxHashSet::default();
        collect_vars_rec(term, &mut vars, &mut visited, manager);
        vars
    }

    fn collect_vars_rec(
        term: TermId,
        vars: &mut FxHashSet<Spur>,
        visited: &mut FxHashSet<TermId>,
        manager: &TermManager,
    ) {
        if visited.contains(&term) {
            return;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return;
        };

        if let TermKind::Var(name) = t.kind {
            vars.insert(name);
            return;
        }

        match &t.kind {
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    collect_vars_rec(arg, vars, visited, manager);
                }
            }
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                collect_vars_rec(*arg, vars, visited, manager);
            }
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    collect_vars_rec(arg, vars, visited, manager);
                }
            }
            _ => {}
        }
    }

    /// Substitute variables in a term
    pub fn substitute(
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        let mut cache = FxHashMap::default();
        substitute_cached(term, subst, manager, &mut cache)
    }

    fn substitute_cached(
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        if let Some(&cached) = cache.get(&term) {
            return cached;
        }

        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        let result = match &t.kind {
            TermKind::Var(name) => subst.get(name).copied().unwrap_or(term),
            TermKind::Not(arg) => {
                let new_arg = substitute_cached(*arg, subst, manager, cache);
                manager.mk_not(new_arg)
            }
            TermKind::And(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| substitute_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_and(new_args)
            }
            TermKind::Or(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| substitute_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_or(new_args)
            }
            _ => term,
        };

        cache.insert(term, result);
        result
    }

    /// Calculate term depth
    pub fn term_depth(term: TermId, manager: &TermManager) -> usize {
        let mut visited = FxHashMap::default();
        term_depth_cached(term, manager, &mut visited)
    }

    fn term_depth_cached(
        term: TermId,
        manager: &TermManager,
        visited: &mut FxHashMap<TermId, usize>,
    ) -> usize {
        if let Some(&cached) = visited.get(&term) {
            return cached;
        }

        let Some(t) = manager.get(term) else {
            return 1;
        };

        let depth = match &t.kind {
            TermKind::And(args) | TermKind::Or(args) => {
                let max_child = args
                    .iter()
                    .map(|&arg| term_depth_cached(arg, manager, visited))
                    .max()
                    .unwrap_or(0);
                1 + max_child
            }
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                1 + term_depth_cached(*arg, manager, visited)
            }
            TermKind::Apply { args, .. } => {
                let max_child = args
                    .iter()
                    .map(|&arg| term_depth_cached(arg, manager, visited))
                    .max()
                    .unwrap_or(0);
                1 + max_child
            }
            _ => 1,
        };

        visited.insert(term, depth);
        depth
    }

    /// Calculate term size (number of nodes)
    pub fn term_size(term: TermId, manager: &TermManager) -> usize {
        let mut visited = FxHashSet::default();
        term_size_rec(term, manager, &mut visited)
    }

    fn term_size_rec(
        term: TermId,
        manager: &TermManager,
        visited: &mut FxHashSet<TermId>,
    ) -> usize {
        if visited.contains(&term) {
            return 0;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return 1;
        };

        let children_size = match &t.kind {
            TermKind::And(args) | TermKind::Or(args) => args
                .iter()
                .map(|&arg| term_size_rec(arg, manager, visited))
                .sum(),
            TermKind::Not(arg) | TermKind::Neg(arg) => term_size_rec(*arg, manager, visited),
            TermKind::Apply { args, .. } => args
                .iter()
                .map(|&arg| term_size_rec(arg, manager, visited))
                .sum(),
            _ => 0,
        };

        1 + children_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::ast::TermManager;

    #[test]
    fn test_is_ground_constant() {
        let manager = TermManager::new();
        let term = manager.mk_true();
        assert!(utils::is_ground(term, &manager));
    }

    #[test]
    fn test_term_depth_constant() {
        let manager = TermManager::new();
        let term = manager.mk_true();
        assert_eq!(utils::term_depth(term, &manager), 1);
    }

    #[test]
    fn test_term_size_constant() {
        let manager = TermManager::new();
        let term = manager.mk_true();
        assert_eq!(utils::term_size(term, &manager), 1);
    }

    #[test]
    fn test_free_vars_constant() {
        let manager = TermManager::new();
        let term = manager.mk_true();
        let vars = utils::free_vars(term, &manager);
        assert_eq!(vars.len(), 0);
    }
}
