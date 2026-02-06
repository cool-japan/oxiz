//! Pattern representation and compilation for E-matching
//!
//! This module provides the core pattern data structures and compilation logic
//! for E-matching. Patterns represent the structure of terms that need to be
//! matched against the E-graph.
//!
//! # Pattern Language
//!
//! Patterns can contain:
//! - **Variables**: Bound quantifier variables (e.g., `?x`, `?y`)
//! - **Constants**: Ground terms (e.g., `5`, `true`)
//! - **Function applications**: `f(?x, ?y)`
//! - **Ground terms**: Terms without bound variables
//!
//! # Compilation
//!
//! Patterns are compiled into efficient matching instructions that can be
//! executed against the term index.

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::{OxizError, Result};
use crate::sort::SortId;
use lasso::Spur;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::fmt;

/// A pattern for E-matching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pattern {
    /// Root term of the pattern
    pub root: TermId,
    /// Bound variables in the pattern
    pub variables: SmallVec<[PatternVariable; 4]>,
    /// The kind of pattern
    pub kind: PatternKind,
    /// Estimated cost of matching this pattern
    pub cost: u32,
    /// Whether this pattern is ground (contains no variables)
    pub is_ground: bool,
}

/// A variable in a pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatternVariable {
    /// Variable name (interned string)
    pub name: Spur,
    /// Variable sort
    pub sort: SortId,
    /// Index in the bound variable list
    pub index: usize,
}

/// Classification of pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternKind {
    /// Simple pattern: single function application
    Simple,
    /// Nested pattern: contains sub-patterns
    Nested,
    /// Ground pattern: no variables
    Ground,
    /// Variable-only pattern (not useful for matching)
    VarOnly,
}

/// Internal representation of a pattern as a DAG
#[derive(Debug, Clone)]
pub struct PatternNode {
    /// The term ID this node represents
    pub term: TermId,
    /// Variable index if this is a variable node
    pub var_index: Option<usize>,
    /// Child nodes
    pub children: SmallVec<[usize; 4]>,
    /// Whether this node must be matched exactly (not modulo E-graph)
    pub exact_match: bool,
}

/// Statistics about pattern compilation
#[derive(Debug, Clone, Default)]
pub struct PatternStats {
    /// Number of patterns compiled
    pub patterns_compiled: usize,
    /// Number of pattern variables
    pub total_variables: usize,
    /// Number of ground patterns
    pub ground_patterns: usize,
    /// Number of nested patterns
    pub nested_patterns: usize,
    /// Average pattern depth
    pub avg_depth: f64,
    /// Maximum pattern depth
    pub max_depth: usize,
}

/// Configuration for pattern compilation
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// Maximum pattern depth
    pub max_depth: usize,
    /// Whether to allow variable-only patterns
    pub allow_var_only: bool,
    /// Whether to allow ground patterns
    pub allow_ground: bool,
    /// Maximum number of variables per pattern
    pub max_variables: usize,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            allow_var_only: false,
            allow_ground: true,
            max_variables: 10,
        }
    }
}

/// Compiles patterns from terms
#[derive(Debug)]
pub struct PatternCompiler {
    /// Configuration
    config: PatternConfig,
    /// Statistics
    stats: PatternStats,
    /// Cache of compiled patterns
    cache: FxHashMap<TermId, Pattern>,
}

impl PatternCompiler {
    /// Create a new pattern compiler
    pub fn new(config: PatternConfig) -> Self {
        Self {
            config,
            stats: PatternStats::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Create a pattern compiler with default configuration
    pub fn new_default() -> Self {
        Self::new(PatternConfig::default())
    }

    /// Compile a term into a pattern
    ///
    /// The term should be from a quantifier body, and `bound_vars` should
    /// contain the quantifier's bound variables.
    pub fn compile(
        &mut self,
        term: TermId,
        bound_vars: &[(Spur, SortId)],
        manager: &TermManager,
    ) -> Result<Pattern> {
        // Check cache first
        if let Some(pattern) = self.cache.get(&term) {
            return Ok(pattern.clone());
        }

        // Build variable map
        let var_map: FxHashMap<Spur, usize> = bound_vars
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (*name, i))
            .collect();

        // Extract pattern variables
        let mut variables = SmallVec::new();
        let mut var_names = FxHashSet::default();
        self.collect_pattern_variables(term, bound_vars, &mut variables, &mut var_names, manager)?;

        // Check variable count limit
        if variables.len() > self.config.max_variables {
            return Err(OxizError::EmatchError(format!(
                "Pattern has {} variables, exceeding limit of {}",
                variables.len(),
                self.config.max_variables
            )));
        }

        // Determine pattern kind
        let kind = self.classify_pattern(term, &var_map, manager)?;

        // Check if pattern kind is allowed
        if kind == PatternKind::VarOnly && !self.config.allow_var_only {
            return Err(OxizError::EmatchError(
                "Variable-only patterns are not allowed".to_string(),
            ));
        }

        if kind == PatternKind::Ground && !self.config.allow_ground {
            return Err(OxizError::EmatchError(
                "Ground patterns are not allowed".to_string(),
            ));
        }

        // Compute pattern cost
        let cost = self.compute_cost(term, manager)?;

        // Check depth limit
        let depth = self.compute_depth(term, manager)?;
        if depth > self.config.max_depth {
            return Err(OxizError::EmatchError(format!(
                "Pattern depth {} exceeds maximum {}",
                depth, self.config.max_depth
            )));
        }

        let pattern = Pattern {
            root: term,
            variables,
            kind,
            cost,
            is_ground: kind == PatternKind::Ground,
        };

        // Update statistics
        self.stats.patterns_compiled += 1;
        self.stats.total_variables += pattern.variables.len();
        match kind {
            PatternKind::Ground => self.stats.ground_patterns += 1,
            PatternKind::Nested => self.stats.nested_patterns += 1,
            _ => {}
        }
        if depth > self.stats.max_depth {
            self.stats.max_depth = depth;
        }
        self.stats.avg_depth = (self.stats.avg_depth * (self.stats.patterns_compiled - 1) as f64
            + depth as f64)
            / self.stats.patterns_compiled as f64;

        // Cache the pattern
        self.cache.insert(term, pattern.clone());

        Ok(pattern)
    }

    /// Collect all pattern variables from a term
    fn collect_pattern_variables(
        &self,
        term: TermId,
        bound_vars: &[(Spur, SortId)],
        variables: &mut SmallVec<[PatternVariable; 4]>,
        seen: &mut FxHashSet<Spur>,
        manager: &TermManager,
    ) -> Result<()> {
        let Some(t) = manager.get(term) else {
            return Err(OxizError::EmatchError(format!(
                "Term {:?} not found in manager",
                term
            )));
        };

        match &t.kind {
            TermKind::Var(name) => {
                // Check if this is a bound variable
                if let Some((idx, (_, sort))) =
                    bound_vars.iter().enumerate().find(|(_, (n, _))| n == name)
                    && !seen.contains(name)
                {
                    variables.push(PatternVariable {
                        name: *name,
                        sort: *sort,
                        index: idx,
                    });
                    seen.insert(*name);
                }
            }
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    self.collect_pattern_variables(arg, bound_vars, variables, seen, manager)?;
                }
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs) => {
                self.collect_pattern_variables(*lhs, bound_vars, variables, seen, manager)?;
                self.collect_pattern_variables(*rhs, bound_vars, variables, seen, manager)?;
            }
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                for &arg in args.iter() {
                    self.collect_pattern_variables(arg, bound_vars, variables, seen, manager)?;
                }
            }
            TermKind::Not(inner) | TermKind::Neg(inner) => {
                self.collect_pattern_variables(*inner, bound_vars, variables, seen, manager)?;
            }
            TermKind::Ite(c, t, e) => {
                self.collect_pattern_variables(*c, bound_vars, variables, seen, manager)?;
                self.collect_pattern_variables(*t, bound_vars, variables, seen, manager)?;
                self.collect_pattern_variables(*e, bound_vars, variables, seen, manager)?;
            }
            TermKind::Select(arr, idx) => {
                self.collect_pattern_variables(*arr, bound_vars, variables, seen, manager)?;
                self.collect_pattern_variables(*idx, bound_vars, variables, seen, manager)?;
            }
            TermKind::Store(arr, idx, val) => {
                self.collect_pattern_variables(*arr, bound_vars, variables, seen, manager)?;
                self.collect_pattern_variables(*idx, bound_vars, variables, seen, manager)?;
                self.collect_pattern_variables(*val, bound_vars, variables, seen, manager)?;
            }
            // Constants and other ground terms don't contribute variables
            _ => {}
        }

        Ok(())
    }

    /// Classify a pattern by its structure
    fn classify_pattern(
        &self,
        term: TermId,
        var_map: &FxHashMap<Spur, usize>,
        manager: &TermManager,
    ) -> Result<PatternKind> {
        let Some(t) = manager.get(term) else {
            return Err(OxizError::EmatchError(format!(
                "Term {:?} not found in manager",
                term
            )));
        };

        match &t.kind {
            TermKind::Var(name) if var_map.contains_key(name) => Ok(PatternKind::VarOnly),
            TermKind::Var(_)
            | TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => Ok(PatternKind::Ground),
            TermKind::Apply { args, .. } => {
                // Check if all arguments are ground or variables
                let mut has_var = false;
                let mut has_nested = false;

                for &arg in args.iter() {
                    let arg_kind = self.classify_pattern(arg, var_map, manager)?;
                    match arg_kind {
                        PatternKind::VarOnly => has_var = true,
                        PatternKind::Nested | PatternKind::Simple => has_nested = true,
                        _ => {}
                    }
                }

                if has_nested {
                    Ok(PatternKind::Nested)
                } else if has_var {
                    Ok(PatternKind::Simple)
                } else {
                    Ok(PatternKind::Ground)
                }
            }
            _ => {
                // For other terms, check recursively
                if self.contains_bound_var(term, var_map, manager)? {
                    Ok(PatternKind::Nested)
                } else {
                    Ok(PatternKind::Ground)
                }
            }
        }
    }

    /// Check if a term contains any bound variables
    fn contains_bound_var(
        &self,
        term: TermId,
        var_map: &FxHashMap<Spur, usize>,
        manager: &TermManager,
    ) -> Result<bool> {
        let Some(t) = manager.get(term) else {
            return Ok(false);
        };

        match &t.kind {
            TermKind::Var(name) => Ok(var_map.contains_key(name)),
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    if self.contains_bound_var(arg, var_map, manager)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs) => Ok(self.contains_bound_var(*lhs, var_map, manager)?
                || self.contains_bound_var(*rhs, var_map, manager)?),
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                for &arg in args.iter() {
                    if self.contains_bound_var(arg, var_map, manager)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            TermKind::Not(inner) | TermKind::Neg(inner) => {
                self.contains_bound_var(*inner, var_map, manager)
            }
            TermKind::Ite(c, t, e) => Ok(self.contains_bound_var(*c, var_map, manager)?
                || self.contains_bound_var(*t, var_map, manager)?
                || self.contains_bound_var(*e, var_map, manager)?),
            TermKind::Select(arr, idx) => Ok(self.contains_bound_var(*arr, var_map, manager)?
                || self.contains_bound_var(*idx, var_map, manager)?),
            TermKind::Store(arr, idx, val) => Ok(self
                .contains_bound_var(*arr, var_map, manager)?
                || self.contains_bound_var(*idx, var_map, manager)?
                || self.contains_bound_var(*val, var_map, manager)?),
            _ => Ok(false),
        }
    }

    /// Compute the matching cost of a pattern
    ///
    /// Lower cost means more efficient matching. Cost is based on:
    /// - Number of variables (more variables = higher cost)
    /// - Pattern depth (deeper patterns = higher cost)
    /// - Ground subterms (ground terms = lower cost)
    fn compute_cost(&self, term: TermId, manager: &TermManager) -> Result<u32> {
        let Some(t) = manager.get(term) else {
            return Ok(1000); // Penalty for missing terms
        };

        let base_cost = match &t.kind {
            TermKind::Var(_) => 10, // Variables are expensive to match
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => 1, // Constants are cheap
            TermKind::Apply { args, .. } => {
                let mut cost = 5; // Base cost for function application
                for &arg in args.iter() {
                    cost += self.compute_cost(arg, manager)?;
                }
                cost
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs) => {
                3 + self.compute_cost(*lhs, manager)? + self.compute_cost(*rhs, manager)?
            }
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                let mut cost = 3;
                for &arg in args.iter() {
                    cost += self.compute_cost(arg, manager)?;
                }
                cost
            }
            TermKind::Not(inner) | TermKind::Neg(inner) => {
                2 + self.compute_cost(*inner, manager)?
            }
            TermKind::Ite(c, t, e) => {
                5 + self.compute_cost(*c, manager)?
                    + self.compute_cost(*t, manager)?
                    + self.compute_cost(*e, manager)?
            }
            TermKind::Select(arr, idx) => {
                4 + self.compute_cost(*arr, manager)? + self.compute_cost(*idx, manager)?
            }
            TermKind::Store(arr, idx, val) => {
                5 + self.compute_cost(*arr, manager)?
                    + self.compute_cost(*idx, manager)?
                    + self.compute_cost(*val, manager)?
            }
            _ => 20, // Default cost for unknown terms
        };

        Ok(base_cost)
    }

    /// Compute the depth of a pattern
    fn compute_depth(&self, term: TermId, manager: &TermManager) -> Result<usize> {
        let Some(t) = manager.get(term) else {
            return Ok(0);
        };

        let child_depth = match &t.kind {
            TermKind::Var(_)
            | TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => 0,
            TermKind::Apply { args, .. } => {
                let mut max_depth = 0;
                for &arg in args.iter() {
                    let depth = self.compute_depth(arg, manager)?;
                    if depth > max_depth {
                        max_depth = depth;
                    }
                }
                max_depth
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs) => {
                let left_depth = self.compute_depth(*lhs, manager)?;
                let right_depth = self.compute_depth(*rhs, manager)?;
                left_depth.max(right_depth)
            }
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                let mut max_depth = 0;
                for &arg in args.iter() {
                    let depth = self.compute_depth(arg, manager)?;
                    if depth > max_depth {
                        max_depth = depth;
                    }
                }
                max_depth
            }
            TermKind::Not(inner) | TermKind::Neg(inner) => self.compute_depth(*inner, manager)?,
            TermKind::Ite(c, t, e) => {
                let c_depth = self.compute_depth(*c, manager)?;
                let t_depth = self.compute_depth(*t, manager)?;
                let e_depth = self.compute_depth(*e, manager)?;
                c_depth.max(t_depth).max(e_depth)
            }
            TermKind::Select(arr, idx) => {
                let arr_depth = self.compute_depth(*arr, manager)?;
                let idx_depth = self.compute_depth(*idx, manager)?;
                arr_depth.max(idx_depth)
            }
            TermKind::Store(arr, idx, val) => {
                let arr_depth = self.compute_depth(*arr, manager)?;
                let idx_depth = self.compute_depth(*idx, manager)?;
                let val_depth = self.compute_depth(*val, manager)?;
                arr_depth.max(idx_depth).max(val_depth)
            }
            _ => 0,
        };

        Ok(child_depth + 1)
    }

    /// Build a pattern DAG for efficient matching
    pub fn build_dag(&self, pattern: &Pattern, manager: &TermManager) -> Result<Vec<PatternNode>> {
        let mut nodes = Vec::new();
        let mut node_map: FxHashMap<TermId, usize> = FxHashMap::default();
        self.build_dag_recursive(pattern.root, pattern, &mut nodes, &mut node_map, manager)?;
        Ok(nodes)
    }

    /// Recursive helper for building pattern DAG
    fn build_dag_recursive(
        &self,
        term: TermId,
        pattern: &Pattern,
        nodes: &mut Vec<PatternNode>,
        node_map: &mut FxHashMap<TermId, usize>,
        manager: &TermManager,
    ) -> Result<usize> {
        // Check if we've already created a node for this term
        if let Some(&node_idx) = node_map.get(&term) {
            return Ok(node_idx);
        }

        let Some(t) = manager.get(term) else {
            return Err(OxizError::EmatchError(format!(
                "Term {:?} not found in manager",
                term
            )));
        };

        // Check if this is a pattern variable
        let var_index = if let TermKind::Var(name) = &t.kind {
            pattern.variables.iter().position(|v| v.name == *name)
        } else {
            None
        };

        // Build children first
        let children = match &t.kind {
            TermKind::Apply { args, .. } => {
                let mut child_indices = SmallVec::new();
                for &arg in args.iter() {
                    let idx = self.build_dag_recursive(arg, pattern, nodes, node_map, manager)?;
                    child_indices.push(idx);
                }
                child_indices
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs) => {
                let left_idx = self.build_dag_recursive(*lhs, pattern, nodes, node_map, manager)?;
                let right_idx =
                    self.build_dag_recursive(*rhs, pattern, nodes, node_map, manager)?;
                smallvec::smallvec![left_idx, right_idx]
            }
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                let mut child_indices = SmallVec::new();
                for &arg in args.iter() {
                    let idx = self.build_dag_recursive(arg, pattern, nodes, node_map, manager)?;
                    child_indices.push(idx);
                }
                child_indices
            }
            TermKind::Not(inner) | TermKind::Neg(inner) => {
                let idx = self.build_dag_recursive(*inner, pattern, nodes, node_map, manager)?;
                smallvec::smallvec![idx]
            }
            TermKind::Ite(c, t, e) => {
                let c_idx = self.build_dag_recursive(*c, pattern, nodes, node_map, manager)?;
                let t_idx = self.build_dag_recursive(*t, pattern, nodes, node_map, manager)?;
                let e_idx = self.build_dag_recursive(*e, pattern, nodes, node_map, manager)?;
                smallvec::smallvec![c_idx, t_idx, e_idx]
            }
            TermKind::Select(arr, idx) => {
                let arr_idx = self.build_dag_recursive(*arr, pattern, nodes, node_map, manager)?;
                let idx_idx = self.build_dag_recursive(*idx, pattern, nodes, node_map, manager)?;
                smallvec::smallvec![arr_idx, idx_idx]
            }
            TermKind::Store(arr, idx, val) => {
                let arr_idx = self.build_dag_recursive(*arr, pattern, nodes, node_map, manager)?;
                let idx_idx = self.build_dag_recursive(*idx, pattern, nodes, node_map, manager)?;
                let val_idx = self.build_dag_recursive(*val, pattern, nodes, node_map, manager)?;
                smallvec::smallvec![arr_idx, idx_idx, val_idx]
            }
            _ => SmallVec::new(),
        };

        // Create the node
        let node = PatternNode {
            term,
            var_index,
            children,
            exact_match: false, // Will be set by optimizer later
        };

        let node_idx = nodes.len();
        nodes.push(node);
        node_map.insert(term, node_idx);

        Ok(node_idx)
    }

    /// Get statistics
    pub fn stats(&self) -> &PatternStats {
        &self.stats
    }

    /// Clear the pattern cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = PatternStats::default();
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pattern(root={:?}, vars={}, kind={:?}, cost={})",
            self.root,
            self.variables.len(),
            self.kind,
            self.cost
        )
    }
}

impl fmt::Display for PatternKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternKind::Simple => write!(f, "Simple"),
            PatternKind::Nested => write!(f, "Nested"),
            PatternKind::Ground => write!(f, "Ground"),
            PatternKind::VarOnly => write!(f, "VarOnly"),
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
    fn test_pattern_config_default() {
        let config = PatternConfig::default();
        assert_eq!(config.max_depth, 10);
        assert!(!config.allow_var_only);
        assert!(config.allow_ground);
        assert_eq!(config.max_variables, 10);
    }

    #[test]
    fn test_pattern_compiler_creation() {
        let compiler = PatternCompiler::new_default();
        assert_eq!(compiler.stats.patterns_compiled, 0);
    }

    #[test]
    fn test_compile_simple_pattern() {
        let mut manager = setup();
        let mut compiler = PatternCompiler::new_default();

        // Create pattern: f(x) where x is a bound variable
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars = vec![(x_name, int_sort)];

        let pattern = compiler.compile(f_x, &bound_vars, &manager).unwrap();

        assert_eq!(pattern.root, f_x);
        assert_eq!(pattern.variables.len(), 1);
        assert_eq!(pattern.variables[0].name, x_name);
        assert_eq!(pattern.kind, PatternKind::Simple);
        assert!(!pattern.is_ground);
    }

    #[test]
    fn test_compile_ground_pattern() {
        let mut manager = setup();
        let mut compiler = PatternCompiler::new_default();

        // Create ground pattern: f(5)
        let int_sort = manager.sorts.int_sort;
        let five = manager.mk_int(5);
        let f_five = manager.mk_apply("f", [five], int_sort);

        let bound_vars = vec![];
        let pattern = compiler.compile(f_five, &bound_vars, &manager).unwrap();

        assert_eq!(pattern.root, f_five);
        assert_eq!(pattern.variables.len(), 0);
        assert_eq!(pattern.kind, PatternKind::Ground);
        assert!(pattern.is_ground);
    }

    #[test]
    fn test_compile_nested_pattern() {
        let mut manager = setup();
        let mut compiler = PatternCompiler::new_default();

        // Create nested pattern: f(g(x)) where x is a bound variable
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let g_x = manager.mk_apply("g", [x], int_sort);
        let f_g_x = manager.mk_apply("f", [g_x], int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars = vec![(x_name, int_sort)];

        let pattern = compiler.compile(f_g_x, &bound_vars, &manager).unwrap();

        assert_eq!(pattern.root, f_g_x);
        assert_eq!(pattern.variables.len(), 1);
        assert_eq!(pattern.kind, PatternKind::Nested);
        assert!(!pattern.is_ground);
    }

    #[test]
    fn test_compile_multiple_variables() {
        let mut manager = setup();
        let mut compiler = PatternCompiler::new_default();

        // Create pattern: f(x, y) where x and y are bound variables
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let f_xy = manager.mk_apply("f", [x, y], int_sort);

        let x_name = manager.intern_str("x");
        let y_name = manager.intern_str("y");
        let bound_vars = vec![(x_name, int_sort), (y_name, int_sort)];

        let pattern = compiler.compile(f_xy, &bound_vars, &manager).unwrap();

        assert_eq!(pattern.root, f_xy);
        assert_eq!(pattern.variables.len(), 2);
        assert_eq!(pattern.kind, PatternKind::Simple);
    }

    #[test]
    fn test_pattern_cost_calculation() {
        let mut manager = setup();
        let compiler = PatternCompiler::new_default();

        let int_sort = manager.sorts.int_sort;

        // Ground term should have lower cost
        let five = manager.mk_int(5);
        let ground_cost = compiler.compute_cost(five, &manager).unwrap();

        // Variable should have higher cost
        let x = manager.mk_var("x", int_sort);
        let var_cost = compiler.compute_cost(x, &manager).unwrap();

        assert!(var_cost > ground_cost);
    }

    #[test]
    fn test_pattern_depth_calculation() {
        let mut manager = setup();
        let compiler = PatternCompiler::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        // Depth 1: x
        let depth1 = compiler.compute_depth(x, &manager).unwrap();
        assert_eq!(depth1, 1);

        // Depth 2: f(x)
        let f_x = manager.mk_apply("f", [x], int_sort);
        let depth2 = compiler.compute_depth(f_x, &manager).unwrap();
        assert_eq!(depth2, 2);

        // Depth 3: g(f(x))
        let g_f_x = manager.mk_apply("g", [f_x], int_sort);
        let depth3 = compiler.compute_depth(g_f_x, &manager).unwrap();
        assert_eq!(depth3, 3);
    }

    #[test]
    fn test_pattern_dag_build() {
        let mut manager = setup();
        let mut compiler = PatternCompiler::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars = vec![(x_name, int_sort)];

        let pattern = compiler.compile(f_x, &bound_vars, &manager).unwrap();
        let dag = compiler.build_dag(&pattern, &manager).unwrap();

        // Should have nodes for x and f(x)
        assert!(dag.len() >= 2);

        // Find the root node (f(x))
        let root_node = dag.last().unwrap();
        assert_eq!(root_node.term, f_x);
        assert!(!root_node.children.is_empty());
    }

    #[test]
    fn test_var_only_pattern_rejected() {
        let mut manager = setup();
        let config = PatternConfig {
            allow_var_only: false,
            ..Default::default()
        };
        let mut compiler = PatternCompiler::new(config);

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars = vec![(x_name, int_sort)];

        let result = compiler.compile(x, &bound_vars, &manager);
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_caching() {
        let mut manager = setup();
        let mut compiler = PatternCompiler::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);

        let x_name = manager.intern_str("x");
        let bound_vars = vec![(x_name, int_sort)];

        // Compile twice
        let pattern1 = compiler.compile(f_x, &bound_vars, &manager).unwrap();
        let pattern2 = compiler.compile(f_x, &bound_vars, &manager).unwrap();

        // Should get the same pattern from cache
        assert_eq!(pattern1, pattern2);
        // Should only count as one compilation
        assert_eq!(compiler.stats.patterns_compiled, 1);
    }

    #[test]
    fn test_pattern_stats() {
        let mut manager = setup();
        let mut compiler = PatternCompiler::new_default();

        let int_sort = manager.sorts.int_sort;

        // Compile a few patterns
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let x_name = manager.intern_str("x");
        let bound_vars = vec![(x_name, int_sort)];

        compiler.compile(f_x, &bound_vars, &manager).unwrap();

        let five = manager.mk_int(5);
        let f_five = manager.mk_apply("f", [five], int_sort);
        compiler.compile(f_five, &[], &manager).unwrap();

        let stats = compiler.stats();
        assert_eq!(stats.patterns_compiled, 2);
        assert_eq!(stats.ground_patterns, 1);
    }
}
