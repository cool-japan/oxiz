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
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use core::fmt;
use smallvec::SmallVec;

/// Sub-terms a pattern walk descends into.
///
/// The exhaustive [`crate::ast::traversal::get_children`], except that
/// nested binders are opaque: a `forall`/`exists`/`let`/`match` inside a
/// pattern may rebind a name, so a `Var` below it is not an occurrence of
/// the pattern's own variable.
///
/// Everything else is descended into. The per-kind lists these walks used
/// ended in silent catch-alls, so the operands of string, bit-vector,
/// floating-point, datatype and `distinct` terms were invisible: pattern
/// variables under them were never collected, and
/// [`PatternCompiler::build_dag`] produced a node with no children for
/// them — a pattern DAG that cannot match what the pattern says.
fn pattern_children(kind: &TermKind) -> SmallVec<[TermId; 4]> {
    match kind {
        TermKind::Forall { .. }
        | TermKind::Exists { .. }
        | TermKind::Let { .. }
        | TermKind::Match { .. } => SmallVec::new(),
        other => crate::ast::traversal::get_children(other),
    }
}

/// Children-before-parents listing of the sub-terms of `root`, computed with
/// an explicit stack. Shared sub-terms appear exactly once.
fn pattern_postorder(root: TermId, manager: &TermManager) -> Vec<TermId> {
    let mut order = Vec::new();
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack = vec![(root, false)];

    while let Some((current, expanded)) = stack.pop() {
        if expanded {
            order.push(current);
            continue;
        }
        if !visited.insert(current) {
            continue;
        }
        stack.push((current, true));
        if let Some(t) = manager.get(current) {
            for child in pattern_children(&t.kind) {
                if !visited.contains(&child) {
                    stack.push((child, false));
                }
            }
        }
    }

    order
}

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

    /// Collect all pattern variables from a term.
    ///
    /// Iterative pre-order walk (children pushed in reverse so they are
    /// visited left-to-right, preserving the order in which variables were
    /// first encountered by the recursive form), with a visited set so a
    /// shared sub-term is not re-expanded.
    fn collect_pattern_variables(
        &self,
        term: TermId,
        bound_vars: &[(Spur, SortId)],
        variables: &mut SmallVec<[PatternVariable; 4]>,
        seen: &mut FxHashSet<Spur>,
        manager: &TermManager,
    ) -> Result<()> {
        let mut stack = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(t) = manager.get(current) else {
                return Err(OxizError::EmatchError(format!(
                    "Term {:?} not found in manager",
                    current
                )));
            };

            if let TermKind::Var(name) = &t.kind {
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
                continue;
            }

            let children = pattern_children(&t.kind);
            stack.extend(children.iter().rev().copied());
        }

        Ok(())
    }

    /// Classify a pattern by its structure.
    ///
    /// Iterative bottom-up classification with a memo, so nested
    /// applications neither recurse without bound nor re-classify shared
    /// sub-terms once per path.
    fn classify_pattern(
        &self,
        term: TermId,
        var_map: &FxHashMap<Spur, usize>,
        manager: &TermManager,
    ) -> Result<PatternKind> {
        let mut memo: FxHashMap<TermId, PatternKind> = FxHashMap::default();

        for current in pattern_postorder(term, manager) {
            let Some(t) = manager.get(current) else {
                return Err(OxizError::EmatchError(format!(
                    "Term {:?} not found in manager",
                    current
                )));
            };

            let kind = match &t.kind {
                TermKind::Var(name) if var_map.contains_key(name) => PatternKind::VarOnly,
                TermKind::Var(_)
                | TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_) => PatternKind::Ground,
                TermKind::Apply { args, .. } => {
                    // Check if all arguments are ground or variables
                    let mut has_var = false;
                    let mut has_nested = false;

                    for arg in args.iter() {
                        match memo.get(arg) {
                            Some(PatternKind::VarOnly) => has_var = true,
                            Some(PatternKind::Nested | PatternKind::Simple) => has_nested = true,
                            _ => {}
                        }
                    }

                    if has_nested {
                        PatternKind::Nested
                    } else if has_var {
                        PatternKind::Simple
                    } else {
                        PatternKind::Ground
                    }
                }
                _ => {
                    // For other terms, the classification is decided by
                    // whether a bound variable occurs anywhere below.
                    if self.contains_bound_var(current, var_map, manager)? {
                        PatternKind::Nested
                    } else {
                        PatternKind::Ground
                    }
                }
            };
            memo.insert(current, kind);
        }

        memo.get(&term)
            .copied()
            .ok_or_else(|| OxizError::EmatchError(format!("Term {:?} not found in manager", term)))
    }

    /// Check if a term contains any bound variables.
    ///
    /// Iterative with a visited set: the recursive form both recursed once
    /// per level of nesting and re-expanded shared sub-terms, and its
    /// catch-all reported "no bound variable" for every kind it did not
    /// enumerate — which classified a genuine pattern as ground and
    /// discarded it.
    fn contains_bound_var(
        &self,
        term: TermId,
        var_map: &FxHashMap<Spur, usize>,
        manager: &TermManager,
    ) -> Result<bool> {
        let mut stack = vec![term];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(t) = manager.get(current) else {
                continue;
            };
            if let TermKind::Var(name) = &t.kind {
                if var_map.contains_key(name) {
                    return Ok(true);
                }
                continue;
            }
            stack.extend(pattern_children(&t.kind));
        }

        Ok(false)
    }

    /// Compute the matching cost of a pattern
    ///
    /// Lower cost means more efficient matching. Cost is based on:
    /// - Number of variables (more variables = higher cost)
    /// - Pattern depth (deeper patterns = higher cost)
    /// - Ground subterms (ground terms = lower cost)
    ///
    /// Evaluated bottom-up over an explicit post-order with a memo: the
    /// recursive form re-expanded shared sub-terms once per path (a
    /// doubling DAG made it exponential) and had no depth bound. Each
    /// occurrence still contributes its cost, so the value is exactly the
    /// one the recursion produced; the accumulation saturates rather than
    /// overflowing.
    fn compute_cost(&self, term: TermId, manager: &TermManager) -> Result<u32> {
        let mut memo: FxHashMap<TermId, u32> = FxHashMap::default();

        for current in pattern_postorder(term, manager) {
            let Some(t) = manager.get(current) else {
                memo.insert(current, 1000); // Penalty for missing terms
                continue;
            };
            let child = |id: &TermId, memo: &FxHashMap<TermId, u32>| -> u32 {
                memo.get(id).copied().unwrap_or(1000)
            };
            let sum = |base: u32, ids: &[TermId], memo: &FxHashMap<TermId, u32>| -> u32 {
                ids.iter()
                    .fold(base, |acc, id| acc.saturating_add(child(id, memo)))
            };
            let cost = match &t.kind {
                TermKind::Var(_) => 10, // Variables are expensive to match
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_) => 1, // Constants are cheap
                TermKind::Apply { args, .. } => sum(5, args, &memo),
                TermKind::Eq(lhs, rhs)
                | TermKind::Lt(lhs, rhs)
                | TermKind::Le(lhs, rhs)
                | TermKind::Gt(lhs, rhs)
                | TermKind::Ge(lhs, rhs)
                | TermKind::Sub(lhs, rhs)
                | TermKind::Div(lhs, rhs) => sum(3, &[*lhs, *rhs], &memo),
                TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::And(args)
                | TermKind::Or(args) => sum(3, args, &memo),
                TermKind::Not(inner) | TermKind::Neg(inner) => sum(2, &[*inner], &memo),
                TermKind::Ite(c, then_b, else_b) => sum(5, &[*c, *then_b, *else_b], &memo),
                TermKind::Select(arr, idx) => sum(4, &[*arr, *idx], &memo),
                TermKind::Store(arr, idx, val) => sum(5, &[*arr, *idx, *val], &memo),
                _ => 20, // Default cost for unknown terms
            };
            memo.insert(current, cost);
        }

        Ok(memo.get(&term).copied().unwrap_or(1000))
    }

    /// Compute the depth of a pattern.
    ///
    /// Bottom-up over an explicit post-order with a memo, measuring depth
    /// through *every* operand. The old per-kind list reported depth 1 for
    /// any kind it did not enumerate, which let arbitrarily deep patterns
    /// slip past the `max_depth` limit that exists to reject them.
    fn compute_depth(&self, term: TermId, manager: &TermManager) -> Result<usize> {
        let mut memo: FxHashMap<TermId, usize> = FxHashMap::default();

        for current in pattern_postorder(term, manager) {
            let Some(t) = manager.get(current) else {
                memo.insert(current, 0);
                continue;
            };
            let child_depth = pattern_children(&t.kind)
                .iter()
                .map(|c| memo.get(c).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            memo.insert(current, child_depth + 1);
        }

        Ok(memo.get(&term).copied().unwrap_or(0))
    }

    /// Build a pattern DAG for efficient matching
    pub fn build_dag(&self, pattern: &Pattern, manager: &TermManager) -> Result<Vec<PatternNode>> {
        let mut nodes = Vec::new();
        let mut node_map: FxHashMap<TermId, usize> = FxHashMap::default();
        self.build_dag_recursive(pattern.root, pattern, &mut nodes, &mut node_map, manager)?;
        Ok(nodes)
    }

    /// Iterative helper for building the pattern DAG.
    ///
    /// Children are created before their parent (post-order, left to
    /// right), so node indices are numbered exactly as the recursive form
    /// numbered them. The node's children now come from the exhaustive
    /// [`pattern_children`]: the previous catch-all created a childless
    /// node for every kind it did not enumerate, i.e. a DAG that silently
    /// claimed the pattern had no structure below that point.
    fn build_dag_recursive(
        &self,
        term: TermId,
        pattern: &Pattern,
        nodes: &mut Vec<PatternNode>,
        node_map: &mut FxHashMap<TermId, usize>,
        manager: &TermManager,
    ) -> Result<usize> {
        let mut stack = vec![(term, false)];

        while let Some((current, expanded)) = stack.pop() {
            // Check if we've already created a node for this term
            if node_map.contains_key(&current) {
                continue;
            }

            let Some(t) = manager.get(current) else {
                return Err(OxizError::EmatchError(format!(
                    "Term {:?} not found in manager",
                    current
                )));
            };
            let children = pattern_children(&t.kind);

            if !expanded {
                stack.push((current, true));
                for &child in children.iter().rev() {
                    stack.push((child, false));
                }
                continue;
            }

            // Check if this is a pattern variable
            let var_index = if let TermKind::Var(name) = &t.kind {
                pattern.variables.iter().position(|v| v.name == *name)
            } else {
                None
            };

            let mut child_indices: SmallVec<[usize; 4]> = SmallVec::new();
            for child in &children {
                let Some(&idx) = node_map.get(child) else {
                    return Err(OxizError::EmatchError(format!(
                        "Pattern child {:?} was not built before its parent",
                        child
                    )));
                };
                child_indices.push(idx);
            }

            let node = PatternNode {
                term: current,
                var_index,
                children: child_indices,
                exact_match: false, // Will be set by optimizer later
            };

            let node_idx = nodes.len();
            nodes.push(node);
            node_map.insert(current, node_idx);
        }

        node_map
            .get(&term)
            .copied()
            .ok_or_else(|| OxizError::EmatchError(format!("Term {:?} not found in manager", term)))
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

        let pattern = compiler
            .compile(f_x, &bound_vars, &manager)
            .expect("test operation should succeed");

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
        let pattern = compiler
            .compile(f_five, &bound_vars, &manager)
            .expect("test operation should succeed");

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

        let pattern = compiler
            .compile(f_g_x, &bound_vars, &manager)
            .expect("test operation should succeed");

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

        let pattern = compiler
            .compile(f_xy, &bound_vars, &manager)
            .expect("test operation should succeed");

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
        let ground_cost = compiler
            .compute_cost(five, &manager)
            .expect("test operation should succeed");

        // Variable should have higher cost
        let x = manager.mk_var("x", int_sort);
        let var_cost = compiler
            .compute_cost(x, &manager)
            .expect("test operation should succeed");

        assert!(var_cost > ground_cost);
    }

    #[test]
    fn test_pattern_depth_calculation() {
        let mut manager = setup();
        let compiler = PatternCompiler::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        // Depth 1: x
        let depth1 = compiler
            .compute_depth(x, &manager)
            .expect("test operation should succeed");
        assert_eq!(depth1, 1);

        // Depth 2: f(x)
        let f_x = manager.mk_apply("f", [x], int_sort);
        let depth2 = compiler
            .compute_depth(f_x, &manager)
            .expect("test operation should succeed");
        assert_eq!(depth2, 2);

        // Depth 3: g(f(x))
        let g_f_x = manager.mk_apply("g", [f_x], int_sort);
        let depth3 = compiler
            .compute_depth(g_f_x, &manager)
            .expect("test operation should succeed");
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

        let pattern = compiler
            .compile(f_x, &bound_vars, &manager)
            .expect("test operation should succeed");
        let dag = compiler
            .build_dag(&pattern, &manager)
            .expect("test operation should succeed");

        // Should have nodes for x and f(x)
        assert!(dag.len() >= 2);

        // Find the root node (f(x))
        let root_node = dag.last().expect("collection should not be empty");
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
        let pattern1 = compiler
            .compile(f_x, &bound_vars, &manager)
            .expect("test operation should succeed");
        let pattern2 = compiler
            .compile(f_x, &bound_vars, &manager)
            .expect("test operation should succeed");

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

        compiler
            .compile(f_x, &bound_vars, &manager)
            .expect("test operation should succeed");

        let five = manager.mk_int(5);
        let f_five = manager.mk_apply("f", [five], int_sort);
        compiler
            .compile(f_five, &[], &manager)
            .expect("test operation should succeed");

        let stats = compiler.stats();
        assert_eq!(stats.patterns_compiled, 2);
        assert_eq!(stats.ground_patterns, 1);
    }
}

#[cfg(test)]
mod deep_walk_tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_pattern_variables_seen_under_store() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let arr = manager.mk_var("a", array_sort);
        let x = manager.mk_var("x", int_sort);
        let value = manager.mk_int(1);
        let stored = manager.mk_store(arr, x, value);

        let x_name = manager.intern_str("x");
        let bound_vars = vec![(x_name, int_sort)];
        let mut compiler = PatternCompiler::new_default();
        let pattern = compiler
            .compile(stored, &bound_vars, &manager)
            .expect("compilation should succeed");

        assert!(
            pattern.variables.iter().any(|v| v.name == x_name),
            "pattern variable below `store` was lost"
        );
        assert!(!pattern.is_ground);

        // The DAG must carry the operands of `store`.
        let nodes = compiler
            .build_dag(&pattern, &manager)
            .expect("dag build should succeed");
        let root = nodes.last().expect("dag has a root node");
        assert_eq!(root.term, stored);
        assert_eq!(root.children.len(), 3);
    }

    #[test]
    fn test_pattern_walks_deep_nesting_do_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let x = manager.mk_var("x", int_sort);
                let mut term = x;
                for _ in 0..7_500 {
                    term = manager.mk_apply("f", [term], int_sort);
                }

                let x_name = manager.intern_str("x");
                let var_map: FxHashMap<Spur, usize> = [(x_name, 0)].into_iter().collect();
                let compiler = PatternCompiler::new_default();

                let contains = compiler
                    .contains_bound_var(term, &var_map, &manager)
                    .expect("walk should succeed");
                let depth = compiler
                    .compute_depth(term, &manager)
                    .expect("walk should succeed");
                let cost = compiler
                    .compute_cost(term, &manager)
                    .expect("walk should succeed");
                let kind = compiler
                    .classify_pattern(term, &var_map, &manager)
                    .expect("walk should succeed");
                (contains, depth, cost, kind)
            })
            .expect("thread spawn should succeed");

        let (contains, depth, cost, kind) = handle.join().expect("deep walks must not overflow");
        assert!(contains);
        assert_eq!(depth, 7_501);
        assert!(cost > 0);
        assert_eq!(kind, PatternKind::Nested);
    }

    #[test]
    fn test_pattern_cost_shared_dag_is_fast() {
        // 40 doubling levels: exponential re-expansion would never finish.
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let mut level = x;
        for _ in 0..40 {
            level = manager.mk_apply("f", [level, level], int_sort);
        }

        let compiler = PatternCompiler::new_default();
        let cost = compiler
            .compute_cost(level, &manager)
            .expect("walk should succeed");
        assert_eq!(cost, u32::MAX, "saturating accumulation expected");
        let depth = compiler
            .compute_depth(level, &manager)
            .expect("walk should succeed");
        assert_eq!(depth, 41);
    }
}
