//! Recursive CHC support and analysis.
//!
//! This module provides utilities for detecting and handling recursive
//! predicates in CHC systems, which are common in verification of
//! recursive functions and data structures.
//!
//! Reference: Z3's recursive predicate handling in Spacer

use crate::chc::{ChcSystem, PredId, Rule};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use tracing::{debug, trace};

/// Errors in recursive CHC analysis
#[derive(Error, Debug)]
pub enum RecursiveError {
    /// Invalid recursion pattern
    #[error("invalid recursion pattern: {0}")]
    InvalidPattern(String),
    /// Cyclic dependency detected
    #[error("cyclic dependency in non-recursive context")]
    CyclicDependency,
}

/// Type of recursion in a predicate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursionKind {
    /// Not recursive
    NonRecursive,
    /// Directly recursive (predicate appears in its own rules)
    DirectRecursive,
    /// Mutually recursive with other predicates
    MutuallyRecursive,
    /// Nested recursion (recursive calls within recursive calls)
    NestedRecursive,
}

/// Information about a recursive predicate
#[derive(Debug, Clone)]
pub struct RecursiveInfo {
    /// The predicate ID
    pub pred: PredId,
    /// Kind of recursion
    pub kind: RecursionKind,
    /// Predicates this one depends on
    pub dependencies: HashSet<PredId>,
    /// Predicates that depend on this one
    pub dependents: HashSet<PredId>,
    /// Recursive rules (rules that contain the predicate in both head and body)
    pub recursive_rules: Vec<usize>, // Rule indices
    /// Base case rules (non-recursive rules)
    pub base_rules: Vec<usize>,
}

impl RecursiveInfo {
    /// Create new recursive info
    pub fn new(pred: PredId) -> Self {
        Self {
            pred,
            kind: RecursionKind::NonRecursive,
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
            recursive_rules: Vec::new(),
            base_rules: Vec::new(),
        }
    }

    /// Check if predicate is recursive
    pub fn is_recursive(&self) -> bool {
        self.kind != RecursionKind::NonRecursive
    }

    /// Check if predicate has base cases
    pub fn has_base_cases(&self) -> bool {
        !self.base_rules.is_empty()
    }

    /// Get recursion depth (number of predicates in mutual recursion)
    pub fn recursion_depth(&self) -> usize {
        match self.kind {
            RecursionKind::NonRecursive => 0,
            RecursionKind::DirectRecursive => 1,
            RecursionKind::MutuallyRecursive => self.dependencies.len(),
            RecursionKind::NestedRecursive => self.dependencies.len() + 1,
        }
    }
}

/// Analyzer for recursive CHC systems
pub struct RecursiveAnalyzer<'a> {
    /// The CHC system to analyze
    system: &'a ChcSystem,
    /// Recursive information for each predicate
    info: HashMap<PredId, RecursiveInfo>,
}

impl<'a> RecursiveAnalyzer<'a> {
    /// Create a new recursive analyzer
    pub fn new(system: &'a ChcSystem) -> Self {
        Self {
            system,
            info: HashMap::new(),
        }
    }

    /// Analyze the CHC system for recursion
    pub fn analyze(&mut self) -> Result<(), RecursiveError> {
        debug!("Analyzing CHC system for recursion");

        // Initialize info for all predicates
        for pred in self.system.predicates() {
            self.info.insert(pred.id, RecursiveInfo::new(pred.id));
        }

        // Build dependency graph
        self.build_dependency_graph()?;

        // Detect recursion kinds
        self.detect_recursion_kinds()?;

        // Classify rules
        self.classify_rules()?;

        debug!(
            "Found {} recursive predicates",
            self.info
                .values()
                .filter(|info| info.is_recursive())
                .count()
        );

        Ok(())
    }

    /// Build the dependency graph between predicates
    fn build_dependency_graph(&mut self) -> Result<(), RecursiveError> {
        for rule in self.system.rules() {
            if let Some(head_pred) = rule.head_predicate() {
                // Collect body predicates first
                let body_preds: Vec<PredId> =
                    rule.body.predicates.iter().map(|app| app.pred).collect();

                // Get or create info for head predicate
                let head_info = self
                    .info
                    .entry(head_pred)
                    .or_insert_with(|| RecursiveInfo::new(head_pred));

                // Add dependencies from body predicates
                for body_pred in &body_preds {
                    head_info.dependencies.insert(*body_pred);
                }

                // Now update body predicates (separate borrow)
                for body_pred in body_preds {
                    let body_info = self
                        .info
                        .entry(body_pred)
                        .or_insert_with(|| RecursiveInfo::new(body_pred));
                    body_info.dependents.insert(head_pred);
                }
            }
        }

        Ok(())
    }

    /// Detect recursion kinds for each predicate
    fn detect_recursion_kinds(&mut self) -> Result<(), RecursiveError> {
        // Clone the info to avoid borrow issues
        let pred_ids: Vec<PredId> = self.info.keys().copied().collect();

        for pred_id in pred_ids {
            let kind = self.detect_predicate_recursion(pred_id)?;

            if let Some(info) = self.info.get_mut(&pred_id) {
                info.kind = kind;
                trace!("Predicate {:?} has recursion kind {:?}", pred_id, kind);
            }
        }

        Ok(())
    }

    /// Detect recursion kind for a specific predicate
    fn detect_predicate_recursion(&self, pred: PredId) -> Result<RecursionKind, RecursiveError> {
        let info = self
            .info
            .get(&pred)
            .ok_or_else(|| RecursiveError::InvalidPattern("predicate not found".to_string()))?;

        // Check for direct recursion
        if info.dependencies.contains(&pred) {
            // Check for nested recursion (depends on other recursive predicates)
            let has_recursive_deps = info.dependencies.iter().any(|dep| {
                if let Some(dep_info) = self.info.get(dep) {
                    dep_info.dependencies.contains(&pred) || dep_info.dependencies.contains(dep)
                } else {
                    false
                }
            });

            if has_recursive_deps {
                return Ok(RecursionKind::NestedRecursive);
            } else {
                return Ok(RecursionKind::DirectRecursive);
            }
        }

        // Check for mutual recursion
        for dep in &info.dependencies {
            if let Some(dep_info) = self.info.get(dep)
                && dep_info.dependencies.contains(&pred)
            {
                return Ok(RecursionKind::MutuallyRecursive);
            }
        }

        Ok(RecursionKind::NonRecursive)
    }

    /// Classify rules as recursive or base cases
    fn classify_rules(&mut self) -> Result<(), RecursiveError> {
        for (rule_idx, rule) in self.system.rules().enumerate() {
            if let Some(head_pred) = rule.head_predicate() {
                let is_recursive = self.is_rule_recursive(rule);

                if let Some(info) = self.info.get_mut(&head_pred) {
                    if is_recursive {
                        info.recursive_rules.push(rule_idx);
                    } else {
                        info.base_rules.push(rule_idx);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if a rule is recursive
    fn is_rule_recursive(&self, rule: &Rule) -> bool {
        if let Some(head_pred) = rule.head_predicate() {
            // Check if head predicate appears in body
            rule.body
                .predicates
                .iter()
                .any(|body_app| body_app.pred == head_pred)
        } else {
            false
        }
    }

    /// Get recursive info for a predicate
    pub fn get_info(&self, pred: PredId) -> Option<&RecursiveInfo> {
        self.info.get(&pred)
    }

    /// Get all recursive predicates
    pub fn recursive_predicates(&self) -> impl Iterator<Item = &RecursiveInfo> {
        self.info.values().filter(|info| info.is_recursive())
    }

    /// Get strongly connected components (mutual recursion groups)
    pub fn strongly_connected_components(&self) -> Vec<Vec<PredId>> {
        let mut sccs = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        for pred_id in self.info.keys() {
            if !visited.contains(pred_id) {
                self.tarjan_scc(
                    *pred_id,
                    &mut visited,
                    &mut stack,
                    &mut sccs,
                    &mut HashMap::new(),
                    &mut 0,
                );
            }
        }

        sccs
    }

    /// Tarjan's algorithm for finding SCCs
    #[allow(clippy::too_many_arguments)]
    fn tarjan_scc(
        &self,
        pred: PredId,
        visited: &mut HashSet<PredId>,
        stack: &mut Vec<PredId>,
        sccs: &mut Vec<Vec<PredId>>,
        indices: &mut HashMap<PredId, usize>,
        index_counter: &mut usize,
    ) {
        visited.insert(pred);
        indices.insert(pred, *index_counter);
        let mut low_link = *index_counter;
        *index_counter += 1;
        stack.push(pred);

        if let Some(info) = self.info.get(&pred) {
            for &dep in &info.dependencies {
                if !visited.contains(&dep) {
                    self.tarjan_scc(dep, visited, stack, sccs, indices, index_counter);
                    if let Some(&dep_low) = indices.get(&dep) {
                        low_link = low_link.min(dep_low);
                    }
                } else if stack.contains(&dep)
                    && let Some(&dep_idx) = indices.get(&dep)
                {
                    low_link = low_link.min(dep_idx);
                }
            }
        }

        if low_link == indices[&pred] {
            let mut scc = Vec::new();
            while let Some(node) = stack.pop() {
                scc.push(node);
                if node == pred {
                    break;
                }
            }
            if scc.len() > 1
                || (scc.len() == 1
                    && self
                        .info
                        .get(&scc[0])
                        .map(|i| i.dependencies.contains(&scc[0]))
                        .unwrap_or(false))
            {
                sccs.push(scc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::TermManager;

    #[test]
    fn test_recursion_kind() {
        let info = RecursiveInfo::new(PredId(0));
        assert_eq!(info.kind, RecursionKind::NonRecursive);
        assert!(!info.is_recursive());
    }

    #[test]
    fn test_recursive_info() {
        let mut info = RecursiveInfo::new(PredId(0));
        info.kind = RecursionKind::DirectRecursive;
        info.dependencies.insert(PredId(0));
        info.recursive_rules.push(0);
        info.base_rules.push(1);

        assert!(info.is_recursive());
        assert!(info.has_base_cases());
        assert_eq!(info.recursion_depth(), 1);
    }

    #[test]
    fn test_analyzer_empty_system() {
        let system = ChcSystem::new();
        let mut analyzer = RecursiveAnalyzer::new(&system);
        assert!(analyzer.analyze().is_ok());
    }

    #[test]
    fn test_analyzer_simple_system() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        // Create a simple non-recursive system
        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let init_constraint = terms.mk_eq(x, zero);

        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            init_constraint,
            inv,
            [x],
        );

        let mut analyzer = RecursiveAnalyzer::new(&system);
        assert!(analyzer.analyze().is_ok());

        // Check that predicate is non-recursive
        let info = analyzer.get_info(inv);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.kind, RecursionKind::NonRecursive);
    }

    #[test]
    fn test_scc_computation() {
        let system = ChcSystem::new();
        let analyzer = RecursiveAnalyzer::new(&system);
        let sccs = analyzer.strongly_connected_components();
        assert!(sccs.is_empty());
    }
}
