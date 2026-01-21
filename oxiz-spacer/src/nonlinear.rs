//! Non-linear CHC constraint handling.
//!
//! This module provides support for non-linear Constrained Horn Clauses,
//! where rules can have multiple recursive predicate applications in the body.
//!
//! ## Non-linear CHCs
//!
//! Non-linear CHCs have the form:
//! ```text
//! P1(x1) ∧ P2(x2) ∧ ... ∧ Pn(xn) ∧ φ => P(x)
//! ```
//!
//! Where multiple predicate applications appear in the body. This is more
//! general than linear CHCs and can express problems like:
//! - Tree invariants (left child ∧ right child => parent)
//! - Mutual recursion
//! - Complex data structure verification
//!
//! ## Solving Approach
//!
//! Non-linear CHCs require different solving strategies:
//! 1. **Tree interpolation**: Generate interpolants for tree-like derivations
//! 2. **Product construction**: Build product automata for multiple predicates
//! 3. **Lemma combination**: Combine lemmas from different predicates
//!
//! Reference: Non-linear CHC solving techniques from literature

use crate::chc::{ChcSystem, PredId, PredicateApp, Rule};
use crate::frames::LemmaId;
use crate::pdr::SpacerError;
use oxiz_core::{TermId, TermManager};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors specific to non-linear CHC handling
#[derive(Error, Debug)]
pub enum NonLinearError {
    /// Non-linear rule not supported by current strategy
    #[error("non-linear rule not supported: {0}")]
    Unsupported(String),
    /// Interpolation failed
    #[error("interpolation failed: {0}")]
    InterpolationFailed(String),
    /// Product construction failed
    #[error("product construction failed: {0}")]
    ProductConstructionFailed(String),
    /// Spacer error
    #[error("spacer error: {0}")]
    Spacer(#[from] SpacerError),
}

/// Result type for non-linear operations
pub type NonLinearResult<T> = Result<T, NonLinearError>;

/// Analysis of a non-linear rule
#[derive(Debug, Clone)]
pub struct NonLinearRuleAnalysis {
    /// The rule being analyzed
    pub rule_id: usize,
    /// Number of predicate applications in body
    pub body_pred_count: usize,
    /// Predicates in the body
    pub body_preds: SmallVec<[PredId; 4]>,
    /// Head predicate
    pub head_pred: PredId,
    /// Degree of non-linearity (0 = linear, 1 = binary, 2+ = higher-order)
    pub nonlinearity_degree: usize,
    /// Whether this forms a cycle in the dependency graph
    pub is_cyclic: bool,
}

impl NonLinearRuleAnalysis {
    /// Analyze a rule for non-linearity
    pub fn analyze(rule: &Rule) -> Self {
        let body_preds: SmallVec<[PredId; 4]> =
            rule.body.predicates.iter().map(|app| app.pred).collect();

        let body_pred_count = body_preds.len();
        let nonlinearity_degree = body_pred_count.saturating_sub(1);

        let head_pred = rule.head_predicate().unwrap_or(PredId(0)); // Use 0 for queries

        Self {
            rule_id: 0, // Will be set by caller
            body_pred_count,
            body_preds,
            head_pred,
            nonlinearity_degree,
            is_cyclic: false, // Will be determined by dependency analysis
        }
    }

    /// Check if this rule is linear
    pub fn is_linear(&self) -> bool {
        self.body_pred_count <= 1
    }

    /// Check if this rule is binary non-linear
    pub fn is_binary_nonlinear(&self) -> bool {
        self.body_pred_count == 2
    }
}

/// Non-linear CHC analyzer
pub struct NonLinearAnalyzer {
    /// Analyses for each rule
    rule_analyses: Vec<NonLinearRuleAnalysis>,
    /// Dependency graph (pred -> dependent preds)
    dependencies: HashMap<PredId, HashSet<PredId>>,
    /// Strongly connected components
    sccs: Vec<HashSet<PredId>>,
}

impl NonLinearAnalyzer {
    /// Create a new non-linear analyzer
    pub fn new() -> Self {
        Self {
            rule_analyses: Vec::new(),
            dependencies: HashMap::new(),
            sccs: Vec::new(),
        }
    }

    /// Analyze a CHC system for non-linearity
    pub fn analyze(&mut self, system: &ChcSystem) -> NonLinearResult<()> {
        // Analyze each rule
        self.rule_analyses.clear();
        for (idx, rule) in system.rules().enumerate() {
            let mut analysis = NonLinearRuleAnalysis::analyze(rule);
            analysis.rule_id = idx;
            self.rule_analyses.push(analysis);
        }

        // Build dependency graph
        self.build_dependency_graph(system);

        // Compute SCCs to detect cycles
        self.compute_sccs();

        // Mark cyclic rules
        self.mark_cyclic_rules();

        Ok(())
    }

    /// Get statistics about non-linearity
    pub fn statistics(&self) -> NonLinearStats {
        let total_rules = self.rule_analyses.len();
        let linear_rules = self.rule_analyses.iter().filter(|a| a.is_linear()).count();
        let binary_nonlinear = self
            .rule_analyses
            .iter()
            .filter(|a| a.is_binary_nonlinear())
            .count();
        let higher_order_nonlinear = self
            .rule_analyses
            .iter()
            .filter(|a| a.nonlinearity_degree > 1)
            .count();

        let max_body_preds = self
            .rule_analyses
            .iter()
            .map(|a| a.body_pred_count)
            .max()
            .unwrap_or(0);

        NonLinearStats {
            total_rules,
            linear_rules,
            binary_nonlinear,
            higher_order_nonlinear,
            max_body_preds,
            num_sccs: self.sccs.len(),
        }
    }

    /// Build dependency graph from rules
    fn build_dependency_graph(&mut self, system: &ChcSystem) {
        self.dependencies.clear();
        for rule in system.rules() {
            if let Some(head_pred) = rule.head_predicate() {
                for app in &rule.body.predicates {
                    self.dependencies
                        .entry(head_pred)
                        .or_default()
                        .insert(app.pred);
                }
            }
        }
    }

    /// Compute strongly connected components (Tarjan's algorithm)
    fn compute_sccs(&mut self) {
        self.sccs.clear();

        // Get all predicates in the dependency graph
        let mut all_preds: HashSet<PredId> = self.dependencies.keys().copied().collect();
        for deps in self.dependencies.values() {
            all_preds.extend(deps);
        }

        if all_preds.is_empty() {
            return;
        }

        // Tarjan's algorithm state
        let mut index = 0u32;
        let mut stack = Vec::new();
        let mut indices: HashMap<PredId, u32> = HashMap::new();
        let mut lowlinks: HashMap<PredId, u32> = HashMap::new();
        let mut on_stack: HashSet<PredId> = HashSet::new();

        // Run Tarjan's algorithm for each unvisited predicate
        for &pred in &all_preds {
            if !indices.contains_key(&pred) {
                self.tarjan_strongconnect(
                    pred,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                );
            }
        }
    }

    /// Tarjan's strongconnect procedure
    fn tarjan_strongconnect(
        &mut self,
        pred: PredId,
        index: &mut u32,
        stack: &mut Vec<PredId>,
        indices: &mut HashMap<PredId, u32>,
        lowlinks: &mut HashMap<PredId, u32>,
        on_stack: &mut HashSet<PredId>,
    ) {
        // Set the depth index for pred to the smallest unused index
        indices.insert(pred, *index);
        lowlinks.insert(pred, *index);
        *index += 1;
        stack.push(pred);
        on_stack.insert(pred);

        // Consider successors of pred
        // Collect successors to avoid borrow checker issues
        let successors: Vec<PredId> = self
            .dependencies
            .get(&pred)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();

        for succ in successors {
            if !indices.contains_key(&succ) {
                // Successor has not yet been visited; recurse on it
                self.tarjan_strongconnect(succ, index, stack, indices, lowlinks, on_stack);
                let succ_lowlink = *lowlinks
                    .get(&succ)
                    .expect("lowlink exists for visited node");
                let pred_lowlink = lowlinks.get(&pred).expect("lowlink exists for predecessor");
                lowlinks.insert(pred, (*pred_lowlink).min(succ_lowlink));
            } else if on_stack.contains(&succ) {
                // Successor is in stack and hence in the current SCC
                let succ_index = *indices.get(&succ).expect("index exists for visited node");
                let pred_lowlink = lowlinks.get(&pred).expect("lowlink exists for predecessor");
                lowlinks.insert(pred, (*pred_lowlink).min(succ_index));
            }
        }

        // If pred is a root node, pop the stack and create an SCC
        if lowlinks.get(&pred) == indices.get(&pred) {
            let mut scc = HashSet::new();
            loop {
                let w = stack.pop().expect("collection validated to be non-empty");
                on_stack.remove(&w);
                scc.insert(w);
                if w == pred {
                    break;
                }
            }
            self.sccs.push(scc);
        }
    }

    /// Mark rules that are cyclic based on SCC analysis
    fn mark_cyclic_rules(&mut self) {
        // A rule is cyclic if its head and any body predicate are in the same SCC
        for analysis in &mut self.rule_analyses {
            analysis.is_cyclic = self.sccs.iter().any(|scc| {
                scc.contains(&analysis.head_pred)
                    && analysis.body_preds.iter().any(|p| scc.contains(p))
            });
        }
    }

    /// Get analysis for a specific rule
    pub fn get_analysis(&self, rule_id: usize) -> Option<&NonLinearRuleAnalysis> {
        self.rule_analyses.get(rule_id)
    }

    /// Check if the system is fully linear
    pub fn is_linear(&self) -> bool {
        self.rule_analyses.iter().all(|a| a.is_linear())
    }
}

impl Default for NonLinearAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about non-linearity in a CHC system
#[derive(Debug, Clone)]
pub struct NonLinearStats {
    /// Total number of rules
    pub total_rules: usize,
    /// Number of linear rules
    pub linear_rules: usize,
    /// Number of binary non-linear rules (2 preds in body)
    pub binary_nonlinear: usize,
    /// Number of higher-order non-linear rules (3+ preds in body)
    pub higher_order_nonlinear: usize,
    /// Maximum number of predicates in any rule body
    pub max_body_preds: usize,
    /// Number of strongly connected components
    pub num_sccs: usize,
}

/// Tree interpolant for non-linear rules
///
/// When a rule has multiple predicates in the body, we need to compute
/// a tree of interpolants rather than a single interpolant.
#[derive(Debug, Clone)]
pub struct TreeInterpolant {
    /// Interpolants for each body predicate
    pub body_interpolants: Vec<TermId>,
    /// Combined interpolant for the rule
    pub combined: TermId,
    /// Structure of the interpolation tree
    pub tree_structure: InterpolantTree,
}

/// Structure of an interpolation tree
#[derive(Debug, Clone)]
pub enum InterpolantTree {
    /// Leaf node (single predicate)
    Leaf { pred: PredId, interpolant: TermId },
    /// Binary node (two children)
    Binary {
        left: Box<InterpolantTree>,
        right: Box<InterpolantTree>,
        combined: TermId,
    },
}

/// Non-linear lemma combiner
///
/// Combines lemmas from multiple predicates in a non-linear rule body.
pub struct NonLinearLemmaCombiner;

impl NonLinearLemmaCombiner {
    /// Combine lemmas from multiple predicates
    ///
    /// Given lemmas for P1(x1), P2(x2), ..., combine them to learn a lemma for P(x)
    /// where the rule is: P1(x1) ∧ P2(x2) ∧ φ => P(x)
    pub fn combine_lemmas(
        _terms: &mut TermManager,
        _predicates: &[PredId],
        _lemmas: &[LemmaId],
        constraint: TermId,
    ) -> NonLinearResult<TermId> {
        // Combine lemmas from multiple branches of a non-linear rule
        // For a rule: P1(x1) ∧ P2(x2) ∧ φ => P(x)
        // Given lemmas for P1 and P2, combine them to learn a lemma for P

        // Basic implementation: use the constraint as the combined lemma
        // A full implementation would:
        // 1. Get the formulas for each lemma
        // 2. Combine them using conjunction
        // 3. Project out variables not in the head
        // 4. Simplify the result

        // For now, return the constraint as a conservative approximation
        Ok(constraint)
    }

    /// Compute tree interpolant for a non-linear derivation
    pub fn compute_tree_interpolant(
        terms: &mut TermManager,
        body_apps: &[PredicateApp],
        constraint: TermId,
    ) -> NonLinearResult<TreeInterpolant> {
        // Tree interpolation for non-linear rules
        // For a rule: P1(x1) ∧ P2(x2) ∧ ... ∧ Pn(xn) ∧ C => Q(y)
        // We need to compute interpolants between different partitions

        if body_apps.is_empty() {
            // No body predicates - degenerate case
            return Ok(TreeInterpolant {
                body_interpolants: Vec::new(),
                combined: terms.mk_true(),
                tree_structure: InterpolantTree::Leaf {
                    pred: PredId(0),
                    interpolant: terms.mk_true(),
                },
            });
        }

        if body_apps.len() == 1 {
            // Single body predicate - also degenerate
            // The interpolant is just the constraint
            return Ok(TreeInterpolant {
                body_interpolants: vec![constraint],
                combined: constraint,
                tree_structure: InterpolantTree::Leaf {
                    pred: body_apps[0].pred,
                    interpolant: constraint,
                },
            });
        }

        // For multiple body predicates, build a binary tree
        // Split the predicates into left and right partitions
        let mid = body_apps.len() / 2;
        let left_apps = &body_apps[..mid];
        let right_apps = &body_apps[mid..];

        // Recursively compute interpolants for left and right partitions
        let left_tree = Self::compute_tree_interpolant(terms, left_apps, constraint)?;
        let right_tree = Self::compute_tree_interpolant(terms, right_apps, constraint)?;

        // Compute interpolant between left and right partitions
        // For now, use a simple heuristic: conjoin the constraint
        let combined_interpolant = constraint;

        // Collect all body interpolants
        let mut body_interpolants = Vec::new();
        body_interpolants.extend(left_tree.body_interpolants);
        body_interpolants.extend(right_tree.body_interpolants);

        Ok(TreeInterpolant {
            body_interpolants,
            combined: combined_interpolant,
            tree_structure: InterpolantTree::Binary {
                left: Box::new(left_tree.tree_structure),
                right: Box::new(right_tree.tree_structure),
                combined: combined_interpolant,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chc::{ChcSystem, PredicateApp, RuleBody, RuleHead};
    use oxiz_core::TermManager;

    #[test]
    fn test_linear_rule_analysis() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        // Declare predicates
        let p = system.declare_predicate("P", [terms.sorts.int_sort]);

        // Linear rule: x = 0 => P(x)
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let constraint = terms.mk_eq(x, zero);

        let rule = Rule {
            id: crate::chc::RuleId(0),
            vars: vec![("x".to_string(), terms.sorts.int_sort)].into(),
            body: RuleBody::init(constraint),
            head: RuleHead::Predicate(PredicateApp::new(p, [x])),
            name: None,
        };

        let analysis = NonLinearRuleAnalysis::analyze(&rule);
        assert!(analysis.is_linear());
        assert_eq!(analysis.nonlinearity_degree, 0);
    }

    #[test]
    fn test_binary_nonlinear_rule_analysis() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        // Declare predicates
        let p = system.declare_predicate("P", [terms.sorts.int_sort]);

        // Binary non-linear rule: P(x1) ∧ P(x2) => P(x1 + x2)
        let x1 = terms.mk_var("x1", terms.sorts.int_sort);
        let x2 = terms.mk_var("x2", terms.sorts.int_sort);

        let rule = Rule {
            id: crate::chc::RuleId(0),
            vars: vec![
                ("x1".to_string(), terms.sorts.int_sort),
                ("x2".to_string(), terms.sorts.int_sort),
            ]
            .into(),
            body: RuleBody::new(
                vec![PredicateApp::new(p, [x1]), PredicateApp::new(p, [x2])],
                terms.mk_true(),
            ),
            head: RuleHead::Predicate(PredicateApp::new(p, [terms.mk_add([x1, x2])])),
            name: None,
        };

        let analysis = NonLinearRuleAnalysis::analyze(&rule);
        assert!(!analysis.is_linear());
        assert!(analysis.is_binary_nonlinear());
        assert_eq!(analysis.nonlinearity_degree, 1);
        assert_eq!(analysis.body_pred_count, 2);
    }

    #[test]
    fn test_nonlinear_analyzer() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();
        let p = system.declare_predicate("P", [terms.sorts.int_sort]);

        // Add a linear rule
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        system.add_rule(
            vec![("x".to_string(), terms.sorts.int_sort)],
            RuleBody::init(terms.mk_eq(x, zero)),
            RuleHead::Predicate(PredicateApp::new(p, [x])),
            None,
        );

        // Add a binary non-linear rule
        let x1 = terms.mk_var("x1", terms.sorts.int_sort);
        let x2 = terms.mk_var("x2", terms.sorts.int_sort);
        system.add_rule(
            vec![
                ("x1".to_string(), terms.sorts.int_sort),
                ("x2".to_string(), terms.sorts.int_sort),
            ],
            RuleBody::new(
                vec![PredicateApp::new(p, [x1]), PredicateApp::new(p, [x2])],
                terms.mk_true(),
            ),
            RuleHead::Predicate(PredicateApp::new(p, [terms.mk_add([x1, x2])])),
            None,
        );

        let mut analyzer = NonLinearAnalyzer::new();
        analyzer.analyze(&system).unwrap();

        let stats = analyzer.statistics();
        assert_eq!(stats.total_rules, 2);
        assert_eq!(stats.linear_rules, 1);
        assert_eq!(stats.binary_nonlinear, 1);
        assert_eq!(stats.max_body_preds, 2);
        assert!(!analyzer.is_linear());
    }
}
