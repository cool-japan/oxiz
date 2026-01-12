//! Term Graph for QE Analysis
//!
//! Provides term structure analysis for quantifier elimination.
//! Based on Garcia-Contreras et al. (CAV'23) term graph analysis.

use crate::ast::{TermId, TermKind, TermManager};
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for term graph analysis
#[derive(Debug, Clone)]
pub struct TermGraphConfig {
    /// Maximum depth to analyze
    pub max_depth: usize,
    /// Track term occurrences
    pub track_occurrences: bool,
    /// Compute term influence
    pub compute_influence: bool,
}

impl Default for TermGraphConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            track_occurrences: true,
            compute_influence: true,
        }
    }
}

/// Kind of term node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TermNodeKind {
    /// Variable
    Variable,
    /// Constant
    Constant,
    /// Boolean connective
    Boolean,
    /// Arithmetic operation
    Arithmetic,
    /// Comparison
    Comparison,
    /// Quantifier
    Quantifier,
    /// Other
    Other,
}

/// A node in the term graph
#[derive(Debug, Clone)]
pub struct TermNode {
    /// Term ID
    pub term_id: TermId,
    /// Kind of node
    pub kind: TermNodeKind,
    /// Children (subterm IDs)
    pub children: Vec<TermId>,
    /// Parents (superterm IDs)
    pub parents: Vec<TermId>,
    /// Number of occurrences
    pub occurrences: u32,
    /// Depth in the term tree
    pub depth: u32,
    /// Variables that appear in this term
    pub free_vars: HashSet<TermId>,
}

impl TermNode {
    /// Create a new term node
    fn new(term_id: TermId, kind: TermNodeKind) -> Self {
        Self {
            term_id,
            kind,
            children: Vec::new(),
            parents: Vec::new(),
            occurrences: 1,
            depth: 0,
            free_vars: HashSet::new(),
        }
    }
}

/// Statistics for term graph analysis
#[derive(Debug, Clone, Default)]
pub struct TermGraphStats {
    /// Number of nodes
    pub num_nodes: u64,
    /// Number of edges
    pub num_edges: u64,
    /// Maximum depth
    pub max_depth: u32,
    /// Number of variables
    pub num_variables: u64,
    /// Number of quantifiers
    pub num_quantifiers: u64,
}

/// Term graph for QE analysis
#[derive(Debug)]
pub struct TermGraph {
    /// Nodes in the graph
    nodes: HashMap<TermId, TermNode>,
    /// Root terms
    roots: Vec<TermId>,
    /// Variables in scope
    variables: HashSet<TermId>,
    /// Quantified variables
    quantified_vars: HashSet<TermId>,
    /// Configuration
    config: TermGraphConfig,
    /// Statistics
    stats: TermGraphStats,
}

impl TermGraph {
    /// Create a new term graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            variables: HashSet::new(),
            quantified_vars: HashSet::new(),
            config: TermGraphConfig::default(),
            stats: TermGraphStats::default(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: TermGraphConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            variables: HashSet::new(),
            quantified_vars: HashSet::new(),
            config,
            stats: TermGraphStats::default(),
        }
    }

    /// Build term graph from a formula
    pub fn from_formula(formula: TermId, manager: &TermManager) -> Self {
        let mut graph = Self::new();
        graph.add_formula(formula, manager);
        graph
    }

    /// Add a formula to the graph
    pub fn add_formula(&mut self, formula: TermId, manager: &TermManager) {
        self.roots.push(formula);
        self.build_graph(formula, manager, 0);
        self.compute_parents();
        self.compute_free_vars();
    }

    /// Build the graph by traversing the term
    fn build_graph(&mut self, term: TermId, manager: &TermManager, depth: u32) {
        if depth > self.config.max_depth as u32 {
            return;
        }

        if self.nodes.contains_key(&term) {
            if self.config.track_occurrences
                && let Some(node) = self.nodes.get_mut(&term)
            {
                node.occurrences += 1;
            }
            return;
        }

        let t = match manager.get(term) {
            Some(t) => t,
            None => return,
        };

        let kind = Self::classify_term(&t.kind);
        let mut node = TermNode::new(term, kind);
        node.depth = depth;

        // Track variables
        if kind == TermNodeKind::Variable {
            self.variables.insert(term);
        }

        // Track quantified variables (vars are (name, sort) tuples, not TermIds)
        if let TermKind::Forall { .. } | TermKind::Exists { .. } = &t.kind {
            // Note: quantified variables are identified by name, not TermId
            // We track the quantifier term itself as a quantified node
            self.stats.num_quantifiers += 1;
        }

        // Get children and recurse
        let children = Self::get_children(&t.kind);
        node.children = children.clone();

        self.nodes.insert(term, node);
        self.stats.num_nodes += 1;

        if depth > self.stats.max_depth {
            self.stats.max_depth = depth;
        }

        for child in children {
            self.build_graph(child, manager, depth + 1);
            self.stats.num_edges += 1;
        }
    }

    /// Classify a term kind
    fn classify_term(kind: &TermKind) -> TermNodeKind {
        match kind {
            TermKind::Var(_) => TermNodeKind::Variable,
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. } => TermNodeKind::Constant,
            TermKind::Not(_)
            | TermKind::And(_)
            | TermKind::Or(_)
            | TermKind::Xor(_, _)
            | TermKind::Implies(_, _) => TermNodeKind::Boolean,
            TermKind::Lt(_, _)
            | TermKind::Le(_, _)
            | TermKind::Gt(_, _)
            | TermKind::Ge(_, _)
            | TermKind::Eq(_, _)
            | TermKind::Distinct(_) => TermNodeKind::Comparison,
            TermKind::Add(_)
            | TermKind::Sub(_, _)
            | TermKind::Mul(_)
            | TermKind::Div(_, _)
            | TermKind::Mod(_, _)
            | TermKind::Neg(_) => TermNodeKind::Arithmetic,
            TermKind::Forall { .. } | TermKind::Exists { .. } => TermNodeKind::Quantifier,
            _ => TermNodeKind::Other,
        }
    }

    /// Get children of a term kind
    fn get_children(kind: &TermKind) -> Vec<TermId> {
        match kind {
            TermKind::Not(a) | TermKind::Neg(a) | TermKind::BvNot(a) => vec![*a],
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::Distinct(args) => args.to_vec(),
            TermKind::Xor(a, b)
            | TermKind::Implies(a, b)
            | TermKind::Eq(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::BvAnd(a, b)
            | TermKind::BvOr(a, b)
            | TermKind::BvXor(a, b)
            | TermKind::BvAdd(a, b)
            | TermKind::BvSub(a, b)
            | TermKind::BvMul(a, b)
            | TermKind::BvConcat(a, b) => vec![*a, *b],
            TermKind::Ite(a, b, c) => vec![*a, *b, *c],
            TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => vec![*body],
            _ => Vec::new(),
        }
    }

    /// Compute parent pointers
    fn compute_parents(&mut self) {
        let mut parents_map: HashMap<TermId, Vec<TermId>> = HashMap::new();

        for (term_id, node) in &self.nodes {
            for &child in &node.children {
                parents_map.entry(child).or_default().push(*term_id);
            }
        }

        for (term_id, parents) in parents_map {
            if let Some(node) = self.nodes.get_mut(&term_id) {
                node.parents = parents;
            }
        }
    }

    /// Compute free variables for each term
    fn compute_free_vars(&mut self) {
        // Process in bottom-up order (by depth, highest first)
        let mut terms_by_depth: Vec<(u32, TermId)> = self
            .nodes
            .iter()
            .map(|(&id, node)| (node.depth, id))
            .collect();
        terms_by_depth.sort_by(|a, b| b.0.cmp(&a.0)); // Descending

        for (_, term_id) in terms_by_depth {
            let children: Vec<_> = self
                .nodes
                .get(&term_id)
                .map(|n| n.children.clone())
                .unwrap_or_default();

            let is_var = self
                .nodes
                .get(&term_id)
                .map(|n| n.kind == TermNodeKind::Variable)
                .unwrap_or(false);

            let mut free_vars = HashSet::new();
            if is_var {
                free_vars.insert(term_id);
            }

            for child in children {
                if let Some(child_node) = self.nodes.get(&child) {
                    free_vars.extend(child_node.free_vars.iter().copied());
                }
            }

            if let Some(node) = self.nodes.get_mut(&term_id) {
                node.free_vars = free_vars;
            }
        }

        self.stats.num_variables = self.variables.len() as u64;
    }

    /// Identify important variables for QE (high influence)
    pub fn identify_important_variables(&self) -> Vec<TermId> {
        let mut var_scores: HashMap<TermId, u32> = HashMap::new();

        for node in self.nodes.values() {
            for &var in &node.free_vars {
                if self.variables.contains(&var) && !self.quantified_vars.contains(&var) {
                    *var_scores.entry(var).or_default() += node.occurrences;
                }
            }
        }

        let mut vars: Vec<_> = var_scores.into_iter().collect();
        vars.sort_by(|a, b| b.1.cmp(&a.1)); // Descending by score
        vars.into_iter().map(|(v, _)| v).collect()
    }

    /// Get terms containing a specific variable
    pub fn terms_with_variable(&self, var: TermId) -> Vec<TermId> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.free_vars.contains(&var))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get the node for a term
    pub fn get_node(&self, term: TermId) -> Option<&TermNode> {
        self.nodes.get(&term)
    }

    /// Get all roots
    pub fn roots(&self) -> &[TermId] {
        &self.roots
    }

    /// Get all variables
    pub fn variables(&self) -> &HashSet<TermId> {
        &self.variables
    }

    /// Get quantified variables
    pub fn quantified_vars(&self) -> &HashSet<TermId> {
        &self.quantified_vars
    }

    /// Get statistics
    pub fn stats(&self) -> &TermGraphStats {
        &self.stats
    }

    /// Check if a variable is shared (appears in multiple subformulas)
    pub fn is_shared_variable(&self, var: TermId) -> bool {
        let terms = self.terms_with_variable(var);
        terms.len() > 1
    }

    /// Get subformulas rooted at comparison terms
    pub fn comparison_subformulas(&self) -> Vec<TermId> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.kind == TermNodeKind::Comparison)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Compute the influence of a variable (how many terms depend on it)
    pub fn variable_influence(&self, var: TermId) -> u32 {
        self.terms_with_variable(var).len() as u32
    }

    /// Perform a BFS traversal from a root
    pub fn bfs_traversal(&self, root: TermId) -> Vec<TermId> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back(root);

        while let Some(term) = queue.pop_front() {
            if visited.contains(&term) {
                continue;
            }
            visited.insert(term);
            result.push(term);

            if let Some(node) = self.nodes.get(&term) {
                for &child in &node.children {
                    queue.push_back(child);
                }
            }
        }

        result
    }
}

impl Default for TermGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_graph_creation() {
        let graph = TermGraph::new();
        assert_eq!(graph.nodes.len(), 0);
        assert!(graph.roots().is_empty());
    }

    #[test]
    fn test_term_node_creation() {
        let t = TermId::from(1u32);
        let node = TermNode::new(t, TermNodeKind::Variable);
        assert_eq!(node.term_id, t);
        assert_eq!(node.kind, TermNodeKind::Variable);
        assert_eq!(node.occurrences, 1);
    }

    #[test]
    fn test_term_graph_config() {
        let config = TermGraphConfig::default();
        assert_eq!(config.max_depth, 100);
        assert!(config.track_occurrences);
        assert!(config.compute_influence);
    }

    #[test]
    fn test_classify_term() {
        use lasso::Key;
        assert_eq!(
            TermGraph::classify_term(&TermKind::Var(
                lasso::Spur::try_from_usize(0).expect("valid")
            )),
            TermNodeKind::Variable
        );
        assert_eq!(
            TermGraph::classify_term(&TermKind::True),
            TermNodeKind::Constant
        );
        assert_eq!(
            TermGraph::classify_term(&TermKind::IntConst(42.into())),
            TermNodeKind::Constant
        );
    }

    #[test]
    fn test_term_graph_stats() {
        let stats = TermGraphStats::default();
        assert_eq!(stats.num_nodes, 0);
        assert_eq!(stats.num_edges, 0);
        assert_eq!(stats.max_depth, 0);
    }

    #[test]
    fn test_empty_graph_operations() {
        let graph = TermGraph::new();
        let t = TermId::from(1u32);

        assert!(graph.get_node(t).is_none());
        assert!(graph.terms_with_variable(t).is_empty());
        assert!(!graph.is_shared_variable(t));
        assert_eq!(graph.variable_influence(t), 0);
    }

    #[test]
    fn test_bfs_traversal_empty() {
        let graph = TermGraph::new();
        let t = TermId::from(1u32);
        let result = graph.bfs_traversal(t);
        assert_eq!(result, vec![t]);
    }

    #[test]
    fn test_comparison_subformulas_empty() {
        let graph = TermGraph::new();
        assert!(graph.comparison_subformulas().is_empty());
    }
}
