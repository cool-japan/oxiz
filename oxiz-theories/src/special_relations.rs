//! Special Relations Theory
//!
//! Supports reasoning about binary relations with special properties:
//! - Linear Orders (LO): Total, transitive, antisymmetric, reflexive
//! - Partial Orders (PO): Transitive, antisymmetric, reflexive
//! - Piecewise Linear Orders (PLO): PO + left/right tree properties
//! - Tree Orders (TO): PO + right tree property
//! - Transitive Closures (TC): Transitive property
//!
//! # Properties
//!
//! - **Transitive**: R(x,y) ∧ R(y,z) → R(x,z)
//! - **Reflexive**: R(x,x)
//! - **Antisymmetric**: R(x,y) ∧ R(y,x) → x = y
//! - **Total**: R(x,y) ∨ R(y,x)
//! - **LeftTree**: R(y,x) ∧ R(z,x) → R(y,z) ∨ R(z,y)
//! - **RightTree**: R(x,y) ∧ R(x,z) → R(y,z) ∨ R(z,y)

use lasso::Spur;
use oxiz_core::SortId;
use oxiz_core::ast::TermId;
use std::collections::{HashMap, HashSet};

/// Properties that a relation can have
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelationProperties {
    bits: u8,
}

impl RelationProperties {
    /// No properties
    pub const NONE: Self = Self { bits: 0x00 };
    /// Transitive: R(x,y) ∧ R(y,z) → R(x,z)
    pub const TRANSITIVE: Self = Self { bits: 0x01 };
    /// Reflexive: R(x,x)
    pub const REFLEXIVE: Self = Self { bits: 0x02 };
    /// Antisymmetric: R(x,y) ∧ R(y,x) → x = y
    pub const ANTISYMMETRIC: Self = Self { bits: 0x04 };
    /// Left tree: R(y,x) ∧ R(z,x) → R(y,z) ∨ R(z,y)
    pub const LEFT_TREE: Self = Self { bits: 0x08 };
    /// Right tree: R(x,y) ∧ R(x,z) → R(y,z) ∨ R(z,y)
    pub const RIGHT_TREE: Self = Self { bits: 0x10 };
    /// Total: R(x,y) ∨ R(y,x)
    pub const TOTAL: Self = Self { bits: 0x20 };

    /// Combine properties
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Check if a property is set
    pub const fn has(self, prop: Self) -> bool {
        (self.bits & prop.bits) == prop.bits
    }

    /// Check if transitive
    pub const fn is_transitive(self) -> bool {
        self.has(Self::TRANSITIVE)
    }

    /// Check if reflexive
    pub const fn is_reflexive(self) -> bool {
        self.has(Self::REFLEXIVE)
    }

    /// Check if antisymmetric
    pub const fn is_antisymmetric(self) -> bool {
        self.has(Self::ANTISYMMETRIC)
    }

    /// Check if total
    pub const fn is_total(self) -> bool {
        self.has(Self::TOTAL)
    }
}

/// Kind of special relation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationKind {
    /// Linear Order (total, transitive, antisymmetric, reflexive)
    LinearOrder,
    /// Partial Order (transitive, antisymmetric, reflexive)
    PartialOrder,
    /// Piecewise Linear Order (PO + left tree + right tree)
    PiecewiseLinearOrder,
    /// Tree Order (transitive, antisymmetric, reflexive, right tree)
    TreeOrder,
    /// Transitive Closure
    TransitiveClosure,
}

impl RelationKind {
    /// Get the properties for this relation kind
    pub fn properties(self) -> RelationProperties {
        match self {
            RelationKind::LinearOrder => RelationProperties::TRANSITIVE
                .union(RelationProperties::REFLEXIVE)
                .union(RelationProperties::ANTISYMMETRIC)
                .union(RelationProperties::TOTAL),
            RelationKind::PartialOrder => RelationProperties::TRANSITIVE
                .union(RelationProperties::REFLEXIVE)
                .union(RelationProperties::ANTISYMMETRIC),
            RelationKind::PiecewiseLinearOrder => RelationProperties::TRANSITIVE
                .union(RelationProperties::REFLEXIVE)
                .union(RelationProperties::ANTISYMMETRIC)
                .union(RelationProperties::LEFT_TREE)
                .union(RelationProperties::RIGHT_TREE),
            RelationKind::TreeOrder => RelationProperties::TRANSITIVE
                .union(RelationProperties::REFLEXIVE)
                .union(RelationProperties::ANTISYMMETRIC)
                .union(RelationProperties::RIGHT_TREE),
            RelationKind::TransitiveClosure => RelationProperties::TRANSITIVE,
        }
    }
}

/// Edge in the relation graph (x R y)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelationEdge {
    /// Source term
    pub from: TermId,
    /// Destination term
    pub to: TermId,
}

impl RelationEdge {
    /// Create a new relation edge
    pub fn new(from: TermId, to: TermId) -> Self {
        Self { from, to }
    }
}

/// Definition of a special relation
#[derive(Debug, Clone)]
pub struct RelationDef {
    /// Relation name
    pub name: Spur,
    /// Domain sort
    pub domain: SortId,
    /// Relation kind
    pub kind: RelationKind,
    /// Properties
    pub properties: RelationProperties,
}

impl RelationDef {
    /// Create a new relation definition
    pub fn new(name: Spur, domain: SortId, kind: RelationKind) -> Self {
        Self {
            name,
            domain,
            kind,
            properties: kind.properties(),
        }
    }
}

/// Statistics for special relations
#[derive(Debug, Clone, Default)]
pub struct SpecialRelationStats {
    /// Number of edges added
    pub num_edges: usize,
    /// Number of transitive closures computed
    pub num_closures: usize,
    /// Number of conflicts found
    pub num_conflicts: usize,
    /// Number of propagations
    pub num_propagations: usize,
}

impl SpecialRelationStats {
    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Special relation solver
pub struct SpecialRelationSolver {
    /// Relation definitions
    relations: HashMap<Spur, RelationDef>,
    /// Edges for each relation (relation_name -> set of edges)
    edges: HashMap<Spur, HashSet<RelationEdge>>,
    /// Transitive closure cache (relation_name -> closure)
    closures: HashMap<Spur, HashSet<RelationEdge>>,
    /// Statistics
    stats: SpecialRelationStats,
    /// Context stack for push/pop
    context_stack: Vec<usize>,
}

impl SpecialRelationSolver {
    /// Create a new special relation solver
    pub fn new() -> Self {
        Self {
            relations: HashMap::new(),
            edges: HashMap::new(),
            closures: HashMap::new(),
            stats: SpecialRelationStats::default(),
            context_stack: Vec::new(),
        }
    }

    /// Define a new special relation
    pub fn define_relation(&mut self, name: Spur, domain: SortId, kind: RelationKind) {
        let def = RelationDef::new(name, domain, kind);
        self.relations.insert(name, def);
        self.edges.insert(name, HashSet::new());
        self.closures.insert(name, HashSet::new());
    }

    /// Check if a relation is defined
    pub fn is_defined(&self, name: Spur) -> bool {
        self.relations.contains_key(&name)
    }

    /// Get a relation definition
    pub fn get_relation(&self, name: Spur) -> Option<&RelationDef> {
        self.relations.get(&name)
    }

    /// Add an edge R(from, to)
    pub fn add_edge(&mut self, relation: Spur, from: TermId, to: TermId) -> bool {
        let edge = RelationEdge::new(from, to);

        if let Some(edge_set) = self.edges.get_mut(&relation) {
            if edge_set.insert(edge) {
                self.stats.num_edges = self.stats.num_edges.saturating_add(1);
                // Invalidate closure cache
                if let Some(c) = self.closures.get_mut(&relation) {
                    c.clear();
                }
                true
            } else {
                false // Edge already exists
            }
        } else {
            false // Relation not defined
        }
    }

    /// Check if an edge exists R(from, to)
    pub fn has_edge(&self, relation: Spur, from: TermId, to: TermId) -> bool {
        let edge = RelationEdge::new(from, to);
        self.edges
            .get(&relation)
            .is_some_and(|edges| edges.contains(&edge))
    }

    /// Compute transitive closure of a relation
    pub fn compute_closure(&mut self, relation: Spur) -> Option<HashSet<RelationEdge>> {
        // Check if cached
        if let Some(closure) = self.closures.get(&relation)
            && !closure.is_empty()
        {
            return Some(closure.clone());
        }

        let edges = self.edges.get(&relation)?.clone();
        let def = self.relations.get(&relation)?;

        if !def.properties.is_transitive() {
            // Non-transitive relations don't have meaningful closures
            return Some(edges);
        }

        self.stats.num_closures = self.stats.num_closures.saturating_add(1);

        // Warshall's algorithm for transitive closure
        let mut closure: HashSet<_> = edges.clone();

        // Add reflexive edges if needed
        if def.properties.is_reflexive() {
            let all_nodes: HashSet<_> = edges.iter().flat_map(|e| vec![e.from, e.to]).collect();
            for &node in &all_nodes {
                closure.insert(RelationEdge::new(node, node));
            }
        }

        // Compute transitive closure
        let mut changed = true;
        while changed {
            changed = false;
            let current = closure.clone();

            for e1 in &current {
                for e2 in &current {
                    if e1.to == e2.from {
                        // R(e1.from, e1.to) ∧ R(e1.to, e2.to) → R(e1.from, e2.to)
                        let new_edge = RelationEdge::new(e1.from, e2.to);
                        if closure.insert(new_edge) {
                            changed = true;
                        }
                    }
                }
            }
        }

        self.closures.insert(relation, closure.clone());
        Some(closure)
    }

    /// Check for antisymmetry violations
    ///
    /// Returns conflicts: (x, y) where R(x,y) ∧ R(y,x) but x ≠ y
    pub fn check_antisymmetry(&mut self, relation: Spur) -> Vec<(TermId, TermId)> {
        let _def = match self.relations.get(&relation) {
            Some(d) if d.properties.is_antisymmetric() => d,
            _ => return Vec::new(),
        };

        let closure = match self.compute_closure(relation) {
            Some(c) => c,
            None => return Vec::new(),
        };

        let mut violations = Vec::new();

        for edge in &closure {
            let reverse = RelationEdge::new(edge.to, edge.from);
            if edge.from != edge.to && closure.contains(&reverse) {
                // R(x,y) ∧ R(y,x) → x = y (antisymmetry)
                // This is a conflict if x ≠ y
                violations.push((edge.from, edge.to));
            }
        }

        if !violations.is_empty() {
            self.stats.num_conflicts = self.stats.num_conflicts.saturating_add(1);
        }

        violations
    }

    /// Propagate consequences based on relation properties
    ///
    /// Returns new edges that can be derived
    pub fn propagate(&mut self, relation: Spur) -> Vec<RelationEdge> {
        let _def = match self.relations.get(&relation) {
            Some(d) => d.clone(),
            None => return Vec::new(),
        };

        let mut new_edges = Vec::new();

        // Get current edges before computing closure
        let current_edges = self.edges.get(&relation).cloned().unwrap_or_default();

        // Compute closure to get all implied edges
        if let Some(closure) = self.compute_closure(relation) {
            for edge in &closure {
                if !current_edges.contains(edge) {
                    new_edges.push(*edge);
                }
            }
        }

        if !new_edges.is_empty() {
            self.stats.num_propagations =
                self.stats.num_propagations.saturating_add(new_edges.len());
        }

        new_edges
    }

    /// Check totality property
    ///
    /// For a total relation, R(x,y) ∨ R(y,x) must hold for all x, y
    /// Returns pairs (x, y) that need branching
    pub fn get_totality_branches(&self, relation: Spur) -> Vec<(TermId, TermId)> {
        let _def = match self.relations.get(&relation) {
            Some(d) if d.properties.is_total() => d,
            _ => return Vec::new(),
        };

        let edges = match self.edges.get(&relation) {
            Some(e) => e,
            None => return Vec::new(),
        };

        // Collect all nodes
        let all_nodes: HashSet<_> = edges.iter().flat_map(|e| vec![e.from, e.to]).collect();

        let mut branches = Vec::new();

        for &x in &all_nodes {
            for &y in &all_nodes {
                if x != y {
                    let fwd = RelationEdge::new(x, y);
                    let rev = RelationEdge::new(y, x);

                    // If neither R(x,y) nor R(y,x), need to branch
                    if !edges.contains(&fwd) && !edges.contains(&rev) {
                        branches.push((x, y));
                    }
                }
            }
        }

        branches
    }

    /// Push a new context level
    pub fn push(&mut self) {
        self.context_stack.push(self.edges.len());
    }

    /// Pop context levels
    pub fn pop(&mut self, levels: usize) {
        for _ in 0..levels {
            if self.context_stack.pop().is_some() {
                // Simplified pop - in production would track exact edge additions
                // For now, just clear closures to force recomputation
                for closure in self.closures.values_mut() {
                    closure.clear();
                }
            }
        }
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.edges.clear();
        self.closures.clear();
        self.context_stack.clear();
        self.stats.reset();
    }

    /// Get statistics
    pub fn stats(&self) -> &SpecialRelationStats {
        &self.stats
    }
}

impl Default for SpecialRelationSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lasso::Key;

    #[test]
    fn test_relation_properties() {
        let props = RelationProperties::TRANSITIVE.union(RelationProperties::REFLEXIVE);
        assert!(props.is_transitive());
        assert!(props.is_reflexive());
        assert!(!props.is_antisymmetric());
        assert!(!props.is_total());
    }

    #[test]
    fn test_linear_order_properties() {
        let props = RelationKind::LinearOrder.properties();
        assert!(props.is_transitive());
        assert!(props.is_reflexive());
        assert!(props.is_antisymmetric());
        assert!(props.is_total());
    }

    #[test]
    fn test_partial_order_properties() {
        let props = RelationKind::PartialOrder.properties();
        assert!(props.is_transitive());
        assert!(props.is_reflexive());
        assert!(props.is_antisymmetric());
        assert!(!props.is_total());
    }

    #[test]
    fn test_solver_creation() {
        let solver = SpecialRelationSolver::new();
        assert_eq!(solver.stats().num_edges, 0);
    }

    #[test]
    fn test_define_relation() {
        let mut solver = SpecialRelationSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let domain = SortId::new(0);

        solver.define_relation(name, domain, RelationKind::PartialOrder);
        assert!(solver.is_defined(name));

        let def = solver.get_relation(name).expect("relation exists");
        assert!(def.properties.is_transitive());
        assert!(def.properties.is_reflexive());
    }

    #[test]
    fn test_add_edge() {
        let mut solver = SpecialRelationSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let domain = SortId::new(0);

        solver.define_relation(name, domain, RelationKind::PartialOrder);

        let x = TermId::new(1);
        let y = TermId::new(2);

        assert!(solver.add_edge(name, x, y));
        assert!(solver.has_edge(name, x, y));
        assert!(!solver.has_edge(name, y, x));
        assert_eq!(solver.stats().num_edges, 1);
    }

    #[test]
    fn test_transitive_closure() {
        let mut solver = SpecialRelationSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let domain = SortId::new(0);

        solver.define_relation(name, domain, RelationKind::PartialOrder);

        let x = TermId::new(1);
        let y = TermId::new(2);
        let z = TermId::new(3);

        // Add R(x,y) and R(y,z)
        solver.add_edge(name, x, y);
        solver.add_edge(name, y, z);

        // Compute closure - should include R(x,z)
        let closure = solver.compute_closure(name).expect("closure exists");

        // Check transitive edge exists
        assert!(closure.contains(&RelationEdge::new(x, z)));
        // Check reflexive edges (PO is reflexive)
        assert!(closure.contains(&RelationEdge::new(x, x)));
        assert!(closure.contains(&RelationEdge::new(y, y)));
        assert!(closure.contains(&RelationEdge::new(z, z)));
    }

    #[test]
    fn test_antisymmetry_check() {
        let mut solver = SpecialRelationSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let domain = SortId::new(0);

        solver.define_relation(name, domain, RelationKind::PartialOrder);

        let x = TermId::new(1);
        let y = TermId::new(2);

        // Add R(x,y) and R(y,x) - violates antisymmetry
        solver.add_edge(name, x, y);
        solver.add_edge(name, y, x);

        let violations = solver.check_antisymmetry(name);
        assert!(!violations.is_empty());
        assert!(violations.contains(&(x, y)) || violations.contains(&(y, x)));
    }

    #[test]
    fn test_propagate() {
        let mut solver = SpecialRelationSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let domain = SortId::new(0);

        solver.define_relation(name, domain, RelationKind::PartialOrder);

        let x = TermId::new(1);
        let y = TermId::new(2);
        let z = TermId::new(3);

        solver.add_edge(name, x, y);
        solver.add_edge(name, y, z);

        let new_edges = solver.propagate(name);

        // Should propagate R(x,z) and reflexive edges
        assert!(!new_edges.is_empty());
    }

    #[test]
    fn test_totality_branches() {
        let mut solver = SpecialRelationSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let domain = SortId::new(0);

        solver.define_relation(name, domain, RelationKind::LinearOrder);

        let x = TermId::new(1);
        let y = TermId::new(2);
        let z = TermId::new(3);

        // Add R(x,y) and R(x,z) to create 3 nodes
        solver.add_edge(name, x, y);
        solver.add_edge(name, x, z);

        let branches = solver.get_totality_branches(name);

        // Should need to branch on (y,z) or (z,y) since neither is present
        assert!(!branches.is_empty());
        // The pair (y, z) or (z, y) should be in branches
        assert!(branches.contains(&(y, z)) || branches.contains(&(z, y)));
    }

    #[test]
    fn test_push_pop() {
        let mut solver = SpecialRelationSolver::new();
        let name = Spur::try_from_usize(0).expect("valid spur");
        let domain = SortId::new(0);

        solver.define_relation(name, domain, RelationKind::PartialOrder);

        let x = TermId::new(1);
        let y = TermId::new(2);

        solver.push();
        solver.add_edge(name, x, y);
        assert!(solver.has_edge(name, x, y));

        solver.pop(1);
        // After pop, closures should be cleared
    }
}
