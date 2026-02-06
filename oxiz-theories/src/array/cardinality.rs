//! Array Cardinality Constraints
//!
//! This module implements cardinality constraints for arrays, including:
//! - Domain cardinality bounds
//! - Range cardinality constraints
//! - Finite array reasoning
//! - Pigeonhole principle applications
//!
//! Reference: Z3's array cardinality reasoning and finite model finding

use oxiz_core::ast::TermId;
use oxiz_core::error::{OxizError, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// Cardinality bound for a set/domain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CardinalityBound {
    /// Exactly n elements
    Exact(u64),
    /// At most n elements
    AtMost(u64),
    /// At least n elements
    AtLeast(u64),
    /// Between min and max elements
    Range { min: u64, max: u64 },
    /// Infinite (unbounded)
    Infinite,
}

impl fmt::Display for CardinalityBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CardinalityBound::Exact(n) => write!(f, "= {}", n),
            CardinalityBound::AtMost(n) => write!(f, "≤ {}", n),
            CardinalityBound::AtLeast(n) => write!(f, "≥ {}", n),
            CardinalityBound::Range { min, max } => write!(f, "[{}, {}]", min, max),
            CardinalityBound::Infinite => write!(f, "∞"),
        }
    }
}

impl CardinalityBound {
    /// Check if this bound is finite
    pub fn is_finite(&self) -> bool {
        !matches!(self, CardinalityBound::Infinite)
    }

    /// Get the maximum value (if finite)
    pub fn max_value(&self) -> Option<u64> {
        match self {
            CardinalityBound::Exact(n) | CardinalityBound::AtMost(n) => Some(*n),
            CardinalityBound::Range { max, .. } => Some(*max),
            CardinalityBound::AtLeast(_) | CardinalityBound::Infinite => None,
        }
    }

    /// Get the minimum value
    pub fn min_value(&self) -> u64 {
        match self {
            CardinalityBound::Exact(n) | CardinalityBound::AtLeast(n) => *n,
            CardinalityBound::Range { min, .. } => *min,
            CardinalityBound::AtMost(_) | CardinalityBound::Infinite => 0,
        }
    }

    /// Intersect two cardinality bounds
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (CardinalityBound::Exact(n1), CardinalityBound::Exact(n2)) => {
                if n1 == n2 {
                    Some(CardinalityBound::Exact(*n1))
                } else {
                    None // Inconsistent
                }
            }
            (CardinalityBound::Exact(n), CardinalityBound::AtMost(max))
            | (CardinalityBound::AtMost(max), CardinalityBound::Exact(n)) => {
                if n <= max {
                    Some(CardinalityBound::Exact(*n))
                } else {
                    None
                }
            }
            (CardinalityBound::AtMost(n1), CardinalityBound::AtMost(n2)) => {
                Some(CardinalityBound::AtMost((*n1).min(*n2)))
            }
            (CardinalityBound::AtLeast(n1), CardinalityBound::AtLeast(n2)) => {
                Some(CardinalityBound::AtLeast((*n1).max(*n2)))
            }
            (
                CardinalityBound::Range {
                    min: min1,
                    max: max1,
                },
                CardinalityBound::Range {
                    min: min2,
                    max: max2,
                },
            ) => {
                let min = (*min1).max(*min2);
                let max = (*max1).min(*max2);
                if min <= max {
                    Some(CardinalityBound::Range { min, max })
                } else {
                    None
                }
            }
            (CardinalityBound::Infinite, other) | (other, CardinalityBound::Infinite) => {
                Some(*other)
            }
            _ => {
                // Handle mixed cases
                let min = self.min_value().max(other.min_value());
                if let (Some(max1), Some(max2)) = (self.max_value(), other.max_value()) {
                    let max = max1.min(max2);
                    if min <= max {
                        Some(CardinalityBound::Range { min, max })
                    } else {
                        None
                    }
                } else {
                    Some(CardinalityBound::AtLeast(min))
                }
            }
        }
    }
}

/// Cardinality constraint for arrays
#[derive(Debug, Clone)]
pub struct ArrayCardinalityConstraint {
    /// Array variable
    pub array: u32,
    /// Domain cardinality (number of valid indices)
    pub domain_cardinality: CardinalityBound,
    /// Range cardinality (number of distinct values)
    pub range_cardinality: CardinalityBound,
    /// Number of modified locations
    pub modified_locations: Option<CardinalityBound>,
}

impl ArrayCardinalityConstraint {
    /// Create a new cardinality constraint
    pub fn new(array: u32) -> Self {
        Self {
            array,
            domain_cardinality: CardinalityBound::Infinite,
            range_cardinality: CardinalityBound::Infinite,
            modified_locations: None,
        }
    }

    /// Set domain cardinality
    pub fn with_domain_cardinality(mut self, bound: CardinalityBound) -> Self {
        self.domain_cardinality = bound;
        self
    }

    /// Set range cardinality
    pub fn with_range_cardinality(mut self, bound: CardinalityBound) -> Self {
        self.range_cardinality = bound;
        self
    }

    /// Set modified locations bound
    pub fn with_modified_locations(mut self, bound: CardinalityBound) -> Self {
        self.modified_locations = Some(bound);
        self
    }

    /// Check if this is a finite array
    pub fn is_finite(&self) -> bool {
        self.domain_cardinality.is_finite()
    }
}

/// Cardinality constraint manager for arrays
pub struct CardinalityConstraintManager {
    /// Constraints for each array
    constraints: FxHashMap<u32, ArrayCardinalityConstraint>,
    /// Global domain bounds (for index sorts)
    domain_bounds: FxHashMap<u32, CardinalityBound>,
    /// Pigeonhole constraints
    pigeonhole_constraints: Vec<PigeonholeConstraint>,
}

/// Pigeonhole principle constraint
#[derive(Debug, Clone)]
pub struct PigeonholeConstraint {
    /// Number of pigeons
    pub num_pigeons: u64,
    /// Number of holes
    pub num_holes: u64,
    /// Array representing the mapping
    pub array: u32,
    /// Indices (pigeons)
    pub indices: Vec<u32>,
}

impl Default for CardinalityConstraintManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CardinalityConstraintManager {
    /// Create a new manager
    pub fn new() -> Self {
        Self {
            constraints: FxHashMap::default(),
            domain_bounds: FxHashMap::default(),
            pigeonhole_constraints: Vec::new(),
        }
    }

    /// Add a cardinality constraint
    pub fn add_constraint(&mut self, constraint: ArrayCardinalityConstraint) {
        self.constraints.insert(constraint.array, constraint);
    }

    /// Get constraint for an array
    pub fn get_constraint(&self, array: u32) -> Option<&ArrayCardinalityConstraint> {
        self.constraints.get(&array)
    }

    /// Set domain bound for a sort
    pub fn set_domain_bound(&mut self, sort: u32, bound: CardinalityBound) {
        self.domain_bounds.insert(sort, bound);
    }

    /// Get domain bound for a sort
    pub fn get_domain_bound(&self, sort: u32) -> CardinalityBound {
        self.domain_bounds
            .get(&sort)
            .copied()
            .unwrap_or(CardinalityBound::Infinite)
    }

    /// Add a pigeonhole constraint
    pub fn add_pigeonhole(&mut self, constraint: PigeonholeConstraint) -> Result<()> {
        // Check if constraint is satisfiable
        if constraint.num_pigeons > constraint.num_holes {
            return Err(OxizError::Unsupported(
                "Pigeonhole principle violation".to_string(),
            ));
        }
        self.pigeonhole_constraints.push(constraint);
        Ok(())
    }

    /// Check if pigeonhole principle applies
    pub fn check_pigeonhole(&self) -> Option<Vec<TermId>> {
        for constraint in &self.pigeonhole_constraints {
            if constraint.num_pigeons > constraint.num_holes {
                // Violation found
                return Some(vec![TermId::new(0)]); // Would return actual conflict
            }
        }
        None
    }

    /// Infer cardinality bounds from array operations
    pub fn infer_bounds(&mut self, array: u32, operations: &[ArrayOperation]) {
        let mut constraint = self
            .constraints
            .remove(&array)
            .unwrap_or_else(|| ArrayCardinalityConstraint::new(array));

        let mut modified_locations = FxHashSet::default();
        let mut range_values = FxHashSet::default();

        for op in operations {
            match op {
                ArrayOperation::Store { indices, value, .. } => {
                    modified_locations.insert(indices.clone());
                    range_values.insert(*value);
                }
                ArrayOperation::Select { .. } => {
                    // Doesn't affect cardinality
                }
            }
        }

        // Update bounds based on inferred information
        if !modified_locations.is_empty() {
            let count = modified_locations.len() as u64;
            constraint.modified_locations = Some(CardinalityBound::AtLeast(count));
        }

        if !range_values.is_empty() {
            let count = range_values.len() as u64;
            let new_bound = CardinalityBound::AtLeast(count);
            constraint.range_cardinality =
                if let Some(existing) = constraint.range_cardinality.intersect(&new_bound) {
                    existing
                } else {
                    new_bound
                };
        }

        self.constraints.insert(array, constraint);
    }

    /// Generate finite model finding constraints
    pub fn generate_finite_model_constraints(&self, array: u32) -> Vec<FiniteModelConstraint> {
        let mut constraints = Vec::new();

        if let Some(constraint) = self.constraints.get(&array) {
            if let Some(max_domain) = constraint.domain_cardinality.max_value() {
                // Generate domain enumeration
                constraints.push(FiniteModelConstraint::DomainEnumeration {
                    array,
                    max_size: max_domain,
                });
            }

            if let Some(max_range) = constraint.range_cardinality.max_value() {
                // Generate range bound
                constraints.push(FiniteModelConstraint::RangeBound {
                    array,
                    max_distinct_values: max_range,
                });
            }
        }

        constraints
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.domain_bounds.clear();
        self.pigeonhole_constraints.clear();
    }
}

/// Array operation for cardinality analysis
#[derive(Debug, Clone)]
pub enum ArrayOperation {
    /// Select operation
    Select { array: u32, indices: Vec<u32> },
    /// Store operation
    Store {
        array: u32,
        indices: Vec<u32>,
        value: u32,
    },
}

/// Finite model finding constraint
#[derive(Debug, Clone)]
pub enum FiniteModelConstraint {
    /// Enumerate domain up to a maximum size
    DomainEnumeration { array: u32, max_size: u64 },
    /// Bound the number of distinct range values
    RangeBound {
        array: u32,
        max_distinct_values: u64,
    },
    /// Symmetry breaking for finite arrays
    SymmetryBreaking {
        array: u32,
        pattern: SymmetryPattern,
    },
}

/// Symmetry pattern for finite arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryPattern {
    /// Lexicographic ordering
    Lexicographic,
    /// Value-based ordering
    ValueOrdering,
    /// Index-based ordering
    IndexOrdering,
}

/// Cardinality reasoning engine
pub struct CardinalityReasoner {
    /// Constraint manager
    manager: CardinalityConstraintManager,
    /// Derived bounds
    derived_bounds: FxHashMap<u32, CardinalityBound>,
    /// Conflict detection
    conflicts: Vec<CardinalityConflict>,
}

/// Cardinality conflict
#[derive(Debug, Clone)]
pub struct CardinalityConflict {
    /// Array involved
    pub array: u32,
    /// Conflicting bounds
    pub bounds: Vec<CardinalityBound>,
    /// Explanation
    pub explanation: String,
}

impl Default for CardinalityReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl CardinalityReasoner {
    /// Create a new reasoner
    pub fn new() -> Self {
        Self {
            manager: CardinalityConstraintManager::new(),
            derived_bounds: FxHashMap::default(),
            conflicts: Vec::new(),
        }
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: ArrayCardinalityConstraint) {
        self.manager.add_constraint(constraint);
    }

    /// Propagate cardinality constraints
    pub fn propagate(&mut self) -> Result<Vec<CardinalityImplication>> {
        let mut implications = Vec::new();

        // Check for pigeonhole violations
        if let Some(_conflict) = self.manager.check_pigeonhole() {
            return Err(OxizError::Unsupported(
                "Pigeonhole principle violation".to_string(),
            ));
        }

        // Derive bounds through constraint propagation
        for (&array, constraint) in &self.manager.constraints {
            // Domain-range relationship
            if let (Some(domain_max), Some(range_max)) = (
                constraint.domain_cardinality.max_value(),
                constraint.range_cardinality.max_value(),
            ) && range_max > domain_max
            {
                // More distinct values than domain elements is impossible
                self.conflicts.push(CardinalityConflict {
                    array,
                    bounds: vec![constraint.domain_cardinality, constraint.range_cardinality],
                    explanation: "Range cardinality exceeds domain cardinality".to_string(),
                });
                return Err(OxizError::Unsupported(
                    "Cardinality conflict detected".to_string(),
                ));
            }

            // Also check if range minimum exceeds domain maximum
            if let Some(domain_max) = constraint.domain_cardinality.max_value() {
                let range_min = constraint.range_cardinality.min_value();
                if range_min > domain_max {
                    self.conflicts.push(CardinalityConflict {
                        array,
                        bounds: vec![constraint.domain_cardinality, constraint.range_cardinality],
                        explanation: "Range minimum exceeds domain maximum".to_string(),
                    });
                    return Err(OxizError::Unsupported(
                        "Cardinality conflict detected".to_string(),
                    ));
                }
            }

            // Modified locations bound
            if let Some(modified_bound) = &constraint.modified_locations {
                if let Some(domain_max) = constraint.domain_cardinality.max_value()
                    && modified_bound.min_value() > domain_max
                {
                    return Err(OxizError::Unsupported(
                        "Modified locations exceed domain".to_string(),
                    ));
                }

                // If all domain elements are modified, infer range bound
                if let CardinalityBound::Exact(n) = constraint.domain_cardinality
                    && modified_bound.min_value() == n
                {
                    implications.push(CardinalityImplication {
                        array,
                        bound_type: BoundType::Range,
                        old_bound: constraint.range_cardinality,
                        new_bound: CardinalityBound::AtMost(n),
                    });
                }
            }
        }

        Ok(implications)
    }

    /// Get manager
    pub fn manager(&self) -> &CardinalityConstraintManager {
        &self.manager
    }

    /// Get mutable manager
    pub fn manager_mut(&mut self) -> &mut CardinalityConstraintManager {
        &mut self.manager
    }

    /// Get conflicts
    pub fn get_conflicts(&self) -> &[CardinalityConflict] {
        &self.conflicts
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.manager.clear();
        self.derived_bounds.clear();
        self.conflicts.clear();
    }
}

/// Cardinality implication (derived bound)
#[derive(Debug, Clone)]
pub struct CardinalityImplication {
    /// Array affected
    pub array: u32,
    /// Type of bound
    pub bound_type: BoundType,
    /// Old bound
    pub old_bound: CardinalityBound,
    /// New bound
    pub new_bound: CardinalityBound,
}

/// Type of cardinality bound
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundType {
    /// Domain bound
    Domain,
    /// Range bound
    Range,
    /// Modified locations bound
    ModifiedLocations,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cardinality_bound_exact() {
        let bound = CardinalityBound::Exact(10);
        assert!(bound.is_finite());
        assert_eq!(bound.max_value(), Some(10));
        assert_eq!(bound.min_value(), 10);
    }

    #[test]
    fn test_cardinality_bound_at_most() {
        let bound = CardinalityBound::AtMost(5);
        assert_eq!(bound.max_value(), Some(5));
        assert_eq!(bound.min_value(), 0);
    }

    #[test]
    fn test_cardinality_bound_range() {
        let bound = CardinalityBound::Range { min: 5, max: 10 };
        assert_eq!(bound.min_value(), 5);
        assert_eq!(bound.max_value(), Some(10));
    }

    #[test]
    fn test_cardinality_bound_infinite() {
        let bound = CardinalityBound::Infinite;
        assert!(!bound.is_finite());
        assert_eq!(bound.max_value(), None);
    }

    #[test]
    fn test_bound_intersection_exact() {
        let b1 = CardinalityBound::Exact(5);
        let b2 = CardinalityBound::Exact(5);
        let result = b1.intersect(&b2);
        assert_eq!(result, Some(CardinalityBound::Exact(5)));

        let b3 = CardinalityBound::Exact(10);
        let result2 = b1.intersect(&b3);
        assert_eq!(result2, None);
    }

    #[test]
    fn test_bound_intersection_at_most() {
        let b1 = CardinalityBound::AtMost(10);
        let b2 = CardinalityBound::AtMost(5);
        let result = b1.intersect(&b2);
        assert_eq!(result, Some(CardinalityBound::AtMost(5)));
    }

    #[test]
    fn test_array_cardinality_constraint() {
        let constraint = ArrayCardinalityConstraint::new(100)
            .with_domain_cardinality(CardinalityBound::Exact(10))
            .with_range_cardinality(CardinalityBound::AtMost(5));

        assert_eq!(constraint.array, 100);
        assert!(constraint.is_finite());
    }

    #[test]
    fn test_constraint_manager() {
        let mut manager = CardinalityConstraintManager::new();

        let constraint = ArrayCardinalityConstraint::new(100)
            .with_domain_cardinality(CardinalityBound::Exact(10));

        manager.add_constraint(constraint);

        let retrieved = manager.get_constraint(100);
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().is_finite());
    }

    #[test]
    fn test_pigeonhole_violation() {
        let mut manager = CardinalityConstraintManager::new();

        let constraint = PigeonholeConstraint {
            num_pigeons: 11,
            num_holes: 10,
            array: 100,
            indices: vec![],
        };

        let result = manager.add_pigeonhole(constraint);
        assert!(result.is_err());
    }

    #[test]
    fn test_pigeonhole_satisfiable() {
        let mut manager = CardinalityConstraintManager::new();

        let constraint = PigeonholeConstraint {
            num_pigeons: 10,
            num_holes: 10,
            array: 100,
            indices: vec![],
        };

        let result = manager.add_pigeonhole(constraint);
        assert!(result.is_ok());
    }

    #[test]
    fn test_infer_bounds() {
        let mut manager = CardinalityConstraintManager::new();

        let operations = vec![
            ArrayOperation::Store {
                array: 100,
                indices: vec![1],
                value: 42,
            },
            ArrayOperation::Store {
                array: 100,
                indices: vec![2],
                value: 43,
            },
        ];

        manager.infer_bounds(100, &operations);

        let constraint = manager.get_constraint(100).unwrap();
        assert!(constraint.modified_locations.is_some());
    }

    #[test]
    fn test_cardinality_reasoner() {
        let mut reasoner = CardinalityReasoner::new();

        let constraint = ArrayCardinalityConstraint::new(100)
            .with_domain_cardinality(CardinalityBound::Exact(10))
            .with_range_cardinality(CardinalityBound::AtMost(10));

        reasoner.add_constraint(constraint);

        let result = reasoner.propagate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cardinality_conflict() {
        let mut reasoner = CardinalityReasoner::new();

        // Range larger than domain should cause conflict
        let constraint = ArrayCardinalityConstraint::new(100)
            .with_domain_cardinality(CardinalityBound::Exact(10))
            .with_range_cardinality(CardinalityBound::AtLeast(15));

        reasoner.add_constraint(constraint);

        let result = reasoner.propagate();
        assert!(result.is_err());
    }

    #[test]
    fn test_finite_model_constraints() {
        let mut manager = CardinalityConstraintManager::new();

        let constraint = ArrayCardinalityConstraint::new(100)
            .with_domain_cardinality(CardinalityBound::Exact(5))
            .with_range_cardinality(CardinalityBound::AtMost(3));

        manager.add_constraint(constraint);

        let fm_constraints = manager.generate_finite_model_constraints(100);
        assert_eq!(fm_constraints.len(), 2);
    }

    #[test]
    fn test_bound_display() {
        assert_eq!(format!("{}", CardinalityBound::Exact(5)), "= 5");
        assert_eq!(format!("{}", CardinalityBound::AtMost(10)), "≤ 10");
        assert_eq!(format!("{}", CardinalityBound::AtLeast(3)), "≥ 3");
        assert_eq!(
            format!("{}", CardinalityBound::Range { min: 1, max: 10 }),
            "[1, 10]"
        );
        assert_eq!(format!("{}", CardinalityBound::Infinite), "∞");
    }

    #[test]
    fn test_symmetry_patterns() {
        let patterns = [
            SymmetryPattern::Lexicographic,
            SymmetryPattern::ValueOrdering,
            SymmetryPattern::IndexOrdering,
        ];
        assert_eq!(patterns.len(), 3);
    }

    #[test]
    fn test_bound_intersection_range() {
        let b1 = CardinalityBound::Range { min: 5, max: 15 };
        let b2 = CardinalityBound::Range { min: 10, max: 20 };
        let result = b1.intersect(&b2);
        assert_eq!(result, Some(CardinalityBound::Range { min: 10, max: 15 }));
    }

    #[test]
    fn test_bound_intersection_incompatible_range() {
        let b1 = CardinalityBound::Range { min: 5, max: 10 };
        let b2 = CardinalityBound::Range { min: 15, max: 20 };
        let result = b1.intersect(&b2);
        assert_eq!(result, None);
    }
}
