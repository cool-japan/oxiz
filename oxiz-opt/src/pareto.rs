//! Multi-objective Pareto optimization.
//!
//! This module implements Pareto-optimal solution enumeration for
//! multi-objective optimization problems. Given multiple objectives,
//! it finds the Pareto front - the set of solutions where no objective
//! can be improved without worsening another.
//!
//! Reference: Z3's `opt/opt_pareto.cpp`

use crate::maxsat::Weight;
use crate::objective::{Objective, ObjectiveKind};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use smallvec::SmallVec;
use std::cmp::Ordering;
use thiserror::Error;

/// Errors that can occur during Pareto optimization
#[derive(Error, Debug)]
pub enum ParetoError {
    /// No solution exists
    #[error("unsatisfiable")]
    Unsatisfiable,
    /// Resource limit exceeded
    #[error("resource limit")]
    ResourceLimit,
    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result of Pareto optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParetoResult {
    /// Found all Pareto-optimal solutions
    Complete,
    /// Found some solutions but not proven complete
    Partial,
    /// No solution exists
    Unsatisfiable,
    /// Could not determine
    Unknown,
}

/// Configuration for Pareto optimization
#[derive(Debug, Clone)]
pub struct ParetoConfig {
    /// Maximum number of Pareto-optimal solutions to find
    pub max_solutions: u32,
    /// Maximum iterations
    pub max_iterations: u32,
    /// Use box decomposition
    pub box_decomposition: bool,
}

impl Default for ParetoConfig {
    fn default() -> Self {
        Self {
            max_solutions: 1000,
            max_iterations: 100000,
            box_decomposition: true,
        }
    }
}

/// Statistics from Pareto optimization
#[derive(Debug, Clone, Default)]
pub struct ParetoStats {
    /// Number of solver calls
    pub solver_calls: u32,
    /// Number of solutions found
    pub solutions_found: u32,
    /// Number of dominated solutions pruned
    pub dominated_pruned: u32,
    /// Number of boxes explored
    pub boxes_explored: u32,
}

/// A point in objective space
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectivePoint {
    /// Values for each objective
    pub values: SmallVec<[Weight; 4]>,
}

impl ObjectivePoint {
    /// Create a new objective point
    pub fn new(values: impl IntoIterator<Item = Weight>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    /// Get the number of objectives
    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    /// Get the value for a specific objective
    pub fn get(&self, index: usize) -> Option<&Weight> {
        self.values.get(index)
    }

    /// Check if this point dominates another
    ///
    /// A point dominates another if it is at least as good in all objectives
    /// and strictly better in at least one.
    pub fn dominates(&self, other: &ObjectivePoint, kinds: &[ObjectiveKind]) -> bool {
        if self.values.len() != other.values.len() || self.values.len() != kinds.len() {
            return false;
        }

        let mut dominated_in_all = true;
        let mut strictly_better_in_one = false;

        for (i, &kind) in kinds.iter().enumerate().take(self.values.len()) {
            let cmp = self.compare_objective(i, other, kind);
            match cmp {
                Ordering::Less => {
                    // self is worse in this objective
                    dominated_in_all = false;
                    break;
                }
                Ordering::Greater => {
                    // self is strictly better
                    strictly_better_in_one = true;
                }
                Ordering::Equal => {
                    // Equal, continues to dominate
                }
            }
        }

        dominated_in_all && strictly_better_in_one
    }

    /// Compare a single objective value
    fn compare_objective(
        &self,
        index: usize,
        other: &ObjectivePoint,
        kind: ObjectiveKind,
    ) -> Ordering {
        let a = &self.values[index];
        let b = &other.values[index];

        match kind {
            ObjectiveKind::Minimize => {
                // For minimize, smaller is better
                match a.cmp(b) {
                    Ordering::Less => Ordering::Greater, // a < b means a is better
                    Ordering::Greater => Ordering::Less, // a > b means a is worse
                    Ordering::Equal => Ordering::Equal,
                }
            }
            ObjectiveKind::Maximize => {
                // For maximize, larger is better
                a.cmp(b)
            }
        }
    }

    /// Check if this point is dominated by another
    pub fn is_dominated_by(&self, other: &ObjectivePoint, kinds: &[ObjectiveKind]) -> bool {
        other.dominates(self, kinds)
    }
}

/// A box in objective space for box decomposition
#[derive(Debug, Clone)]
pub struct ObjectiveBox {
    /// Lower bounds for each objective
    pub lower: SmallVec<[Option<Weight>; 4]>,
    /// Upper bounds for each objective
    pub upper: SmallVec<[Option<Weight>; 4]>,
}

impl ObjectiveBox {
    /// Create a new unbounded box
    pub fn unbounded(dimension: usize) -> Self {
        Self {
            lower: (0..dimension).map(|_| None).collect(),
            upper: (0..dimension).map(|_| None).collect(),
        }
    }

    /// Create a box with given bounds
    pub fn new(
        lower: impl IntoIterator<Item = Option<Weight>>,
        upper: impl IntoIterator<Item = Option<Weight>>,
    ) -> Self {
        Self {
            lower: lower.into_iter().collect(),
            upper: upper.into_iter().collect(),
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.lower.len()
    }

    /// Check if a point is inside this box
    pub fn contains(&self, point: &ObjectivePoint) -> bool {
        if point.values.len() != self.lower.len() {
            return false;
        }

        for i in 0..self.lower.len() {
            if let Some(ref lo) = self.lower[i]
                && &point.values[i] < lo
            {
                return false;
            }
            if let Some(ref hi) = self.upper[i]
                && &point.values[i] > hi
            {
                return false;
            }
        }
        true
    }

    /// Check if this box is empty (lower > upper for some dimension)
    pub fn is_empty(&self) -> bool {
        for i in 0..self.lower.len() {
            if let (Some(lo), Some(hi)) = (&self.lower[i], &self.upper[i])
                && lo > hi
            {
                return true;
            }
        }
        false
    }

    /// Split this box based on a Pareto-optimal point
    ///
    /// Returns a list of sub-boxes that exclude the dominated region.
    pub fn split_at(&self, point: &ObjectivePoint, kinds: &[ObjectiveKind]) -> Vec<ObjectiveBox> {
        let dim = self.dimension();
        let mut result = Vec::new();

        for (i, &kind) in kinds.iter().enumerate().take(dim) {
            // Create a box where objective i is strictly better than the point
            let mut new_box = self.clone();

            match kind {
                ObjectiveKind::Minimize => {
                    // For minimize, strictly better means strictly less
                    // Set upper bound to point value - epsilon
                    // For integers, this is point - 1
                    match &point.values[i] {
                        Weight::Int(n) => {
                            new_box.upper[i] = Some(Weight::Int(n - BigInt::one()));
                        }
                        Weight::Rational(r) => {
                            // Use a small epsilon
                            let epsilon = BigRational::new(BigInt::one(), BigInt::from(1000000));
                            new_box.upper[i] = Some(Weight::Rational(r - epsilon));
                        }
                        Weight::Infinite => {
                            // Cannot improve on infinity
                            continue;
                        }
                    }
                }
                ObjectiveKind::Maximize => {
                    // For maximize, strictly better means strictly greater
                    match &point.values[i] {
                        Weight::Int(n) => {
                            new_box.lower[i] = Some(Weight::Int(n + BigInt::one()));
                        }
                        Weight::Rational(r) => {
                            let epsilon = BigRational::new(BigInt::one(), BigInt::from(1000000));
                            new_box.lower[i] = Some(Weight::Rational(r + epsilon));
                        }
                        Weight::Infinite => {
                            continue;
                        }
                    }
                }
            }

            if !new_box.is_empty() {
                result.push(new_box);
            }
        }

        result
    }
}

/// A Pareto-optimal solution
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    /// The objective values
    pub point: ObjectivePoint,
    /// Model (variable assignments) - would contain actual model data
    pub model_id: u32,
}

impl ParetoSolution {
    /// Create a new Pareto solution
    pub fn new(point: ObjectivePoint, model_id: u32) -> Self {
        Self { point, model_id }
    }
}

/// Pareto front (set of non-dominated solutions)
#[derive(Debug, Default)]
pub struct ParetoFront {
    /// The Pareto-optimal solutions
    solutions: Vec<ParetoSolution>,
    /// Objective kinds for dominance checking
    kinds: Vec<ObjectiveKind>,
}

impl ParetoFront {
    /// Create a new empty Pareto front
    pub fn new(kinds: impl IntoIterator<Item = ObjectiveKind>) -> Self {
        Self {
            solutions: Vec::new(),
            kinds: kinds.into_iter().collect(),
        }
    }

    /// Try to add a solution to the Pareto front
    ///
    /// Returns true if the solution was added (not dominated).
    /// Also removes any solutions that become dominated.
    pub fn add(&mut self, solution: ParetoSolution) -> bool {
        // Check if new solution is dominated by any existing
        for existing in &self.solutions {
            if solution.point.is_dominated_by(&existing.point, &self.kinds) {
                return false;
            }
        }

        // Remove solutions dominated by the new one
        self.solutions
            .retain(|s| !s.point.is_dominated_by(&solution.point, &self.kinds));

        // Add the new solution
        self.solutions.push(solution);
        true
    }

    /// Get all solutions
    pub fn solutions(&self) -> &[ParetoSolution] {
        &self.solutions
    }

    /// Get the number of solutions
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Clear all solutions
    pub fn clear(&mut self) {
        self.solutions.clear();
    }

    /// Get the ideal point (best value for each objective)
    pub fn ideal_point(&self) -> Option<ObjectivePoint> {
        if self.solutions.is_empty() {
            return None;
        }

        let dim = self.kinds.len();
        let mut values: SmallVec<[Weight; 4]> = SmallVec::with_capacity(dim);

        for i in 0..dim {
            let best =
                self.solutions
                    .iter()
                    .map(|s| &s.point.values[i])
                    .min_by(|a, b| match self.kinds[i] {
                        ObjectiveKind::Minimize => a.cmp(b),
                        ObjectiveKind::Maximize => b.cmp(a),
                    })?;
            values.push(best.clone());
        }

        Some(ObjectivePoint { values })
    }

    /// Get the nadir point (worst value for each objective among Pareto solutions)
    pub fn nadir_point(&self) -> Option<ObjectivePoint> {
        if self.solutions.is_empty() {
            return None;
        }

        let dim = self.kinds.len();
        let mut values: SmallVec<[Weight; 4]> = SmallVec::with_capacity(dim);

        for i in 0..dim {
            let worst =
                self.solutions
                    .iter()
                    .map(|s| &s.point.values[i])
                    .max_by(|a, b| match self.kinds[i] {
                        ObjectiveKind::Minimize => a.cmp(b),
                        ObjectiveKind::Maximize => b.cmp(a),
                    })?;
            values.push(worst.clone());
        }

        Some(ObjectivePoint { values })
    }
}

/// Pareto optimization solver
#[derive(Debug)]
pub struct ParetoSolver {
    /// Objectives to optimize
    objectives: Vec<Objective>,
    /// Configuration
    config: ParetoConfig,
    /// Statistics
    stats: ParetoStats,
    /// Current Pareto front
    front: ParetoFront,
    /// Boxes to explore (for box decomposition)
    boxes: Vec<ObjectiveBox>,
    /// Next model ID
    next_model_id: u32,
}

impl ParetoSolver {
    /// Create a new Pareto solver
    pub fn new() -> Self {
        Self::with_config(ParetoConfig::default())
    }

    /// Create a new Pareto solver with configuration
    pub fn with_config(config: ParetoConfig) -> Self {
        Self {
            objectives: Vec::new(),
            config,
            stats: ParetoStats::default(),
            front: ParetoFront::default(),
            boxes: Vec::new(),
            next_model_id: 0,
        }
    }

    /// Add an objective
    pub fn add_objective(&mut self, objective: Objective) {
        self.objectives.push(objective);
    }

    /// Initialize the solver
    pub fn initialize(&mut self) {
        // Set up the Pareto front with objective kinds
        let kinds: Vec<ObjectiveKind> = self.objectives.iter().map(|o| o.kind).collect();
        self.front = ParetoFront::new(kinds);

        // Initialize with one unbounded box
        if self.config.box_decomposition {
            self.boxes.clear();
            self.boxes
                .push(ObjectiveBox::unbounded(self.objectives.len()));
        }
    }

    /// Get the current Pareto front
    pub fn front(&self) -> &ParetoFront {
        &self.front
    }

    /// Get statistics
    pub fn stats(&self) -> &ParetoStats {
        &self.stats
    }

    /// Record a solution
    ///
    /// Returns true if the solution was Pareto-optimal.
    pub fn record_solution(&mut self, values: impl IntoIterator<Item = Weight>) -> bool {
        let point = ObjectivePoint::new(values);
        let model_id = self.next_model_id;
        self.next_model_id += 1;

        let solution = ParetoSolution::new(point.clone(), model_id);
        let added = self.front.add(solution);

        if added {
            self.stats.solutions_found += 1;

            // Update boxes if using box decomposition
            if self.config.box_decomposition {
                self.update_boxes(&point);
            }
        } else {
            self.stats.dominated_pruned += 1;
        }

        added
    }

    /// Update boxes after finding a Pareto-optimal solution
    fn update_boxes(&mut self, point: &ObjectivePoint) {
        let kinds: Vec<ObjectiveKind> = self.objectives.iter().map(|o| o.kind).collect();

        let mut new_boxes = Vec::new();

        for old_box in self.boxes.drain(..) {
            if old_box.contains(point) {
                // Split this box
                let sub_boxes = old_box.split_at(point, &kinds);
                new_boxes.extend(sub_boxes);
            } else {
                // Keep the box
                new_boxes.push(old_box);
            }
        }

        self.boxes = new_boxes;
        self.stats.boxes_explored += 1;
    }

    /// Check if there are more boxes to explore
    pub fn has_unexplored_boxes(&self) -> bool {
        !self.boxes.is_empty()
    }

    /// Get the next box to explore
    pub fn next_box(&mut self) -> Option<ObjectiveBox> {
        self.boxes.pop()
    }

    /// Check if we've reached the solution limit
    pub fn at_solution_limit(&self) -> bool {
        self.stats.solutions_found >= self.config.max_solutions
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.front.clear();
        self.boxes.clear();
        self.stats = ParetoStats::default();
        self.next_model_id = 0;
    }
}

impl Default for ParetoSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObjectiveId;
    use oxiz_core::ast::TermId;

    #[test]
    fn test_objective_point_creation() {
        let point = ObjectivePoint::new([Weight::from(1), Weight::from(2), Weight::from(3)]);
        assert_eq!(point.dimension(), 3);
        assert_eq!(point.get(0), Some(&Weight::from(1)));
        assert_eq!(point.get(1), Some(&Weight::from(2)));
        assert_eq!(point.get(2), Some(&Weight::from(3)));
    }

    #[test]
    fn test_dominance_minimize() {
        let kinds = vec![ObjectiveKind::Minimize, ObjectiveKind::Minimize];

        let a = ObjectivePoint::new([Weight::from(1), Weight::from(2)]);
        let b = ObjectivePoint::new([Weight::from(2), Weight::from(3)]);
        let c = ObjectivePoint::new([Weight::from(1), Weight::from(3)]);
        // d and e are incomparable: d is better in dim 0, e is better in dim 1
        let d = ObjectivePoint::new([Weight::from(1), Weight::from(4)]);
        let e = ObjectivePoint::new([Weight::from(2), Weight::from(2)]);

        // a dominates b (1 < 2 and 2 < 3)
        assert!(a.dominates(&b, &kinds));
        assert!(!b.dominates(&a, &kinds));

        // a dominates c (1 = 1 and 2 < 3)
        assert!(a.dominates(&c, &kinds));

        // c dominates b (1 < 2 and 3 = 3)
        assert!(c.dominates(&b, &kinds));

        // d and e are incomparable: d[0]=1 < e[0]=2, but d[1]=4 > e[1]=2
        assert!(!d.dominates(&e, &kinds));
        assert!(!e.dominates(&d, &kinds));
    }

    #[test]
    fn test_dominance_maximize() {
        let kinds = vec![ObjectiveKind::Maximize, ObjectiveKind::Maximize];

        let a = ObjectivePoint::new([Weight::from(3), Weight::from(4)]);
        let b = ObjectivePoint::new([Weight::from(2), Weight::from(3)]);

        // a dominates b (3 > 2 and 4 > 3)
        assert!(a.dominates(&b, &kinds));
        assert!(!b.dominates(&a, &kinds));
    }

    #[test]
    fn test_dominance_mixed() {
        let kinds = vec![ObjectiveKind::Minimize, ObjectiveKind::Maximize];

        let a = ObjectivePoint::new([Weight::from(1), Weight::from(5)]); // low cost, high value
        let b = ObjectivePoint::new([Weight::from(2), Weight::from(4)]); // higher cost, lower value

        // a dominates b
        assert!(a.dominates(&b, &kinds));
        assert!(!b.dominates(&a, &kinds));
    }

    #[test]
    fn test_objective_box_contains() {
        let bounds = ObjectiveBox::new(
            [Some(Weight::from(0)), Some(Weight::from(0))],
            [Some(Weight::from(10)), Some(Weight::from(10))],
        );

        let inside = ObjectivePoint::new([Weight::from(5), Weight::from(5)]);
        let outside = ObjectivePoint::new([Weight::from(15), Weight::from(5)]);

        assert!(bounds.contains(&inside));
        assert!(!bounds.contains(&outside));
    }

    #[test]
    fn test_objective_box_is_empty() {
        let empty = ObjectiveBox::new(
            [Some(Weight::from(10)), Some(Weight::from(0))],
            [Some(Weight::from(5)), Some(Weight::from(10))], // 10 > 5 in first dim
        );

        let non_empty = ObjectiveBox::new(
            [Some(Weight::from(0)), Some(Weight::from(0))],
            [Some(Weight::from(10)), Some(Weight::from(10))],
        );

        assert!(empty.is_empty());
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_pareto_front_add() {
        let kinds = vec![ObjectiveKind::Minimize, ObjectiveKind::Minimize];
        let mut front = ParetoFront::new(kinds.clone());

        let p1 = ObjectivePoint::new([Weight::from(1), Weight::from(5)]);
        let p2 = ObjectivePoint::new([Weight::from(3), Weight::from(3)]);
        let p3 = ObjectivePoint::new([Weight::from(5), Weight::from(1)]);
        let dominated = ObjectivePoint::new([Weight::from(4), Weight::from(4)]); // dominated by p2

        assert!(front.add(ParetoSolution::new(p1.clone(), 0)));
        assert!(front.add(ParetoSolution::new(p2.clone(), 1)));
        assert!(front.add(ParetoSolution::new(p3.clone(), 2)));
        assert!(!front.add(ParetoSolution::new(dominated, 3))); // Should be rejected

        assert_eq!(front.len(), 3);
    }

    #[test]
    fn test_pareto_front_removes_dominated() {
        let kinds = vec![ObjectiveKind::Minimize, ObjectiveKind::Minimize];
        let mut front = ParetoFront::new(kinds.clone());

        // Add a point that will be dominated
        let dominated = ObjectivePoint::new([Weight::from(5), Weight::from(5)]);
        assert!(front.add(ParetoSolution::new(dominated, 0)));
        assert_eq!(front.len(), 1);

        // Add a dominating point
        let dominator = ObjectivePoint::new([Weight::from(3), Weight::from(3)]);
        assert!(front.add(ParetoSolution::new(dominator, 1)));

        // The dominated point should be removed
        assert_eq!(front.len(), 1);
    }

    #[test]
    fn test_pareto_solver_basic() {
        let mut solver = ParetoSolver::new();

        solver.add_objective(Objective {
            id: ObjectiveId(0),
            term: TermId::from(1),
            kind: ObjectiveKind::Minimize,
            priority: 0,
        });

        solver.add_objective(Objective {
            id: ObjectiveId(1),
            term: TermId::from(2),
            kind: ObjectiveKind::Minimize,
            priority: 0,
        });

        solver.initialize();

        // Record some solutions
        assert!(solver.record_solution([Weight::from(1), Weight::from(5)]));
        assert!(solver.record_solution([Weight::from(3), Weight::from(3)]));
        assert!(solver.record_solution([Weight::from(5), Weight::from(1)]));

        // This one is dominated
        assert!(!solver.record_solution([Weight::from(4), Weight::from(4)]));

        assert_eq!(solver.front().len(), 3);
        assert_eq!(solver.stats().solutions_found, 3);
        assert_eq!(solver.stats().dominated_pruned, 1);
    }

    #[test]
    fn test_ideal_nadir_points() {
        let kinds = vec![ObjectiveKind::Minimize, ObjectiveKind::Minimize];
        let mut front = ParetoFront::new(kinds);

        front.add(ParetoSolution::new(
            ObjectivePoint::new([Weight::from(1), Weight::from(5)]),
            0,
        ));
        front.add(ParetoSolution::new(
            ObjectivePoint::new([Weight::from(3), Weight::from(3)]),
            1,
        ));
        front.add(ParetoSolution::new(
            ObjectivePoint::new([Weight::from(5), Weight::from(1)]),
            2,
        ));

        let ideal = front.ideal_point().unwrap();
        assert_eq!(ideal.values[0], Weight::from(1)); // Best first objective
        assert_eq!(ideal.values[1], Weight::from(1)); // Best second objective

        let nadir = front.nadir_point().unwrap();
        assert_eq!(nadir.values[0], Weight::from(5)); // Worst first objective
        assert_eq!(nadir.values[1], Weight::from(5)); // Worst second objective
    }
}
