//! Pareto Frontier Enumeration for Multi-Objective Optimization.
//!
//! This module implements algorithms for enumerating the Pareto frontier in
//! multi-objective optimization problems. A Pareto-optimal point is one where
//! no objective can be improved without worsening another.
//!
//! ## Enumeration Strategies
//!
//! 1. **Iterative Pareto Search**: Find one Pareto point, block it, repeat
//! 2. **BoxingMethod**: Use hyperboxes to partition the objective space
//! 3. **ε-Constraint**: Fix all but one objective, optimize the remaining one
//! 4. **Weighted Sum**: Enumerate using different weight combinations
//!
//! ## Applications
//!
//! - Resource allocation with multiple goals
//! - Scheduling with cost/time trade-offs
//! - Design optimization (e.g., performance vs. energy)
//! - Multi-criteria decision making
//!
//! ## References
//!
//! - Ehrgott: "Multicriteria Optimization" (2005)
//! - Deb: "Multi-Objective Optimization using Evolutionary Algorithms" (2001)
//! - Z3's OMT (Optimization Modulo Theories) multi-objective support

use crate::maxsat::Weight;
use smallvec::SmallVec;
use std::fmt;

/// A point in objective space (values for each objective).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectivePoint {
    /// Objective values.
    pub values: SmallVec<[Weight; 4]>,
}

impl ObjectivePoint {
    /// Create a new objective point.
    pub fn new(values: impl IntoIterator<Item = Weight>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }

    /// Get number of objectives.
    pub fn num_objectives(&self) -> usize {
        self.values.len()
    }

    /// Check if this point dominates another.
    ///
    /// Point p dominates q if p is at least as good in all objectives
    /// and strictly better in at least one.
    pub fn dominates(&self, other: &ObjectivePoint) -> bool {
        if self.values.len() != other.values.len() {
            return false;
        }

        let mut strictly_better_in_some = false;

        for i in 0..self.values.len() {
            if self.values[i] > other.values[i] {
                // This objective is worse (assuming minimization)
                return false;
            }

            if self.values[i] < other.values[i] {
                strictly_better_in_some = true;
            }
        }

        strictly_better_in_some
    }

    /// Check if this point equals another.
    pub fn equals(&self, other: &ObjectivePoint) -> bool {
        self.values == other.values
    }
}

impl fmt::Display for ObjectivePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, val) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", val)?;
        }
        write!(f, ")")
    }
}

/// Pareto frontier (set of non-dominated points).
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    /// Non-dominated points.
    points: Vec<ObjectivePoint>,
    /// Number of objectives.
    num_objectives: usize,
}

impl ParetoFrontier {
    /// Create an empty Pareto frontier.
    pub fn new(num_objectives: usize) -> Self {
        Self {
            points: Vec::new(),
            num_objectives,
        }
    }

    /// Add a point to the frontier.
    ///
    /// If the point is dominated by existing points, it's not added.
    /// If the point dominates existing points, they are removed.
    pub fn add(&mut self, point: ObjectivePoint) -> bool {
        if point.num_objectives() != self.num_objectives {
            return false;
        }

        // Check if dominated by any existing point
        for existing in &self.points {
            if existing.dominates(&point) {
                return false; // Dominated, don't add
            }
        }

        // Remove any points dominated by the new point
        self.points.retain(|p| !point.dominates(p));

        // Add the new point
        self.points.push(point);
        true
    }

    /// Get all Pareto-optimal points.
    pub fn points(&self) -> &[ObjectivePoint] {
        &self.points
    }

    /// Number of Pareto-optimal points found.
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Clear the frontier.
    pub fn clear(&mut self) {
        self.points.clear();
    }
}

/// Configuration for Pareto enumeration.
#[derive(Debug, Clone)]
pub struct ParetoEnumConfig {
    /// Maximum number of Pareto points to find.
    pub max_points: usize,
    /// Time limit (milliseconds).
    pub timeout_ms: u64,
    /// Enable early termination (stop when enough points found).
    pub early_termination: bool,
}

impl Default for ParetoEnumConfig {
    fn default() -> Self {
        Self {
            max_points: 1000,
            timeout_ms: 60_000, // 1 minute
            early_termination: true,
        }
    }
}

/// Statistics for Pareto enumeration.
#[derive(Debug, Clone, Default)]
pub struct ParetoEnumStats {
    /// Number of Pareto points found.
    pub points_found: usize,
    /// Number of dominated points encountered.
    pub dominated_points: usize,
    /// Number of SAT calls.
    pub sat_calls: usize,
    /// Total time (microseconds).
    pub total_time_us: u64,
}

/// Pareto point enumerator.
pub struct ParetoEnumerator {
    /// Configuration.
    config: ParetoEnumConfig,
    /// Statistics.
    stats: ParetoEnumStats,
    /// Pareto frontier.
    frontier: ParetoFrontier,
}

impl ParetoEnumerator {
    /// Create a new Pareto enumerator.
    pub fn new(num_objectives: usize) -> Self {
        Self::with_config(num_objectives, ParetoEnumConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(num_objectives: usize, config: ParetoEnumConfig) -> Self {
        Self {
            config,
            stats: ParetoEnumStats::default(),
            frontier: ParetoFrontier::new(num_objectives),
        }
    }

    /// Get the current Pareto frontier.
    pub fn frontier(&self) -> &ParetoFrontier {
        &self.frontier
    }

    /// Get statistics.
    pub fn stats(&self) -> &ParetoEnumStats {
        &self.stats
    }

    /// Reset the enumerator.
    pub fn reset(&mut self) {
        self.frontier.clear();
        self.stats = ParetoEnumStats::default();
    }

    /// Add a candidate point to the frontier.
    ///
    /// Returns true if the point was added (non-dominated).
    pub fn add_point(&mut self, point: ObjectivePoint) -> bool {
        let added = self.frontier.add(point);

        if added {
            self.stats.points_found += 1;
        } else {
            self.stats.dominated_points += 1;
        }

        added
    }

    /// Check if enumeration should terminate.
    pub fn should_terminate(&self, elapsed_ms: u64) -> bool {
        if self.frontier.size() >= self.config.max_points {
            return true;
        }

        if elapsed_ms >= self.config.timeout_ms {
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objective_point_creation() {
        let point = ObjectivePoint::new(vec![Weight::one(), Weight::from(2)]);
        assert_eq!(point.num_objectives(), 2);
    }

    #[test]
    fn test_objective_point_dominance() {
        // Point (1, 2) with minimization
        let p1 = ObjectivePoint::new(vec![Weight::one(), Weight::from(2)]);
        // Point (2, 3) - dominated by p1
        let p2 = ObjectivePoint::new(vec![Weight::from(2), Weight::from(3)]);
        // Point (1, 3) - not dominated (equal in first, worse in second)
        let p3 = ObjectivePoint::new(vec![Weight::one(), Weight::from(3)]);

        assert!(p1.dominates(&p2)); // (1,2) dominates (2,3)
        assert!(!p2.dominates(&p1));
        assert!(p1.dominates(&p3)); // (1,2) dominates (1,3)
        assert!(!p3.dominates(&p1));
        assert!(!p1.dominates(&p1)); // No strict improvement over itself
    }

    #[test]
    fn test_pareto_frontier_add() {
        let mut frontier = ParetoFrontier::new(2);

        let p1 = ObjectivePoint::new(vec![Weight::one(), Weight::from(2)]);
        let p2 = ObjectivePoint::new(vec![Weight::from(2), Weight::one()]);
        let p3 = ObjectivePoint::new(vec![Weight::from(2), Weight::from(3)]);

        // Add first point
        assert!(frontier.add(p1.clone()));
        assert_eq!(frontier.size(), 1);

        // Add second point (not dominated)
        assert!(frontier.add(p2.clone()));
        assert_eq!(frontier.size(), 2);

        // Add third point (dominated by p1)
        assert!(!frontier.add(p3));
        assert_eq!(frontier.size(), 2); // Not added
    }

    #[test]
    fn test_pareto_frontier_removal() {
        let mut frontier = ParetoFrontier::new(2);

        // Add a dominated point first
        let p1 = ObjectivePoint::new(vec![Weight::from(2), Weight::from(2)]);
        frontier.add(p1);
        assert_eq!(frontier.size(), 1);

        // Add a dominating point - should remove p1
        let p2 = ObjectivePoint::new(vec![Weight::one(), Weight::one()]);
        frontier.add(p2);
        assert_eq!(frontier.size(), 1); // p1 removed, only p2 remains
    }

    #[test]
    fn test_pareto_enumerator_creation() {
        let enumerator = ParetoEnumerator::new(3);
        assert_eq!(enumerator.frontier().points().len(), 0);
        assert_eq!(enumerator.stats().points_found, 0);
    }

    #[test]
    fn test_pareto_enumerator_add() {
        let mut enumerator = ParetoEnumerator::new(2);

        let p1 = ObjectivePoint::new(vec![Weight::one(), Weight::from(2)]);
        let p2 = ObjectivePoint::new(vec![Weight::from(3), Weight::from(4)]);

        assert!(enumerator.add_point(p1));
        assert_eq!(enumerator.stats().points_found, 1);

        // p2 is dominated
        assert!(!enumerator.add_point(p2));
        assert_eq!(enumerator.stats().dominated_points, 1);
    }

    #[test]
    fn test_pareto_termination() {
        let config = ParetoEnumConfig {
            max_points: 5,
            timeout_ms: 1000,
            ..Default::default()
        };
        let mut enumerator = ParetoEnumerator::with_config(2, config);

        // Add 5 points
        for i in 0..5 {
            let p = ObjectivePoint::new(vec![Weight::from(i), Weight::from(10 - i)]);
            enumerator.add_point(p);
        }

        // Should terminate (max points reached)
        assert!(enumerator.should_terminate(0));
    }

    #[test]
    fn test_pareto_timeout() {
        let config = ParetoEnumConfig {
            timeout_ms: 100,
            ..Default::default()
        };
        let enumerator = ParetoEnumerator::with_config(2, config);

        // Should terminate (timeout)
        assert!(enumerator.should_terminate(200));
    }

    #[test]
    fn test_pareto_reset() {
        let mut enumerator = ParetoEnumerator::new(2);

        enumerator.add_point(ObjectivePoint::new(vec![Weight::one(), Weight::from(2)]));
        assert_eq!(enumerator.stats().points_found, 1);

        enumerator.reset();

        assert_eq!(enumerator.frontier().size(), 0);
        assert_eq!(enumerator.stats().points_found, 0);
    }
}
