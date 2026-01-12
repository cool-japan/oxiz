//! Backbone detection and computation
//!
//! The backbone of a SAT formula consists of all literals that have the same
//! truth value in every satisfying assignment. Backbone detection can help
//! simplify formulas and guide search strategies.
//!
//! References:
//! - "Computing Backbones of Propositional Formulas" (Janota et al.)
//! - "Backbone Computation for Propositional Formulas" (Marques-Silva)
//! - "Efficient Backbone Computation" (Kullmann)

use crate::literal::Lit;
use std::collections::HashSet;

/// Statistics for backbone computation
#[derive(Debug, Clone, Default)]
pub struct BackboneStats {
    /// Number of backbone literals found
    pub backbone_size: usize,
    /// Number of positive backbone literals
    pub positive_lits: usize,
    /// Number of negative backbone literals
    pub negative_lits: usize,
    /// Number of SAT solver calls made
    pub solver_calls: usize,
    /// Number of iterations performed
    pub iterations: usize,
}

impl BackboneStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Backbone Computation Statistics:");
        println!("  Backbone size: {}", self.backbone_size);
        println!("  Positive literals: {}", self.positive_lits);
        println!("  Negative literals: {}", self.negative_lits);
        println!("  Solver calls: {}", self.solver_calls);
        println!("  Iterations: {}", self.iterations);
        if self.solver_calls > 0 {
            println!(
                "  Avg backbone per call: {:.2}",
                self.backbone_size as f64 / self.solver_calls as f64
            );
        }
    }
}

/// Backbone computation algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneAlgorithm {
    /// Iterative algorithm - test each literal individually
    Iterative,
    /// Binary search - partition literals and test groups
    BinarySearch,
    /// Assumption-based - use assumptions to test multiple literals
    AssumptionBased,
}

/// Backbone detector
#[derive(Debug)]
pub struct BackboneDetector {
    /// Detected backbone literals
    backbone: HashSet<Lit>,
    /// Algorithm to use
    algorithm: BackboneAlgorithm,
    /// Statistics
    stats: BackboneStats,
}

impl Default for BackboneDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl BackboneDetector {
    /// Create a new backbone detector with default algorithm
    #[must_use]
    pub fn new() -> Self {
        Self {
            backbone: HashSet::new(),
            algorithm: BackboneAlgorithm::Iterative,
            stats: BackboneStats::default(),
        }
    }

    /// Create with specific algorithm
    #[must_use]
    pub fn with_algorithm(algorithm: BackboneAlgorithm) -> Self {
        Self {
            backbone: HashSet::new(),
            algorithm,
            stats: BackboneStats::default(),
        }
    }

    /// Check if a literal is in the backbone
    #[must_use]
    pub fn is_backbone(&self, lit: Lit) -> bool {
        self.backbone.contains(&lit)
    }

    /// Add a literal to the backbone
    pub fn add_backbone(&mut self, lit: Lit) {
        if self.backbone.insert(lit) {
            self.stats.backbone_size += 1;
            if lit.is_pos() {
                self.stats.positive_lits += 1;
            } else {
                self.stats.negative_lits += 1;
            }
        }
    }

    /// Remove a literal from the backbone
    pub fn remove_backbone(&mut self, lit: Lit) {
        if self.backbone.remove(&lit) {
            self.stats.backbone_size = self.stats.backbone_size.saturating_sub(1);
            if lit.is_pos() {
                self.stats.positive_lits = self.stats.positive_lits.saturating_sub(1);
            } else {
                self.stats.negative_lits = self.stats.negative_lits.saturating_sub(1);
            }
        }
    }

    /// Detect backbone using iterative algorithm
    ///
    /// For each literal l, check if the formula is UNSAT under assumption ~l.
    /// If UNSAT, then l must be in the backbone.
    pub fn detect_iterative<F>(&mut self, candidates: &[Lit], mut is_sat_with_assumption: F)
    where
        F: FnMut(Lit) -> bool,
    {
        self.backbone.clear();
        self.stats = BackboneStats::default();

        for &lit in candidates {
            self.stats.iterations += 1;
            self.stats.solver_calls += 1;

            // Test if formula is SAT with ~lit
            if !is_sat_with_assumption(!lit) {
                // UNSAT with ~lit means lit must be true
                self.add_backbone(lit);
            }
        }
    }

    /// Detect backbone using binary search
    ///
    /// Recursively partition the candidate set and test groups.
    /// More efficient when the backbone is large.
    pub fn detect_binary_search<F>(&mut self, candidates: &[Lit], is_sat_with_assumptions: F)
    where
        F: Fn(&[Lit]) -> bool + Clone,
    {
        self.backbone.clear();
        self.stats = BackboneStats::default();

        let mut to_process = vec![candidates.to_vec()];

        while let Some(group) = to_process.pop() {
            if group.is_empty() {
                continue;
            }

            self.stats.iterations += 1;
            self.stats.solver_calls += 1;

            // Test if formula is SAT with all negations of this group
            let negated: Vec<_> = group.iter().map(|&lit| !lit).collect();

            if !is_sat_with_assumptions(&negated) {
                // UNSAT - all literals in group are backbone
                for &lit in &group {
                    self.add_backbone(lit);
                }
            } else if group.len() > 1 {
                // SAT - split group and continue
                let mid = group.len() / 2;
                to_process.push(group[..mid].to_vec());
                to_process.push(group[mid..].to_vec());
            }
            // If group has size 1 and is SAT, it's not backbone
        }
    }

    /// Compute rotatable literals (non-backbone)
    ///
    /// A literal is rotatable if it can be both true and false in different
    /// satisfying assignments.
    #[must_use]
    pub fn compute_rotatable(&self, all_lits: &[Lit]) -> Vec<Lit> {
        all_lits
            .iter()
            .filter(|&&lit| !self.is_backbone(lit) && !self.is_backbone(!lit))
            .copied()
            .collect()
    }

    /// Get all backbone literals
    #[must_use]
    pub fn backbone(&self) -> Vec<Lit> {
        self.backbone.iter().copied().collect()
    }

    /// Get backbone size
    #[must_use]
    pub fn size(&self) -> usize {
        self.backbone.len()
    }

    /// Check if backbone is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.backbone.is_empty()
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &BackboneStats {
        &self.stats
    }

    /// Reset the detector
    pub fn clear(&mut self) {
        self.backbone.clear();
        self.stats = BackboneStats::default();
    }

    /// Get the algorithm being used
    #[must_use]
    pub fn algorithm(&self) -> BackboneAlgorithm {
        self.algorithm
    }

    /// Set the algorithm to use
    pub fn set_algorithm(&mut self, algorithm: BackboneAlgorithm) {
        self.algorithm = algorithm;
    }
}

/// Backbone filter for literal selection
///
/// Use backbone information to improve branching decisions
#[derive(Debug)]
pub struct BackboneFilter {
    /// Backbone detector
    detector: BackboneDetector,
    /// Whether to prefer backbone literals
    prefer_backbone: bool,
}

impl Default for BackboneFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl BackboneFilter {
    /// Create a new backbone filter
    #[must_use]
    pub fn new() -> Self {
        Self {
            detector: BackboneDetector::new(),
            prefer_backbone: true,
        }
    }

    /// Filter literals based on backbone
    ///
    /// Prioritize or deprioritize backbone literals based on configuration
    #[must_use]
    pub fn filter(&self, candidates: &[Lit]) -> Vec<Lit> {
        let mut backbone = Vec::new();
        let mut non_backbone = Vec::new();

        for &lit in candidates {
            if self.detector.is_backbone(lit) {
                backbone.push(lit);
            } else {
                non_backbone.push(lit);
            }
        }

        if self.prefer_backbone {
            backbone.extend(non_backbone);
            backbone
        } else {
            non_backbone.extend(backbone);
            non_backbone
        }
    }

    /// Get the backbone detector
    #[must_use]
    pub fn detector(&self) -> &BackboneDetector {
        &self.detector
    }

    /// Get mutable backbone detector
    pub fn detector_mut(&mut self) -> &mut BackboneDetector {
        &mut self.detector
    }

    /// Set whether to prefer backbone literals
    pub fn set_prefer_backbone(&mut self, prefer: bool) {
        self.prefer_backbone = prefer;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_backbone_detector_creation() {
        let detector = BackboneDetector::new();
        assert_eq!(detector.size(), 0);
        assert!(detector.is_empty());
    }

    #[test]
    fn test_add_backbone() {
        let mut detector = BackboneDetector::new();
        let lit = Lit::pos(Var::new(0));

        assert!(!detector.is_backbone(lit));
        detector.add_backbone(lit);
        assert!(detector.is_backbone(lit));
        assert_eq!(detector.size(), 1);
    }

    #[test]
    fn test_remove_backbone() {
        let mut detector = BackboneDetector::new();
        let lit = Lit::pos(Var::new(0));

        detector.add_backbone(lit);
        assert_eq!(detector.size(), 1);

        detector.remove_backbone(lit);
        assert_eq!(detector.size(), 0);
        assert!(!detector.is_backbone(lit));
    }

    #[test]
    fn test_detect_iterative() {
        let mut detector = BackboneDetector::new();
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        // Simulate: a is backbone (formula UNSAT with ~a), b is not
        let is_sat = |lit: Lit| -> bool {
            lit != !a // UNSAT only when testing ~a
        };

        detector.detect_iterative(&[a, b], is_sat);

        assert!(detector.is_backbone(a));
        assert!(!detector.is_backbone(b));
        assert_eq!(detector.size(), 1);
    }

    #[test]
    fn test_compute_rotatable() {
        let mut detector = BackboneDetector::new();
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let c = Lit::pos(Var::new(2));

        detector.add_backbone(a);

        let rotatable = detector.compute_rotatable(&[a, b, c]);

        assert!(!rotatable.contains(&a));
        assert!(rotatable.contains(&b));
        assert!(rotatable.contains(&c));
    }

    #[test]
    fn test_backbone_filter() {
        let mut filter = BackboneFilter::new();
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let c = Lit::pos(Var::new(2));

        filter.detector_mut().add_backbone(a);

        let filtered = filter.filter(&[c, b, a]);

        // Should prioritize backbone (a) first
        assert_eq!(filtered[0], a);
    }

    #[test]
    fn test_statistics() {
        let mut detector = BackboneDetector::new();
        detector.add_backbone(Lit::pos(Var::new(0)));
        detector.add_backbone(Lit::neg(Var::new(1)));

        let stats = detector.stats();
        assert_eq!(stats.backbone_size, 2);
        assert_eq!(stats.positive_lits, 1);
        assert_eq!(stats.negative_lits, 1);
    }

    #[test]
    fn test_algorithm_selection() {
        let mut detector = BackboneDetector::with_algorithm(BackboneAlgorithm::BinarySearch);
        assert_eq!(detector.algorithm(), BackboneAlgorithm::BinarySearch);

        detector.set_algorithm(BackboneAlgorithm::Iterative);
        assert_eq!(detector.algorithm(), BackboneAlgorithm::Iterative);
    }

    #[test]
    fn test_clear() {
        let mut detector = BackboneDetector::new();
        detector.add_backbone(Lit::pos(Var::new(0)));
        assert!(!detector.is_empty());

        detector.clear();
        assert!(detector.is_empty());
        assert_eq!(detector.stats().backbone_size, 0);
    }
}
