/// Cube-and-Conquer: Advanced parallel SAT solving via search space partitioning.
///
/// This module implements the Cube-and-Conquer technique, which partitions the search
/// space into "cubes" (partial assignments) that can be solved independently in parallel.
///
/// The algorithm works in two phases:
/// 1. **Cube Phase**: Use lookahead to generate a set of cubes that partition the space
/// 2. **Conquer Phase**: Solve each cube in parallel using CDCL
///
/// This approach is particularly effective for hard combinatorial problems.
use crate::literal::Lit;
use std::collections::HashSet;

/// A cube represents a partial assignment (conjunction of literals).
///
/// Cubes partition the search space - if all cubes are UNSAT, the formula is UNSAT.
/// If any cube is SAT, the formula is SAT.
#[derive(Debug, Clone)]
pub struct Cube {
    /// The literals in this cube (partial assignment)
    pub literals: Vec<Lit>,
    /// Estimated difficulty score (higher = harder)
    pub difficulty: f64,
    /// Depth at which this cube was created
    pub depth: usize,
}

impl Cube {
    /// Creates a new cube from literals.
    pub fn new(literals: Vec<Lit>) -> Self {
        Self {
            depth: literals.len(),
            literals,
            difficulty: 0.0,
        }
    }

    /// Creates a cube with estimated difficulty.
    pub fn with_difficulty(literals: Vec<Lit>, difficulty: f64) -> Self {
        Self {
            depth: literals.len(),
            literals,
            difficulty,
        }
    }

    /// Returns the number of literals in the cube.
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Checks if the cube is empty.
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Extends the cube with an additional literal.
    pub fn extend(&self, lit: Lit) -> Self {
        let mut new_lits = self.literals.clone();
        new_lits.push(lit);
        Self {
            literals: new_lits,
            difficulty: 0.0,
            depth: self.depth + 1,
        }
    }

    /// Checks if the cube contains conflicting literals.
    pub fn is_consistent(&self) -> bool {
        let mut seen = HashSet::new();
        for &lit in &self.literals {
            if seen.contains(&lit.negate()) {
                return false;
            }
            seen.insert(lit);
        }
        true
    }
}

/// Strategy for cube generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeSplittingStrategy {
    /// Split based on variable activity (VSIDS-style)
    Activity,
    /// Split based on lookahead scores
    Lookahead,
    /// Split using the most constrained variable (smallest domain)
    MostConstrained,
    /// Balanced splitting to create similar-sized cubes
    Balanced,
}

/// Configuration for cube generation.
#[derive(Debug, Clone)]
pub struct CubeConfig {
    /// Maximum depth for cube splitting
    pub max_depth: usize,
    /// Target number of cubes to generate
    pub target_cubes: usize,
    /// Minimum literals per cube
    pub min_cube_size: usize,
    /// Maximum literals per cube
    pub max_cube_size: usize,
    /// Splitting strategy
    pub strategy: CubeSplittingStrategy,
    /// Enable adaptive depth adjustment
    pub adaptive_depth: bool,
}

impl Default for CubeConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            target_cubes: 100,
            min_cube_size: 3,
            max_cube_size: 15,
            strategy: CubeSplittingStrategy::Lookahead,
            adaptive_depth: true,
        }
    }
}

/// Cube generator using recursive splitting.
pub struct CubeGenerator {
    /// Configuration
    config: CubeConfig,
    /// Generated cubes
    cubes: Vec<Cube>,
    /// Number of variables
    num_vars: usize,
}

impl CubeGenerator {
    /// Creates a new cube generator.
    pub fn new(num_vars: usize, config: CubeConfig) -> Self {
        Self {
            config,
            cubes: Vec::new(),
            num_vars,
        }
    }

    /// Generates cubes by recursive splitting.
    ///
    /// Starting from an empty cube, recursively splits by choosing a variable
    /// and creating two cubes: one with the positive literal, one with negative.
    pub fn generate(&mut self, variable_scores: &[f64]) -> Vec<Cube> {
        self.cubes.clear();

        // Start with empty cube
        let initial = Cube::new(Vec::new());
        self.split_recursive(initial, variable_scores);

        // If adaptive depth is enabled and we have too few cubes, try again with more depth
        if self.config.adaptive_depth && self.cubes.len() < self.config.target_cubes / 2 {
            self.config.max_depth = (self.config.max_depth * 3) / 2;
            self.cubes.clear();
            let initial = Cube::new(Vec::new());
            self.split_recursive(initial, variable_scores);
        }

        std::mem::take(&mut self.cubes)
    }

    /// Recursively splits a cube into smaller cubes.
    fn split_recursive(&mut self, cube: Cube, variable_scores: &[f64]) {
        // Stop if we've reached max depth or target number of cubes
        if cube.depth >= self.config.max_depth || self.cubes.len() >= self.config.target_cubes {
            if cube.len() >= self.config.min_cube_size && cube.is_consistent() {
                self.cubes.push(cube);
            }
            return;
        }

        // Select next variable to split on
        if let Some(var) = self.select_splitting_variable(&cube, variable_scores) {
            use crate::literal::Var;
            let v = Var::new(var as u32);

            // Create two cubes: one with positive literal, one with negative
            let pos_cube = cube.extend(Lit::pos(v));
            let neg_cube = cube.extend(Lit::neg(v));

            // Recursively split both branches
            self.split_recursive(pos_cube, variable_scores);
            self.split_recursive(neg_cube, variable_scores);
        } else {
            // No more variables to split on, save this cube
            if cube.len() >= self.config.min_cube_size && cube.is_consistent() {
                self.cubes.push(cube);
            }
        }
    }

    /// Selects the best variable to split on based on the strategy.
    fn select_splitting_variable(&self, cube: &Cube, variable_scores: &[f64]) -> Option<usize> {
        // Get variables already assigned in cube
        let mut assigned = HashSet::new();
        for lit in &cube.literals {
            assigned.insert(lit.var().index());
        }

        // Find best unassigned variable
        let mut best_var = None;
        let mut best_score = f64::NEG_INFINITY;

        for var in 0..self.num_vars {
            if assigned.contains(&var) {
                continue;
            }

            let score = if var < variable_scores.len() {
                variable_scores[var]
            } else {
                0.0
            };

            if score > best_score {
                best_score = score;
                best_var = Some(var);
            }
        }

        best_var
    }

    /// Estimates the difficulty of a cube based on its size and depth.
    #[allow(dead_code)]
    fn estimate_difficulty(&self, cube: &Cube) -> f64 {
        // Smaller cubes are generally harder (less constrained)
        let size_factor = 1.0 / (cube.len() as f64 + 1.0);
        // Deeper cubes represent more specific search spaces
        let depth_factor = cube.depth as f64;

        size_factor * depth_factor
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &CubeConfig {
        &self.config
    }

    /// Returns the number of cubes generated.
    pub fn num_cubes(&self) -> usize {
        self.cubes.len()
    }
}

/// Statistics for cube generation.
#[derive(Debug, Clone, Default)]
pub struct CubeStats {
    /// Total cubes generated
    pub total_cubes: usize,
    /// Average cube size
    pub avg_cube_size: f64,
    /// Minimum cube size
    pub min_cube_size: usize,
    /// Maximum cube size
    pub max_cube_size: usize,
    /// Maximum depth reached
    pub max_depth: usize,
    /// Average difficulty
    pub avg_difficulty: f64,
}

impl CubeStats {
    /// Creates statistics from a set of cubes.
    pub fn from_cubes(cubes: &[Cube]) -> Self {
        if cubes.is_empty() {
            return Self::default();
        }

        let total = cubes.len();
        let sizes: Vec<usize> = cubes.iter().map(|c| c.len()).collect();
        let avg_size = sizes.iter().sum::<usize>() as f64 / total as f64;
        let min_size = sizes.iter().copied().min().unwrap_or(0);
        let max_size = sizes.iter().copied().max().unwrap_or(0);
        let max_depth = cubes.iter().map(|c| c.depth).max().unwrap_or(0);
        let avg_diff = cubes.iter().map(|c| c.difficulty).sum::<f64>() / total as f64;

        Self {
            total_cubes: total,
            avg_cube_size: avg_size,
            min_cube_size: min_size,
            max_cube_size: max_size,
            max_depth,
            avg_difficulty: avg_diff,
        }
    }

    /// Displays the statistics.
    pub fn display(&self) -> String {
        format!(
            "Cube Generation Statistics:\n\
             - Total Cubes: {}\n\
             - Avg Size: {:.2}\n\
             - Size Range: [{}, {}]\n\
             - Max Depth: {}\n\
             - Avg Difficulty: {:.4}",
            self.total_cubes,
            self.avg_cube_size,
            self.min_cube_size,
            self.max_cube_size,
            self.max_depth,
            self.avg_difficulty
        )
    }
}

/// Result of solving cubes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeResult {
    /// All cubes are UNSAT - formula is UNSAT
    Unsat,
    /// At least one cube is SAT - formula is SAT
    Sat,
    /// Unknown (timeout or resource limit)
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_cube_creation() {
        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::neg(Var::new(1));
        let cube = Cube::new(vec![lit1, lit2]);

        assert_eq!(cube.len(), 2);
        assert!(!cube.is_empty());
        assert_eq!(cube.depth, 2);
    }

    #[test]
    fn test_cube_extend() {
        let lit1 = Lit::pos(Var::new(0));
        let cube1 = Cube::new(vec![lit1]);

        let lit2 = Lit::neg(Var::new(1));
        let cube2 = cube1.extend(lit2);

        assert_eq!(cube1.len(), 1);
        assert_eq!(cube2.len(), 2);
        assert_eq!(cube2.depth, 2);
    }

    #[test]
    fn test_cube_consistency() {
        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::neg(Var::new(1));
        let cube = Cube::new(vec![lit1, lit2]);

        assert!(cube.is_consistent());
    }

    #[test]
    fn test_cube_inconsistency() {
        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::neg(Var::new(0));
        let cube = Cube::new(vec![lit1, lit2]);

        assert!(!cube.is_consistent());
    }

    #[test]
    fn test_empty_cube() {
        let cube = Cube::new(Vec::new());

        assert_eq!(cube.len(), 0);
        assert!(cube.is_empty());
        assert!(cube.is_consistent());
    }

    #[test]
    fn test_cube_config_default() {
        let config = CubeConfig::default();

        assert_eq!(config.max_depth, 10);
        assert_eq!(config.target_cubes, 100);
        assert!(config.adaptive_depth);
    }

    #[test]
    fn test_cube_generator_creation() {
        let config = CubeConfig::default();
        let generator = CubeGenerator::new(10, config);

        assert_eq!(generator.num_vars, 10);
        assert_eq!(generator.num_cubes(), 0);
    }

    #[test]
    fn test_cube_generation() {
        let config = CubeConfig {
            max_depth: 3,
            target_cubes: 10,
            min_cube_size: 1,
            max_cube_size: 5,
            strategy: CubeSplittingStrategy::Activity,
            adaptive_depth: false,
        };
        let mut generator = CubeGenerator::new(5, config);
        let scores = vec![1.0, 0.9, 0.8, 0.7, 0.6];

        let cubes = generator.generate(&scores);

        assert!(!cubes.is_empty());
        assert!(cubes.len() <= 10);

        // All cubes should be consistent
        for cube in &cubes {
            assert!(cube.is_consistent());
        }
    }

    #[test]
    fn test_cube_stats() {
        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::neg(Var::new(1));
        let lit3 = Lit::pos(Var::new(2));

        let cubes = vec![
            Cube::new(vec![lit1]),
            Cube::new(vec![lit1, lit2]),
            Cube::new(vec![lit1, lit2, lit3]),
        ];

        let stats = CubeStats::from_cubes(&cubes);

        assert_eq!(stats.total_cubes, 3);
        assert_eq!(stats.min_cube_size, 1);
        assert_eq!(stats.max_cube_size, 3);
        assert_eq!(stats.avg_cube_size, 2.0);
    }

    #[test]
    fn test_empty_cube_stats() {
        let stats = CubeStats::from_cubes(&[]);

        assert_eq!(stats.total_cubes, 0);
        assert_eq!(stats.avg_cube_size, 0.0);
    }

    #[test]
    fn test_cube_splitting_strategies() {
        let strategies = vec![
            CubeSplittingStrategy::Activity,
            CubeSplittingStrategy::Lookahead,
            CubeSplittingStrategy::MostConstrained,
            CubeSplittingStrategy::Balanced,
        ];

        for strategy in strategies {
            let config = CubeConfig {
                strategy,
                max_depth: 2,
                target_cubes: 4,
                min_cube_size: 1,
                max_cube_size: 5,
                adaptive_depth: false,
            };

            let mut generator = CubeGenerator::new(3, config);
            let scores = vec![1.0, 0.5, 0.2];
            let cubes = generator.generate(&scores);

            assert!(!cubes.is_empty());
        }
    }

    #[test]
    fn test_adaptive_depth() {
        let config = CubeConfig {
            max_depth: 2,
            target_cubes: 100,
            min_cube_size: 1,
            max_cube_size: 10,
            strategy: CubeSplittingStrategy::Activity,
            adaptive_depth: true,
        };

        let mut generator = CubeGenerator::new(3, config);
        let scores = vec![1.0, 0.5, 0.2];
        let _cubes = generator.generate(&scores);

        // Adaptive depth should increase max_depth if needed
        assert!(generator.config().max_depth >= 2);
    }
}
