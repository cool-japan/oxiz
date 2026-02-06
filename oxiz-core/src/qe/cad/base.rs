//! CAD Base Case (R^1 Decomposition).
#![allow(dead_code, unused_assignments)] // Under development - CAD not yet fully integrated
//!
//! The base case of cylindrical algebraic decomposition handles univariate
//! polynomials over R^1. This phase isolates real roots and decomposes the
//! real line into intervals and points where polynomials have constant signs.
//!
//! ## Algorithm
//!
//! 1. Collect all projection polynomials (univariate)
//! 2. Isolate real roots of each polynomial
//! 3. Merge roots to create ordered sequence
//! 4. Create cells: points (roots) and intervals (between roots)
//!
//! ## References
//!
//! - Collins: "Quantifier Elimination" (1975)
//! - Z3's `qe/qe_arith_plugin.cpp`

use super::lifting::{Cell, CellType};
use crate::ast::TermId;
use num_bigint::BigInt;
use num_rational::BigRational;

/// Configuration for base case.
#[derive(Debug, Clone)]
pub struct BaseConfig {
    /// Root isolation precision.
    pub precision: BigRational,
    /// Maximum roots per polynomial.
    pub max_roots: usize,
}

impl Default for BaseConfig {
    fn default() -> Self {
        Self {
            precision: BigRational::new(BigInt::from(1), BigInt::from(1000)),
            max_roots: 1000,
        }
    }
}

/// Statistics for base case.
#[derive(Debug, Clone, Default)]
pub struct BaseStats {
    /// Total cells created.
    pub cells_created: u64,
    /// Total roots isolated.
    pub roots_isolated: u64,
    /// Polynomials processed.
    pub polynomials_processed: u64,
}

/// CAD base case engine for R^1.
pub struct BaseCase {
    /// Configuration.
    config: BaseConfig,
    /// Statistics.
    stats: BaseStats,
}

impl BaseCase {
    /// Create a new base case engine.
    pub fn new(config: BaseConfig) -> Self {
        Self {
            config,
            stats: BaseStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(BaseConfig::default())
    }

    /// Decompose R^1 based on projection polynomials.
    ///
    /// Returns a list of cells covering R^1.
    pub fn decompose(&mut self, projection_polynomials: Vec<TermId>) -> Vec<Cell> {
        let mut all_roots = Vec::new();

        // Isolate roots for each polynomial
        for poly in projection_polynomials {
            self.stats.polynomials_processed += 1;
            let roots = self.isolate_roots(poly);
            all_roots.extend(roots);
        }

        // Sort and deduplicate roots
        all_roots.sort();
        all_roots.dedup();

        self.stats.roots_isolated = all_roots.len() as u64;

        // Create cells from roots
        self.create_cells(all_roots)
    }

    /// Isolate real roots of a univariate polynomial.
    fn isolate_roots(&mut self, _poly: TermId) -> Vec<BigRational> {
        // In a real implementation, this would:
        // 1. Convert TermId to actual polynomial
        // 2. Use Sturm sequences or Descartes' rule
        // 3. Perform bisection to isolate roots

        // Simplified: return empty (no roots found)
        Vec::new()
    }

    /// Create cells from an ordered sequence of roots.
    fn create_cells(&mut self, roots: Vec<BigRational>) -> Vec<Cell> {
        let mut cells = Vec::new();
        let mut cell_index = 0;

        if roots.is_empty() {
            // No roots - entire R^1 is one sector
            cells.push(Cell {
                dimension: 1,
                sample_point: vec![BigRational::new(BigInt::from(0), BigInt::from(1))],
                defining_polynomials: vec![],
                cell_type: CellType::Sector,
                index: cell_index,
            });
            cell_index += 1;
        } else {
            // Sector before first root: (-inf, r_0)
            cells.push(Cell {
                dimension: 1,
                sample_point: vec![&roots[0] - BigRational::new(BigInt::from(1), BigInt::from(1))],
                defining_polynomials: vec![],
                cell_type: CellType::Sector,
                index: cell_index,
            });
            cell_index += 1;

            // Alternating sections (at roots) and sectors (between roots)
            for i in 0..roots.len() {
                // Section at root r_i
                cells.push(Cell {
                    dimension: 1,
                    sample_point: vec![roots[i].clone()],
                    defining_polynomials: vec![],
                    cell_type: CellType::Section,
                    index: cell_index,
                });
                cell_index += 1;

                // Sector between r_i and r_{i+1} (if not last root)
                if i < roots.len() - 1 {
                    let mid = (&roots[i] + &roots[i + 1])
                        / BigRational::new(BigInt::from(2), BigInt::from(1));
                    cells.push(Cell {
                        dimension: 1,
                        sample_point: vec![mid],
                        defining_polynomials: vec![],
                        cell_type: CellType::Sector,
                        index: cell_index,
                    });
                    cell_index += 1;
                }
            }

            // Sector after last root: (r_n, +inf)
            // Safety: we're in the else branch so roots is non-empty, but use if-let for no-unwrap policy
            if let Some(last_root) = roots.last() {
                cells.push(Cell {
                    dimension: 1,
                    sample_point: vec![
                        last_root + BigRational::new(BigInt::from(1), BigInt::from(1)),
                    ],
                    defining_polynomials: vec![],
                    cell_type: CellType::Sector,
                    index: cell_index,
                });
            }
        }

        self.stats.cells_created = cells.len() as u64;
        cells
    }

    /// Get statistics.
    pub fn stats(&self) -> &BaseStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = BaseStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_base_case_creation() {
        let base = BaseCase::default_config();
        assert_eq!(base.stats().cells_created, 0);
    }

    #[test]
    fn test_decompose_no_roots() {
        let mut base = BaseCase::default_config();

        // No projection polynomials
        let cells = base.decompose(vec![]);

        // Should create 1 sector covering all of R^1
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].cell_type, CellType::Sector);
        assert_eq!(cells[0].dimension, 1);
    }

    #[test]
    fn test_create_cells_single_root() {
        let mut base = BaseCase::default_config();

        let roots = vec![rat(0)];
        let cells = base.create_cells(roots);

        // Should create: sector, section, sector (3 cells)
        assert_eq!(cells.len(), 3);
        assert_eq!(cells[0].cell_type, CellType::Sector); // (-inf, 0)
        assert_eq!(cells[1].cell_type, CellType::Section); // {0}
        assert_eq!(cells[2].cell_type, CellType::Sector); // (0, +inf)
    }

    #[test]
    fn test_create_cells_multiple_roots() {
        let mut base = BaseCase::default_config();

        let roots = vec![rat(-1), rat(0), rat(1)];
        let cells = base.create_cells(roots);

        // Should create: S S S S S S S (7 cells: 4 sectors, 3 sections)
        assert_eq!(cells.len(), 7);

        // Check pattern
        assert_eq!(cells[0].cell_type, CellType::Sector); // (-inf, -1)
        assert_eq!(cells[1].cell_type, CellType::Section); // {-1}
        assert_eq!(cells[2].cell_type, CellType::Sector); // (-1, 0)
        assert_eq!(cells[3].cell_type, CellType::Section); // {0}
        assert_eq!(cells[4].cell_type, CellType::Sector); // (0, 1)
        assert_eq!(cells[5].cell_type, CellType::Section); // {1}
        assert_eq!(cells[6].cell_type, CellType::Sector); // (1, +inf)
    }

    #[test]
    fn test_sample_points() {
        let mut base = BaseCase::default_config();

        let roots = vec![rat(0), rat(2)];
        let cells = base.create_cells(roots);

        // Check sample points
        assert!(cells[0].sample_point[0] < rat(0)); // Before 0
        assert_eq!(cells[1].sample_point[0], rat(0)); // At 0
        assert!(cells[2].sample_point[0] > rat(0) && cells[2].sample_point[0] < rat(2)); // Between 0 and 2
        assert_eq!(cells[3].sample_point[0], rat(2)); // At 2
        assert!(cells[4].sample_point[0] > rat(2)); // After 2
    }

    #[test]
    fn test_stats() {
        let mut base = BaseCase::default_config();

        let roots = vec![rat(-1), rat(1)];
        let _ = base.create_cells(roots);

        assert_eq!(base.stats().cells_created, 5); // 3 sectors + 2 sections
    }
}
