//! CAD Lifting Phase.
//!
//! The lifting phase constructs cylindrical cells in R^n by recursively lifting
//! the decomposition from R^(n-1). For each cell in the lower dimension, we
//! compute roots of polynomials to determine cell boundaries in the next dimension.
//!
//! ## Algorithm
//!
//! 1. Start with R^1 decomposition (base case)
//! 2. For each cell C in R^k:
//!    a. Evaluate projection polynomials over C
//!    b. Find roots to get critical points
//!    c. Create cells between and at roots
//! 3. Continue until R^n is reached
//!
//! ## References
//!
//! - Collins: "Quantifier Elimination for Real Closed Fields" (1975)
//! - Brown: "Improved Projection for CAD" (2001)
//! - Z3's `qe/qe_arith_plugin.cpp`

use crate::ast::TermId;
use num_rational::BigRational;
use rustc_hash::FxHashMap;

/// Variable identifier.
pub type VarId = usize;

/// A cylindrical cell in R^n.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Dimension of this cell.
    pub dimension: usize,
    /// Sample point representing this cell.
    pub sample_point: Vec<BigRational>,
    /// Defining polynomial terms (for boundaries).
    pub defining_polynomials: Vec<TermId>,
    /// Cell type (section, sector, or mixed).
    pub cell_type: CellType,
    /// Index in the cylindrical structure.
    pub index: usize,
}

/// Type of cylindrical cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellType {
    /// Section: cell on a surface (polynomial = 0).
    Section,
    /// Sector: cell between surfaces (polynomial != 0).
    Sector,
    /// Mixed: product of section and sector in different dimensions.
    Mixed,
}

/// Configuration for lifting phase.
#[derive(Debug, Clone)]
pub struct LiftingConfig {
    /// Enable caching of polynomial evaluations.
    pub enable_caching: bool,
    /// Maximum cells per dimension before pruning.
    pub max_cells_per_dimension: usize,
    /// Sample point selection strategy.
    pub sample_strategy: SampleStrategy,
}

impl Default for LiftingConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cells_per_dimension: 1000,
            sample_strategy: SampleStrategy::Midpoint,
        }
    }
}

/// Strategy for selecting sample points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStrategy {
    /// Use midpoint of interval.
    Midpoint,
    /// Use rational approximation of algebraic number.
    RationalApproximation,
    /// Use algebraic number directly.
    Algebraic,
}

/// Statistics for lifting phase.
#[derive(Debug, Clone, Default)]
pub struct LiftingStats {
    /// Total cells created.
    pub cells_created: u64,
    /// Polynomial evaluations.
    pub polynomial_evals: u64,
    /// Root isolations performed.
    pub root_isolations: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
}

/// CAD lifting engine.
pub struct LiftingEngine {
    /// Configuration.
    config: LiftingConfig,
    /// Statistics.
    stats: LiftingStats,
    /// Cache for polynomial evaluations.
    eval_cache: FxHashMap<(usize, Vec<BigRational>), BigRational>,
    /// Cells at each dimension.
    cells_by_dimension: FxHashMap<usize, Vec<Cell>>,
    /// Next cell index.
    next_cell_index: usize,
}

impl LiftingEngine {
    /// Create a new lifting engine.
    pub fn new(config: LiftingConfig) -> Self {
        Self {
            config,
            stats: LiftingStats::default(),
            eval_cache: FxHashMap::default(),
            cells_by_dimension: FxHashMap::default(),
            next_cell_index: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(LiftingConfig::default())
    }

    /// Lift decomposition from dimension k to k+1.
    pub fn lift_dimension(
        &mut self,
        lower_cells: Vec<Cell>,
        projection_polynomials: Vec<TermId>,
        variable: VarId,
    ) -> Vec<Cell> {
        let mut lifted_cells = Vec::new();

        // Get dimension before moving
        let dimension = lower_cells.first().map(|c| c.dimension + 1).unwrap_or(1);

        for lower_cell in lower_cells {
            // For each lower-dimensional cell, lift to next dimension
            let new_cells = self.lift_cell(&lower_cell, &projection_polynomials, variable);
            lifted_cells.extend(new_cells);
        }

        // Store in dimension map
        self.cells_by_dimension
            .insert(dimension, lifted_cells.clone());

        lifted_cells
    }

    /// Lift a single cell to the next dimension.
    fn lift_cell(
        &mut self,
        cell: &Cell,
        projection_polynomials: &[TermId],
        variable: VarId,
    ) -> Vec<Cell> {
        let mut new_cells = Vec::new();

        // Evaluate projection polynomials at cell's sample point
        let mut roots: Vec<BigRational> = Vec::new();

        for &poly in projection_polynomials {
            // Isolate roots of poly when restricted to this cell
            let cell_roots = self.isolate_roots_in_cell(poly, cell, variable);
            roots.extend(cell_roots);
        }

        // Sort roots
        roots.sort();
        roots.dedup();

        // Create sectors (between roots) and sections (at roots)
        if roots.is_empty() {
            // No roots - entire R^1 is one sector
            let sample = self.choose_sample_point(&[], cell);
            new_cells.push(self.create_cell(cell, sample, CellType::Sector));
        } else {
            // Sector before first root
            let sample = self.choose_sample_point_before(&roots[0], cell);
            new_cells.push(self.create_cell(cell, sample, CellType::Sector));

            // Alternate sections and sectors
            for i in 0..roots.len() {
                // Section at root
                let mut section_sample = cell.sample_point.clone();
                section_sample.push(roots[i].clone());
                new_cells.push(self.create_cell(cell, section_sample, CellType::Section));

                // Sector after root (if not last)
                if i < roots.len() - 1 {
                    let sample = self.choose_sample_point_between(&roots[i], &roots[i + 1], cell);
                    new_cells.push(self.create_cell(cell, sample, CellType::Sector));
                }
            }

            // Sector after last root
            // Safety: we're in the else branch so roots is non-empty, but use if-let for no-unwrap policy
            if let Some(last_root) = roots.last() {
                let sample = self.choose_sample_point_after(last_root, cell);
                new_cells.push(self.create_cell(cell, sample, CellType::Sector));
            }
        }

        self.stats.cells_created += new_cells.len() as u64;
        new_cells
    }

    /// Isolate roots of polynomial in a cell.
    fn isolate_roots_in_cell(
        &mut self,
        _poly: TermId,
        _cell: &Cell,
        _variable: VarId,
    ) -> Vec<BigRational> {
        // Evaluate polynomial coefficients at cell's sample point
        // Then isolate roots of the resulting univariate polynomial

        // For now, simplified: use polynomial's root isolation directly
        // Real implementation needs partial evaluation

        self.stats.root_isolations += 1;

        // Placeholder: assume no roots in cell
        Vec::new()
    }

    /// Choose sample point before a value.
    fn choose_sample_point_before(&self, value: &BigRational, cell: &Cell) -> Vec<BigRational> {
        let mut sample = cell.sample_point.clone();

        match self.config.sample_strategy {
            SampleStrategy::Midpoint => {
                // Choose value - 1
                use num_bigint::BigInt;
                sample.push(value - BigRational::from_integer(BigInt::from(1)));
            }
            _ => {
                sample.push(value - BigRational::from_integer(num_bigint::BigInt::from(1)));
            }
        }

        sample
    }

    /// Choose sample point between two values.
    fn choose_sample_point_between(
        &self,
        a: &BigRational,
        b: &BigRational,
        cell: &Cell,
    ) -> Vec<BigRational> {
        let mut sample = cell.sample_point.clone();

        match self.config.sample_strategy {
            SampleStrategy::Midpoint => {
                // Choose (a + b) / 2
                use num_bigint::BigInt;
                let two = BigRational::from_integer(BigInt::from(2));
                sample.push((a + b) / two);
            }
            _ => {
                let two = BigRational::from_integer(num_bigint::BigInt::from(2));
                sample.push((a + b) / two);
            }
        }

        sample
    }

    /// Choose sample point after a value.
    fn choose_sample_point_after(&self, value: &BigRational, cell: &Cell) -> Vec<BigRational> {
        let mut sample = cell.sample_point.clone();

        match self.config.sample_strategy {
            SampleStrategy::Midpoint => {
                // Choose value + 1
                use num_bigint::BigInt;
                sample.push(value + BigRational::from_integer(BigInt::from(1)));
            }
            _ => {
                sample.push(value + BigRational::from_integer(num_bigint::BigInt::from(1)));
            }
        }

        sample
    }

    /// Choose sample point in unbounded region.
    fn choose_sample_point(&self, _roots: &[BigRational], cell: &Cell) -> Vec<BigRational> {
        let mut sample = cell.sample_point.clone();

        // Choose 0 as sample point
        use num_bigint::BigInt;
        sample.push(BigRational::from_integer(BigInt::from(0)));

        sample
    }

    /// Create a new cell.
    fn create_cell(
        &mut self,
        parent: &Cell,
        sample_point: Vec<BigRational>,
        cell_type: CellType,
    ) -> Cell {
        let cell = Cell {
            dimension: parent.dimension + 1,
            sample_point,
            defining_polynomials: parent.defining_polynomials.clone(),
            cell_type,
            index: self.next_cell_index,
        };

        self.next_cell_index += 1;
        cell
    }

    /// Get cells at a specific dimension.
    pub fn get_cells(&self, dimension: usize) -> Option<&Vec<Cell>> {
        self.cells_by_dimension.get(&dimension)
    }

    /// Get total number of cells.
    pub fn total_cells(&self) -> usize {
        self.cells_by_dimension.values().map(|v| v.len()).sum()
    }

    /// Get statistics.
    pub fn stats(&self) -> &LiftingStats {
        &self.stats
    }

    /// Reset engine state.
    pub fn reset(&mut self) {
        self.eval_cache.clear();
        self.cells_by_dimension.clear();
        self.next_cell_index = 0;
        self.stats = LiftingStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_engine_creation() {
        let engine = LiftingEngine::default_config();
        assert_eq!(engine.stats().cells_created, 0);
    }

    #[test]
    fn test_create_cell() {
        let mut engine = LiftingEngine::default_config();

        let parent = Cell {
            dimension: 0,
            sample_point: vec![],
            defining_polynomials: vec![],
            cell_type: CellType::Sector,
            index: 0,
        };

        let new_cell = engine.create_cell(&parent, vec![rat(1)], CellType::Section);

        assert_eq!(new_cell.dimension, 1);
        assert_eq!(new_cell.sample_point.len(), 1);
        assert_eq!(new_cell.cell_type, CellType::Section);
    }

    #[test]
    fn test_lift_cell_no_roots() {
        let mut engine = LiftingEngine::default_config();

        let cell = Cell {
            dimension: 0,
            sample_point: vec![],
            defining_polynomials: vec![],
            cell_type: CellType::Sector,
            index: 0,
        };

        let projection_polys = vec![]; // No polynomials -> no roots

        let lifted = engine.lift_cell(&cell, &projection_polys, 0);

        // Should create 1 sector (entire R^1)
        assert_eq!(lifted.len(), 1);
        assert_eq!(lifted[0].cell_type, CellType::Sector);
    }

    #[test]
    fn test_sample_point_selection() {
        let engine = LiftingEngine::default_config();

        let cell = Cell {
            dimension: 0,
            sample_point: vec![],
            defining_polynomials: vec![],
            cell_type: CellType::Sector,
            index: 0,
        };

        let sample_before = engine.choose_sample_point_before(&rat(5), &cell);
        assert_eq!(sample_before.len(), 1);
        assert!(sample_before[0] < rat(5));

        let sample_between = engine.choose_sample_point_between(&rat(1), &rat(5), &cell);
        assert_eq!(sample_between.len(), 1);
        assert!(sample_between[0] > rat(1) && sample_between[0] < rat(5));

        let sample_after = engine.choose_sample_point_after(&rat(5), &cell);
        assert_eq!(sample_after.len(), 1);
        assert!(sample_after[0] > rat(5));
    }

    #[test]
    fn test_total_cells() {
        let mut engine = LiftingEngine::default_config();

        let cell1 = Cell {
            dimension: 1,
            sample_point: vec![rat(0)],
            defining_polynomials: vec![],
            cell_type: CellType::Sector,
            index: 0,
        };

        let cell2 = Cell {
            dimension: 1,
            sample_point: vec![rat(1)],
            defining_polynomials: vec![],
            cell_type: CellType::Section,
            index: 1,
        };

        engine.cells_by_dimension.insert(1, vec![cell1, cell2]);

        assert_eq!(engine.total_cells(), 2);
    }
}
