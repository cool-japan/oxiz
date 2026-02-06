//! Cell Decomposition for CAD.
//!
//! Manages the hierarchical cell structure produced by CAD, providing
//! efficient access and manipulation of cells at different levels.
//!
//! ## Cell Structure
//!
//! - **Stack**: Cells organized by dimension (R^1, R^2, ..., R^n)
//! - **Parent-Child**: Each cell has parent in lower dimension
//! - **Sample Points**: Representative point in each cell
//! - **Sign Invariance**: Polynomials have constant sign in each cell
//!
//! ## References
//!
//! - Collins: "Quantifier Elimination for Real Closed Fields by CAD" (1975)
//! - Z3's `qe/qe_arith.cpp`

use rustc_hash::FxHashMap;

use crate::TermId;
use crate::qe::cad::{CellType, SamplePoint};

/// Cell identifier.
pub type CellId = usize;

/// Dimension level (0 = R^0, 1 = R^1, etc.).
pub type Level = usize;

/// A cell in the decomposition with metadata.
#[derive(Debug, Clone)]
pub struct DecomposedCell {
    /// Unique identifier.
    pub id: CellId,
    /// Dimension level.
    pub level: Level,
    /// Parent cell (if level > 0).
    pub parent: Option<CellId>,
    /// Children cells (if not leaf).
    pub children: Vec<CellId>,
    /// Cell type (sector, section, etc.).
    pub cell_type: CellType,
    /// Sample point in this cell.
    pub sample: SamplePoint,
    /// Polynomial signs in this cell.
    pub signs: FxHashMap<TermId, i8>, // polynomial -> sign (-1, 0, 1)
    /// Whether this cell satisfies the formula.
    pub satisfies: Option<bool>,
}

/// Configuration for cell decomposition.
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Maximum number of cells to create.
    pub max_cells: usize,
    /// Enable cell pruning (remove unsatisfying cells).
    pub enable_pruning: bool,
    /// Cache polynomial evaluations.
    pub cache_evaluations: bool,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            max_cells: 100000,
            enable_pruning: true,
            cache_evaluations: true,
        }
    }
}

/// Statistics for cell decomposition.
#[derive(Debug, Clone, Default)]
pub struct DecompositionStats {
    /// Total cells created.
    pub cells_created: u64,
    /// Cells pruned.
    pub cells_pruned: u64,
    /// Sign evaluations performed.
    pub sign_evaluations: u64,
    /// Sample points computed.
    pub samples_computed: u64,
    /// Average children per cell.
    pub avg_children: f64,
}

/// Cell decomposition manager.
pub struct CellDecomposition {
    /// Configuration.
    config: DecompositionConfig,
    /// Statistics.
    stats: DecompositionStats,
    /// All cells by ID.
    cells: FxHashMap<CellId, DecomposedCell>,
    /// Cells organized by level.
    cells_by_level: Vec<Vec<CellId>>,
    /// Next cell ID.
    next_id: CellId,
    /// Maximum dimension.
    max_dimension: usize,
}

impl CellDecomposition {
    /// Create a new cell decomposition.
    pub fn new(config: DecompositionConfig, dimension: usize) -> Self {
        let mut cells_by_level = Vec::new();
        for _ in 0..=dimension {
            cells_by_level.push(Vec::new());
        }

        Self {
            config,
            stats: DecompositionStats::default(),
            cells: FxHashMap::default(),
            cells_by_level,
            next_id: 0,
            max_dimension: dimension,
        }
    }

    /// Create with default configuration.
    pub fn default_config(dimension: usize) -> Self {
        Self::new(DecompositionConfig::default(), dimension)
    }

    /// Add a cell to the decomposition.
    pub fn add_cell(
        &mut self,
        level: Level,
        parent: Option<CellId>,
        cell_type: CellType,
        sample: SamplePoint,
    ) -> Result<CellId, DecompositionError> {
        if self.cells.len() >= self.config.max_cells {
            return Err(DecompositionError::TooManyCells);
        }

        if level > self.max_dimension {
            return Err(DecompositionError::InvalidLevel);
        }

        let id = self.next_id;
        self.next_id += 1;

        let cell = DecomposedCell {
            id,
            level,
            parent,
            children: Vec::new(),
            cell_type,
            sample,
            signs: FxHashMap::default(),
            satisfies: None,
        };

        // Add to parent's children list
        if let Some(parent_id) = parent
            && let Some(parent_cell) = self.cells.get_mut(&parent_id)
        {
            parent_cell.children.push(id);
            self.update_avg_children();
        }

        self.cells.insert(id, cell);
        self.cells_by_level[level].push(id);
        self.stats.cells_created += 1;

        Ok(id)
    }

    /// Remove a cell (and all descendants).
    pub fn remove_cell(&mut self, id: CellId) {
        if let Some(cell) = self.cells.get(&id).cloned() {
            // Remove all children recursively
            for child_id in cell.children {
                self.remove_cell(child_id);
            }

            // Remove from parent's children list
            if let Some(parent_id) = cell.parent
                && let Some(parent) = self.cells.get_mut(&parent_id)
            {
                parent.children.retain(|&c| c != id);
            }

            // Remove from level list
            self.cells_by_level[cell.level].retain(|&c| c != id);

            // Remove the cell itself
            self.cells.remove(&id);
            self.stats.cells_pruned += 1;
        }
    }

    /// Get a cell by ID.
    pub fn get_cell(&self, id: CellId) -> Option<&DecomposedCell> {
        self.cells.get(&id)
    }

    /// Get all cells at a level.
    pub fn cells_at_level(&self, level: Level) -> &[CellId] {
        &self.cells_by_level[level]
    }

    /// Set polynomial sign in a cell.
    pub fn set_sign(&mut self, cell_id: CellId, poly: TermId, sign: i8) {
        if let Some(cell) = self.cells.get_mut(&cell_id) {
            cell.signs.insert(poly, sign);
            self.stats.sign_evaluations += 1;
        }
    }

    /// Get polynomial sign in a cell.
    pub fn get_sign(&self, cell_id: CellId, poly: TermId) -> Option<i8> {
        self.cells
            .get(&cell_id)
            .and_then(|c| c.signs.get(&poly).copied())
    }

    /// Mark a cell as satisfying or not satisfying the formula.
    pub fn set_satisfies(&mut self, cell_id: CellId, satisfies: bool) {
        if let Some(cell) = self.cells.get_mut(&cell_id) {
            cell.satisfies = Some(satisfies);
        }
    }

    /// Prune cells that don't satisfy the formula.
    pub fn prune_unsatisfying(&mut self) {
        if !self.config.enable_pruning {
            return;
        }

        let to_remove: Vec<CellId> = self
            .cells
            .values()
            .filter(|c| c.satisfies == Some(false))
            .map(|c| c.id)
            .collect();

        for id in to_remove {
            self.remove_cell(id);
        }
    }

    /// Get all leaf cells (cells with no children).
    pub fn leaf_cells(&self) -> Vec<CellId> {
        self.cells
            .values()
            .filter(|c| c.children.is_empty())
            .map(|c| c.id)
            .collect()
    }

    /// Get all satisfying cells.
    pub fn satisfying_cells(&self) -> Vec<CellId> {
        self.cells
            .values()
            .filter(|c| c.satisfies == Some(true))
            .map(|c| c.id)
            .collect()
    }

    /// Get total number of cells.
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Get number of cells at each level.
    pub fn cells_per_level(&self) -> Vec<usize> {
        self.cells_by_level.iter().map(|v| v.len()).collect()
    }

    /// Update average children statistics.
    fn update_avg_children(&mut self) {
        let non_leaf_cells: Vec<&DecomposedCell> = self
            .cells
            .values()
            .filter(|c| !c.children.is_empty())
            .collect();

        if non_leaf_cells.is_empty() {
            self.stats.avg_children = 0.0;
        } else {
            let total_children: usize = non_leaf_cells.iter().map(|c| c.children.len()).sum();
            self.stats.avg_children = total_children as f64 / non_leaf_cells.len() as f64;
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &DecompositionStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DecompositionStats::default();
    }

    /// Clear all cells.
    pub fn clear(&mut self) {
        self.cells.clear();
        for level in &mut self.cells_by_level {
            level.clear();
        }
        self.next_id = 0;
    }
}

/// Errors for cell decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionError {
    /// Too many cells created.
    TooManyCells,
    /// Invalid dimension level.
    InvalidLevel,
    /// Cell not found.
    CellNotFound,
}

impl std::fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompositionError::TooManyCells => write!(f, "too many cells"),
            DecompositionError::InvalidLevel => write!(f, "invalid level"),
            DecompositionError::CellNotFound => write!(f, "cell not found"),
        }
    }
}

impl std::error::Error for DecompositionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;
    use num_traits::{One, Zero};

    #[test]
    fn test_decomposition_creation() {
        let decomp = CellDecomposition::default_config(3);
        assert_eq!(decomp.num_cells(), 0);
    }

    #[test]
    fn test_add_cell() {
        let mut decomp = CellDecomposition::default_config(2);

        let sample = SamplePoint::new(vec![BigRational::zero()]);
        let id = decomp
            .add_cell(1, None, CellType::Section, sample)
            .expect("failed to add cell");

        assert_eq!(decomp.num_cells(), 1);
        assert!(decomp.get_cell(id).is_some());
    }

    #[test]
    fn test_parent_child() {
        let mut decomp = CellDecomposition::default_config(2);

        let sample1 = SamplePoint::new(vec![BigRational::zero()]);
        let parent_id = decomp
            .add_cell(0, None, CellType::Section, sample1)
            .expect("failed");

        let sample2 = SamplePoint::new(vec![BigRational::zero(), BigRational::one()]);
        let child_id = decomp
            .add_cell(1, Some(parent_id), CellType::Section, sample2)
            .expect("failed");

        let parent = decomp.get_cell(parent_id).unwrap();
        assert_eq!(parent.children, vec![child_id]);

        let child = decomp.get_cell(child_id).unwrap();
        assert_eq!(child.parent, Some(parent_id));
    }

    #[test]
    fn test_remove_cell() {
        let mut decomp = CellDecomposition::default_config(2);

        let sample = SamplePoint::new(vec![BigRational::zero()]);
        let id = decomp
            .add_cell(1, None, CellType::Section, sample)
            .expect("failed");

        assert_eq!(decomp.num_cells(), 1);

        decomp.remove_cell(id);
        assert_eq!(decomp.num_cells(), 0);
    }

    #[test]
    fn test_pruning() {
        let mut decomp = CellDecomposition::default_config(2);

        let sample = SamplePoint::new(vec![BigRational::zero()]);
        let id = decomp
            .add_cell(1, None, CellType::Section, sample)
            .expect("failed");

        decomp.set_satisfies(id, false);
        decomp.prune_unsatisfying();

        assert_eq!(decomp.num_cells(), 0);
    }

    #[test]
    fn test_stats() {
        let decomp = CellDecomposition::default_config(3);
        assert_eq!(decomp.stats().cells_created, 0);
    }
}
