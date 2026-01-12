//! Clause exchange for parallel/portfolio SAT solving
//!
//! This module implements efficient clause sharing between multiple solver instances
//! running in parallel. High-quality learned clauses are exchanged to improve overall
//! solving performance.
//!
//! References:
//! - "Cube and Conquer: Guiding CDCL SAT Solvers by Lookaheads"
//! - "Portfolio-based Parallel SAT Solving"
//! - "ManySAT: a Parallel SAT Solver"

use crate::clause::Clause;
use crate::literal::Lit;
use smallvec::SmallVec;
use std::sync::{Arc, Mutex};

/// A shared clause that can be exchanged between solvers
#[derive(Debug, Clone)]
pub struct SharedClause {
    /// The literals in the clause
    pub lits: SmallVec<[Lit; 4]>,
    /// LBD (quality metric)
    pub lbd: u32,
    /// Activity (for prioritization)
    pub activity: f64,
    /// Originating solver ID
    pub source_id: usize,
}

impl SharedClause {
    /// Create a new shared clause
    #[must_use]
    pub fn new(lits: SmallVec<[Lit; 4]>, lbd: u32, activity: f64, source_id: usize) -> Self {
        Self {
            lits,
            lbd,
            activity,
            source_id,
        }
    }

    /// Create from a regular clause
    #[must_use]
    pub fn from_clause(clause: &Clause, source_id: usize) -> Self {
        Self {
            lits: clause.lits.clone(),
            lbd: clause.lbd,
            activity: clause.activity,
            source_id,
        }
    }

    /// Check if this clause should be shared based on quality
    #[must_use]
    pub fn is_shareable(&self, max_size: usize, max_lbd: u32) -> bool {
        self.lits.len() <= max_size && self.lbd <= max_lbd
    }

    /// Get the size of the clause
    #[must_use]
    pub fn size(&self) -> usize {
        self.lits.len()
    }
}

/// Statistics for clause exchange
#[derive(Debug, Clone, Default)]
pub struct ExchangeStats {
    /// Number of clauses exported
    pub clauses_exported: usize,
    /// Number of clauses imported
    pub clauses_imported: usize,
    /// Number of duplicate clauses rejected
    pub duplicates_rejected: usize,
    /// Number of too-large clauses rejected
    pub size_rejected: usize,
    /// Number of low-quality clauses rejected
    pub quality_rejected: usize,
    /// Total clauses in exchange buffer
    pub buffer_size: usize,
}

impl ExchangeStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Clause Exchange Statistics:");
        println!("  Exported: {}", self.clauses_exported);
        println!("  Imported: {}", self.clauses_imported);
        println!("  Rejected:");
        println!("    Duplicates: {}", self.duplicates_rejected);
        println!("    Too large:  {}", self.size_rejected);
        println!("    Low quality: {}", self.quality_rejected);
        println!("  Buffer size: {}", self.buffer_size);
    }
}

/// Configuration for clause exchange
#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    /// Maximum clause size to share
    pub max_clause_size: usize,
    /// Maximum LBD to share
    pub max_lbd: u32,
    /// Maximum buffer size per solver
    pub max_buffer_size: usize,
    /// Export only clauses with activity above this threshold
    pub min_activity: f64,
}

impl Default for ExchangeConfig {
    fn default() -> Self {
        Self {
            max_clause_size: 8,    // Share only small clauses
            max_lbd: 4,            // Share only high-quality clauses
            max_buffer_size: 1000, // Limit memory usage
            min_activity: 0.0,     // No activity threshold by default
        }
    }
}

/// Clause exchange buffer shared between solvers
#[derive(Debug)]
pub struct ClauseExchangeBuffer {
    /// Shared buffer of clauses
    buffer: Arc<Mutex<Vec<SharedClause>>>,
    /// Configuration
    config: ExchangeConfig,
    /// Statistics
    stats: ExchangeStats,
    /// Solver ID
    solver_id: usize,
}

impl ClauseExchangeBuffer {
    /// Create a new exchange buffer
    #[must_use]
    pub fn new(solver_id: usize, config: ExchangeConfig) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::new())),
            config,
            stats: ExchangeStats::default(),
            solver_id,
        }
    }

    /// Create with shared buffer (for multiple solvers)
    #[must_use]
    pub fn with_shared_buffer(
        solver_id: usize,
        config: ExchangeConfig,
        buffer: Arc<Mutex<Vec<SharedClause>>>,
    ) -> Self {
        Self {
            buffer,
            config,
            stats: ExchangeStats::default(),
            solver_id,
        }
    }

    /// Get the shared buffer for creating additional exchange instances
    #[must_use]
    pub fn get_shared_buffer(&self) -> Arc<Mutex<Vec<SharedClause>>> {
        Arc::clone(&self.buffer)
    }

    /// Export a clause to the shared buffer
    ///
    /// Returns true if the clause was successfully exported
    pub fn export_clause(&mut self, clause: &SharedClause) -> bool {
        // Check if clause meets quality criteria
        if !clause.is_shareable(self.config.max_clause_size, self.config.max_lbd) {
            if clause.size() > self.config.max_clause_size {
                self.stats.size_rejected += 1;
            } else {
                self.stats.quality_rejected += 1;
            }
            return false;
        }

        // Check activity threshold
        if clause.activity < self.config.min_activity {
            self.stats.quality_rejected += 1;
            return false;
        }

        // Add to shared buffer
        if let Ok(mut buffer) = self.buffer.lock() {
            // Limit buffer size
            if buffer.len() >= self.config.max_buffer_size {
                // Remove lowest quality clause
                if let Some(min_idx) = buffer
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.activity
                            .partial_cmp(&b.activity)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                {
                    buffer.swap_remove(min_idx);
                }
            }

            buffer.push(clause.clone());
            self.stats.clauses_exported += 1;
            self.stats.buffer_size = buffer.len();
            true
        } else {
            false
        }
    }

    /// Import clauses from other solvers
    ///
    /// Returns clauses that haven't been seen by this solver
    pub fn import_clauses(&mut self) -> Vec<SharedClause> {
        let mut imported = Vec::new();

        if let Ok(buffer) = self.buffer.lock() {
            // Get clauses from other solvers
            let clauses: Vec<_> = buffer
                .iter()
                .filter(|c| c.source_id != self.solver_id)
                .cloned()
                .collect();

            for clause in clauses {
                imported.push(clause);
                self.stats.clauses_imported += 1;
            }

            self.stats.buffer_size = buffer.len();
        }

        imported
    }

    /// Clear imported clauses that have been processed
    pub fn clear_processed(&mut self) {
        if let Ok(mut buffer) = self.buffer.lock() {
            // Keep only recent high-quality clauses
            buffer.retain(|c| c.lbd <= 2 || c.activity > 0.5);
            buffer.sort_by(|a, b| {
                b.activity
                    .partial_cmp(&a.activity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            buffer.truncate(self.config.max_buffer_size / 2);
            self.stats.buffer_size = buffer.len();
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &ExchangeStats {
        &self.stats
    }

    /// Get current buffer size
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer.lock().map_or(0, |b| b.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_shared_clause_creation() {
        let lits = SmallVec::from_vec(vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))]);
        let clause = SharedClause::new(lits, 2, 0.5, 0);

        assert_eq!(clause.size(), 2);
        assert_eq!(clause.lbd, 2);
        assert_eq!(clause.source_id, 0);
    }

    #[test]
    fn test_is_shareable() {
        let lits = SmallVec::from_vec(vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))]);
        let clause = SharedClause::new(lits, 2, 0.5, 0);

        assert!(clause.is_shareable(8, 4));
        assert!(!clause.is_shareable(1, 4)); // Too large
        assert!(!clause.is_shareable(8, 1)); // LBD too high
    }

    #[test]
    fn test_exchange_buffer() {
        let config = ExchangeConfig::default();
        let mut buffer = ClauseExchangeBuffer::new(0, config);

        let lits = SmallVec::from_vec(vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))]);
        let clause = SharedClause::new(lits, 2, 0.5, 0);

        assert!(buffer.export_clause(&clause));
        assert_eq!(buffer.stats().clauses_exported, 1);
    }

    #[test]
    fn test_size_rejection() {
        let config = ExchangeConfig {
            max_clause_size: 2,
            ..Default::default()
        };
        let mut buffer = ClauseExchangeBuffer::new(0, config);

        let lits = SmallVec::from_vec(vec![
            Lit::pos(Var::new(0)),
            Lit::neg(Var::new(1)),
            Lit::pos(Var::new(2)),
        ]);
        let clause = SharedClause::new(lits, 2, 0.5, 0);

        assert!(!buffer.export_clause(&clause));
        assert_eq!(buffer.stats().size_rejected, 1);
    }

    #[test]
    fn test_lbd_rejection() {
        let config = ExchangeConfig {
            max_lbd: 2,
            ..Default::default()
        };
        let mut buffer = ClauseExchangeBuffer::new(0, config);

        let lits = SmallVec::from_vec(vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))]);
        let clause = SharedClause::new(lits, 5, 0.5, 0);

        assert!(!buffer.export_clause(&clause));
        assert_eq!(buffer.stats().quality_rejected, 1);
    }

    #[test]
    fn test_import_from_other_solver() {
        let config = ExchangeConfig::default();
        let shared = Arc::new(Mutex::new(Vec::new()));

        let mut buffer1 =
            ClauseExchangeBuffer::with_shared_buffer(0, config.clone(), Arc::clone(&shared));
        let mut buffer2 = ClauseExchangeBuffer::with_shared_buffer(1, config, Arc::clone(&shared));

        // Solver 0 exports a clause
        let lits = SmallVec::from_vec(vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))]);
        let clause = SharedClause::new(lits, 2, 0.5, 0);
        buffer1.export_clause(&clause);

        // Solver 1 imports clauses
        let imported = buffer2.import_clauses();
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].source_id, 0);
        assert_eq!(buffer2.stats().clauses_imported, 1);
    }

    #[test]
    fn test_buffer_size_limit() {
        let config = ExchangeConfig {
            max_buffer_size: 3,
            ..Default::default()
        };
        let mut buffer = ClauseExchangeBuffer::new(0, config);

        // Add 4 clauses (should only keep 3)
        for i in 0..4 {
            let lits = SmallVec::from_vec(vec![Lit::pos(Var::new(i))]);
            let clause = SharedClause::new(lits, 2, i as f64, 0);
            buffer.export_clause(&clause);
        }

        assert!(buffer.buffer_size() <= 3);
    }
}
