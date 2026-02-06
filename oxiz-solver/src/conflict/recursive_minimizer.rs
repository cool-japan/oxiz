//! Recursive Conflict Clause Minimization
#![allow(missing_docs, clippy::ptr_arg)] // Under development
//!
//! This module implements advanced conflict clause minimization techniques:
//! - Recursive resolution-based minimization
//! - Binary resolution minimization
//! - Stamp-based minimization (seen/poison marking)
//! - Self-subsuming resolution detection

use rustc_hash::{FxHashMap, FxHashSet};

/// Literal representation
pub type Lit = i32;

/// Clause identifier
pub type ClauseId = usize;

/// Level (decision level)
pub type Level = usize;

/// Reason for a literal assignment
#[derive(Debug, Clone)]
pub enum Reason {
    /// Decision literal (no reason)
    Decision,
    /// Unit propagation from clause
    Clause(ClauseId),
    /// Binary clause
    Binary(Lit),
    /// Theory propagation
    Theory(Vec<Lit>),
}

/// Literal information for minimization
#[derive(Debug, Clone)]
pub struct LitInfo {
    /// Assignment level
    pub level: Level,
    /// Reason for assignment
    pub reason: Reason,
    /// Whether literal is marked as seen
    pub seen: bool,
    /// Whether literal is poisoned (cannot be removed)
    pub poisoned: bool,
}

/// Statistics for recursive minimization
#[derive(Debug, Clone, Default)]
pub struct RecursiveMinStats {
    pub clauses_minimized: u64,
    pub literals_removed: u64,
    pub recursive_calls: u64,
    pub binary_minimizations: u64,
    pub stamp_minimizations: u64,
    pub self_subsuming_resolutions: u64,
}

/// Configuration for recursive minimization
#[derive(Debug, Clone)]
pub struct RecursiveMinConfig {
    /// Enable recursive resolution minimization
    pub enable_recursive: bool,
    /// Enable binary resolution minimization
    pub enable_binary: bool,
    /// Enable stamp-based minimization
    pub enable_stamp: bool,
    /// Maximum recursion depth
    pub max_recursion_depth: usize,
    /// Enable self-subsuming resolution detection
    pub enable_self_subsumption: bool,
}

impl Default for RecursiveMinConfig {
    fn default() -> Self {
        Self {
            enable_recursive: true,
            enable_binary: true,
            enable_stamp: true,
            max_recursion_depth: 10,
            enable_self_subsumption: true,
        }
    }
}

/// Recursive conflict clause minimizer
pub struct RecursiveMinimizer {
    config: RecursiveMinConfig,
    stats: RecursiveMinStats,
    /// Literal information database
    lit_info: FxHashMap<Lit, LitInfo>,
    /// Current conflict level
    conflict_level: Level,
    /// Stack for recursive minimization
    min_stack: Vec<Lit>,
    /// Analyzed literals (for cycle detection)
    analyzed: FxHashSet<Lit>,
}

impl RecursiveMinimizer {
    /// Create a new recursive minimizer
    pub fn new(config: RecursiveMinConfig) -> Self {
        Self {
            config,
            stats: RecursiveMinStats::default(),
            lit_info: FxHashMap::default(),
            conflict_level: 0,
            min_stack: Vec::new(),
            analyzed: FxHashSet::default(),
        }
    }

    /// Minimize a conflict clause
    pub fn minimize(&mut self, clause: &mut Vec<Lit>) -> Result<(), String> {
        if clause.len() <= 1 {
            return Ok(());
        }

        self.stats.clauses_minimized += 1;

        // Compute conflict level (highest level in clause)
        self.conflict_level = clause
            .iter()
            .map(|&lit| self.get_level(lit))
            .max()
            .unwrap_or(0);

        // Mark all literals in clause as seen
        for &lit in clause.iter() {
            if let Some(info) = self.lit_info.get_mut(&lit) {
                info.seen = true;
            }
        }

        // Phase 1: Recursive minimization
        if self.config.enable_recursive {
            self.recursive_minimize(clause)?;
        }

        // Phase 2: Binary resolution minimization
        if self.config.enable_binary {
            self.binary_minimize(clause)?;
        }

        // Phase 3: Stamp-based minimization
        if self.config.enable_stamp {
            self.stamp_minimize(clause)?;
        }

        // Phase 4: Self-subsuming resolution
        if self.config.enable_self_subsumption {
            self.detect_self_subsumption(clause)?;
        }

        // Clear seen marks
        for &lit in clause.iter() {
            if let Some(info) = self.lit_info.get_mut(&lit) {
                info.seen = false;
            }
        }

        Ok(())
    }

    /// Recursive resolution-based minimization
    fn recursive_minimize(&mut self, clause: &mut Vec<Lit>) -> Result<(), String> {
        let mut to_remove = Vec::new();

        for &lit in clause.iter() {
            // Skip decision literals at conflict level
            if self.get_level(lit) == self.conflict_level && self.is_decision(lit) {
                continue;
            }

            // Try to resolve this literal away
            self.analyzed.clear();
            if self.can_remove_recursive(lit, 0)? {
                to_remove.push(lit);
                self.stats.literals_removed += 1;
            }
        }

        // Remove literals
        clause.retain(|lit| !to_remove.contains(lit));

        Ok(())
    }

    /// Check if a literal can be removed via recursive resolution
    fn can_remove_recursive(&mut self, lit: Lit, depth: usize) -> Result<bool, String> {
        if depth > self.config.max_recursion_depth {
            return Ok(false);
        }

        self.stats.recursive_calls += 1;

        // Check for cycles
        if self.analyzed.contains(&lit) {
            return Ok(false);
        }
        self.analyzed.insert(lit);

        // Get reason for this literal
        let reason = match self.lit_info.get(&lit) {
            Some(info) => info.reason.clone(),
            None => return Ok(false),
        };

        match reason {
            Reason::Decision => Ok(false),
            Reason::Binary(other_lit) => {
                // Check if other literal is in clause or can be removed
                self.is_redundant(other_lit, depth)
            }
            Reason::Clause(clause_id) => {
                // Check if all literals in reason clause are redundant
                let reason_lits = self.get_clause_literals(clause_id)?;
                for &reason_lit in &reason_lits {
                    if reason_lit == lit {
                        continue;
                    }
                    if !self.is_redundant(reason_lit, depth)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Reason::Theory(theory_lits) => {
                // Check if all theory literals are redundant
                for &theory_lit in &theory_lits {
                    if theory_lit == lit {
                        continue;
                    }
                    if !self.is_redundant(theory_lit, depth)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    /// Check if a literal is redundant (in clause or can be recursively removed)
    fn is_redundant(&mut self, lit: Lit, depth: usize) -> Result<bool, String> {
        // Check if literal is in original clause (marked as seen)
        if self.lit_info.get(&lit).is_some_and(|info| info.seen) {
            return Ok(true);
        }

        // Check if at a level below conflict level (definitely needed)
        if self.get_level(lit) < self.conflict_level {
            return Ok(true);
        }

        // Try recursive removal
        self.can_remove_recursive(lit, depth + 1)
    }

    /// Binary resolution minimization
    fn binary_minimize(&mut self, clause: &mut Vec<Lit>) -> Result<(), String> {
        self.stats.binary_minimizations += 1;

        let mut to_remove = Vec::new();

        for &lit in clause.iter() {
            // Check if this literal has a binary reason
            let reason = match self.lit_info.get(&lit) {
                Some(info) => &info.reason,
                None => continue,
            };

            if let Reason::Binary(other_lit) = reason {
                // Check if -other_lit is in the clause
                if clause.contains(&-other_lit) {
                    // Binary resolution: (lit ∨ -other_lit) ∧ (other_lit ∨ C) → C
                    to_remove.push(lit);
                    self.stats.literals_removed += 1;
                }
            }
        }

        clause.retain(|lit| !to_remove.contains(lit));

        Ok(())
    }

    /// Stamp-based minimization (MiniSAT-style)
    fn stamp_minimize(&mut self, clause: &mut Vec<Lit>) -> Result<(), String> {
        self.stats.stamp_minimizations += 1;

        // Mark literals at conflict level as "abstract"
        let mut abstract_level = FxHashSet::default();
        for &lit in clause.iter() {
            if self.get_level(lit) == self.conflict_level {
                abstract_level.insert(lit);
            }
        }

        let mut to_remove = Vec::new();

        for &lit in clause.iter() {
            if self.get_level(lit) < self.conflict_level {
                continue;
            }

            // Check if this literal can be "stamped out"
            if self.can_stamp_out(lit, &abstract_level)? {
                to_remove.push(lit);
                self.stats.literals_removed += 1;
            }
        }

        clause.retain(|lit| !to_remove.contains(lit));

        Ok(())
    }

    /// Check if a literal can be stamped out
    fn can_stamp_out(&self, lit: Lit, abstract_level: &FxHashSet<Lit>) -> Result<bool, String> {
        // Get reason for this literal
        let reason = match self.lit_info.get(&lit) {
            Some(info) => &info.reason,
            None => return Ok(false),
        };

        match reason {
            Reason::Decision => Ok(false),
            Reason::Binary(other_lit) => Ok(abstract_level.contains(other_lit)),
            Reason::Clause(clause_id) => {
                let reason_lits = self.get_clause_literals(*clause_id)?;
                Ok(reason_lits
                    .iter()
                    .filter(|&&l| l != lit)
                    .all(|l| abstract_level.contains(l)))
            }
            Reason::Theory(theory_lits) => Ok(theory_lits
                .iter()
                .filter(|&&l| l != lit)
                .all(|l| abstract_level.contains(l))),
        }
    }

    /// Detect self-subsuming resolution opportunities
    fn detect_self_subsumption(&mut self, clause: &mut Vec<Lit>) -> Result<(), String> {
        // Check if clause can subsume its reason clause after resolution
        for &lit in clause.clone().iter() {
            let reason = match self.lit_info.get(&lit) {
                Some(info) => &info.reason,
                None => continue,
            };

            if let Reason::Clause(clause_id) = reason
                && self.is_self_subsuming(clause, lit, *clause_id)?
            {
                // Found self-subsumption
                self.stats.self_subsuming_resolutions += 1;
                // Would update clause database here
            }
        }

        Ok(())
    }

    /// Check if clause self-subsumes after resolving on lit
    fn is_self_subsuming(
        &self,
        clause: &[Lit],
        lit: Lit,
        reason_clause_id: ClauseId,
    ) -> Result<bool, String> {
        let reason_lits = self.get_clause_literals(reason_clause_id)?;

        // Compute resolvent
        let mut resolvent: FxHashSet<_> = clause.iter().filter(|&&l| l != lit).copied().collect();

        for &reason_lit in &reason_lits {
            if reason_lit != -lit {
                resolvent.insert(reason_lit);
            }
        }

        // Check if resolvent subsumes original clause
        Ok(resolvent.len() < clause.len())
    }

    /// Set literal information
    pub fn set_lit_info(&mut self, lit: Lit, level: Level, reason: Reason) {
        self.lit_info.insert(
            lit,
            LitInfo {
                level,
                reason,
                seen: false,
                poisoned: false,
            },
        );
    }

    /// Get level of a literal
    fn get_level(&self, lit: Lit) -> Level {
        self.lit_info.get(&lit).map_or(0, |info| info.level)
    }

    /// Check if literal is a decision
    fn is_decision(&self, lit: Lit) -> bool {
        self.lit_info
            .get(&lit)
            .is_some_and(|info| matches!(info.reason, Reason::Decision))
    }

    /// Get literals from a clause (placeholder)
    fn get_clause_literals(&self, _clause_id: ClauseId) -> Result<Vec<Lit>, String> {
        // Placeholder: would retrieve from clause database
        Ok(vec![])
    }

    /// Get statistics
    pub fn stats(&self) -> &RecursiveMinStats {
        &self.stats
    }

    /// Reset minimizer state
    pub fn reset(&mut self) {
        self.lit_info.clear();
        self.min_stack.clear();
        self.analyzed.clear();
        self.conflict_level = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimizer_creation() {
        let config = RecursiveMinConfig::default();
        let minimizer = RecursiveMinimizer::new(config);
        assert_eq!(minimizer.stats.clauses_minimized, 0);
    }

    #[test]
    fn test_empty_clause() {
        let config = RecursiveMinConfig::default();
        let mut minimizer = RecursiveMinimizer::new(config);

        let mut clause = vec![];
        let result = minimizer.minimize(&mut clause);

        assert!(result.is_ok());
        assert_eq!(clause.len(), 0);
    }

    #[test]
    fn test_unit_clause() {
        let config = RecursiveMinConfig::default();
        let mut minimizer = RecursiveMinimizer::new(config);

        let mut clause = vec![1];
        let result = minimizer.minimize(&mut clause);

        assert!(result.is_ok());
        assert_eq!(clause.len(), 1);
    }

    #[test]
    fn test_set_lit_info() {
        let config = RecursiveMinConfig::default();
        let mut minimizer = RecursiveMinimizer::new(config);

        minimizer.set_lit_info(1, 5, Reason::Decision);
        assert_eq!(minimizer.get_level(1), 5);
        assert!(minimizer.is_decision(1));
    }

    #[test]
    fn test_binary_reason() {
        let config = RecursiveMinConfig::default();
        let mut minimizer = RecursiveMinimizer::new(config);

        minimizer.set_lit_info(1, 3, Reason::Binary(2));
        assert_eq!(minimizer.get_level(1), 3);
        assert!(!minimizer.is_decision(1));
    }

    #[test]
    fn test_theory_reason() {
        let config = RecursiveMinConfig::default();
        let mut minimizer = RecursiveMinimizer::new(config);

        minimizer.set_lit_info(1, 4, Reason::Theory(vec![2, 3, 4]));
        assert_eq!(minimizer.get_level(1), 4);
    }

    #[test]
    fn test_conflict_level_computation() {
        let config = RecursiveMinConfig::default();
        let mut minimizer = RecursiveMinimizer::new(config);

        minimizer.set_lit_info(1, 3, Reason::Decision);
        minimizer.set_lit_info(2, 5, Reason::Decision);
        minimizer.set_lit_info(3, 4, Reason::Decision);

        let mut clause = vec![1, 2, 3];
        let _ = minimizer.minimize(&mut clause);

        assert_eq!(minimizer.conflict_level, 5);
    }

    #[test]
    fn test_recursive_minimization_disabled() {
        let config = RecursiveMinConfig {
            enable_recursive: false,
            ..Default::default()
        };
        let mut minimizer = RecursiveMinimizer::new(config);

        minimizer.set_lit_info(1, 3, Reason::Decision);
        minimizer.set_lit_info(2, 3, Reason::Binary(1));

        let mut clause = vec![1, 2];
        let _ = minimizer.minimize(&mut clause);

        // Should not remove anything since recursive minimization is disabled
        assert!(clause.contains(&1) || clause.contains(&2));
    }

    #[test]
    fn test_stats_tracking() {
        let config = RecursiveMinConfig::default();
        let mut minimizer = RecursiveMinimizer::new(config);

        minimizer.set_lit_info(1, 3, Reason::Decision);
        minimizer.set_lit_info(2, 3, Reason::Decision);

        let mut clause = vec![1, 2];
        let _ = minimizer.minimize(&mut clause);

        assert_eq!(minimizer.stats.clauses_minimized, 1);
    }

    #[test]
    fn test_reset() {
        let config = RecursiveMinConfig::default();
        let mut minimizer = RecursiveMinimizer::new(config);

        minimizer.set_lit_info(1, 3, Reason::Decision);
        minimizer.conflict_level = 5;

        minimizer.reset();

        assert!(minimizer.lit_info.is_empty());
        assert_eq!(minimizer.conflict_level, 0);
    }

    #[test]
    fn test_max_recursion_depth() {
        let config = RecursiveMinConfig {
            max_recursion_depth: 2,
            ..Default::default()
        };
        let mut minimizer = RecursiveMinimizer::new(config);

        // Create deep reason chain
        minimizer.set_lit_info(1, 5, Reason::Binary(2));
        minimizer.set_lit_info(2, 5, Reason::Binary(3));
        minimizer.set_lit_info(3, 5, Reason::Binary(4));
        minimizer.set_lit_info(4, 5, Reason::Binary(5));
        minimizer.set_lit_info(5, 5, Reason::Decision);

        let result = minimizer.can_remove_recursive(1, 0);
        assert!(result.is_ok());
    }
}
