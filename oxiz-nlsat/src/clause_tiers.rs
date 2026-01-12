//! Clause Tier System
//!
//! This module implements a multi-tier clause management system inspired by
//! modern SAT solvers. Clauses are categorized into tiers based on their quality:
//!
//! - **Core**: Essential clauses (input clauses, very low LBD)
//! - **Tier1**: High-quality learned clauses (low LBD, high usage)
//! - **Tier2**: Medium-quality learned clauses
//! - **Local**: Low-quality clauses (recent but not proven useful)
//!
//! Reference: CaDiCaL and Glucose tier-based clause management

use crate::clause::{Clause, ClauseId};
use std::collections::{HashMap, VecDeque};

/// Clause tier/category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClauseTier {
    /// Core clauses: original input clauses and glue clauses (LBD ≤ 2)
    Core,
    /// Tier1: High-quality learned clauses (LBD ≤ 5, frequently used)
    Tier1,
    /// Tier2: Medium-quality learned clauses (LBD ≤ 10)
    Tier2,
    /// Local: Low-quality or recently learned clauses
    Local,
}

impl ClauseTier {
    /// Get the priority of a tier (higher = more important)
    pub fn priority(&self) -> u8 {
        match self {
            ClauseTier::Core => 3,
            ClauseTier::Tier1 => 2,
            ClauseTier::Tier2 => 1,
            ClauseTier::Local => 0,
        }
    }

    /// Check if this tier is protected from deletion
    pub fn is_protected(&self) -> bool {
        matches!(self, ClauseTier::Core | ClauseTier::Tier1)
    }
}

/// Configuration for clause tier system
#[derive(Debug, Clone)]
pub struct ClauseTierConfig {
    /// Enable tier system
    pub enabled: bool,
    /// LBD threshold for Core tier
    pub core_lbd_threshold: u32,
    /// LBD threshold for Tier1
    pub tier1_lbd_threshold: u32,
    /// LBD threshold for Tier2
    pub tier2_lbd_threshold: u32,
    /// Maximum number of clauses in Local tier before cleanup
    pub local_max_size: usize,
    /// Activity threshold for promoting to Tier1
    pub tier1_activity_threshold: f64,
}

impl Default for ClauseTierConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            core_lbd_threshold: 2,
            tier1_lbd_threshold: 5,
            tier2_lbd_threshold: 10,
            local_max_size: 10000,
            tier1_activity_threshold: 0.5,
        }
    }
}

/// Statistics for clause tier system
#[derive(Debug, Default, Clone)]
pub struct ClauseTierStats {
    /// Number of clauses in each tier
    pub core_clauses: usize,
    pub tier1_clauses: usize,
    pub tier2_clauses: usize,
    pub local_clauses: usize,
    /// Number of promotions
    pub promotions: usize,
    /// Number of demotions
    pub demotions: usize,
    /// Number of clauses deleted from Local
    pub local_deleted: usize,
}

impl ClauseTierStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of clauses
    pub fn total_clauses(&self) -> usize {
        self.core_clauses + self.tier1_clauses + self.tier2_clauses + self.local_clauses
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Clause tier manager
pub struct ClauseTierManager {
    config: ClauseTierConfig,
    stats: ClauseTierStats,
    /// Mapping from clause ID to tier
    clause_tiers: HashMap<ClauseId, ClauseTier>,
    /// Queue of recently added local clauses (for aging)
    local_queue: VecDeque<ClauseId>,
}

impl ClauseTierManager {
    /// Create a new clause tier manager
    pub fn new(config: ClauseTierConfig) -> Self {
        Self {
            config,
            stats: ClauseTierStats::new(),
            clause_tiers: HashMap::new(),
            local_queue: VecDeque::new(),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &ClauseTierStats {
        &self.stats
    }

    /// Get the tier of a clause
    pub fn get_tier(&self, clause_id: ClauseId) -> Option<ClauseTier> {
        self.clause_tiers.get(&clause_id).copied()
    }

    /// Determine the initial tier for a clause
    pub fn classify_clause(&self, clause: &Clause) -> ClauseTier {
        if !self.config.enabled {
            return ClauseTier::Local;
        }

        // Input clauses (non-learned) go to Core
        if !clause.is_learned() {
            return ClauseTier::Core;
        }

        // Classify learned clauses based on LBD
        let lbd = clause.lbd();

        if lbd <= self.config.core_lbd_threshold {
            ClauseTier::Core
        } else if lbd <= self.config.tier1_lbd_threshold {
            ClauseTier::Tier1
        } else if lbd <= self.config.tier2_lbd_threshold {
            ClauseTier::Tier2
        } else {
            ClauseTier::Local
        }
    }

    /// Add a clause to the tier system
    pub fn add_clause(&mut self, clause: &Clause) {
        let tier = self.classify_clause(clause);
        let clause_id = clause.id();

        self.clause_tiers.insert(clause_id, tier);

        match tier {
            ClauseTier::Core => self.stats.core_clauses += 1,
            ClauseTier::Tier1 => self.stats.tier1_clauses += 1,
            ClauseTier::Tier2 => self.stats.tier2_clauses += 1,
            ClauseTier::Local => {
                self.stats.local_clauses += 1;
                self.local_queue.push_back(clause_id);
            }
        }
    }

    /// Remove a clause from the tier system
    pub fn remove_clause(&mut self, clause_id: ClauseId) {
        if let Some(tier) = self.clause_tiers.remove(&clause_id) {
            match tier {
                ClauseTier::Core => {
                    self.stats.core_clauses = self.stats.core_clauses.saturating_sub(1)
                }
                ClauseTier::Tier1 => {
                    self.stats.tier1_clauses = self.stats.tier1_clauses.saturating_sub(1)
                }
                ClauseTier::Tier2 => {
                    self.stats.tier2_clauses = self.stats.tier2_clauses.saturating_sub(1)
                }
                ClauseTier::Local => {
                    self.stats.local_clauses = self.stats.local_clauses.saturating_sub(1)
                }
            }
        }
    }

    /// Update clause tier based on activity and LBD
    pub fn update_clause_tier(&mut self, clause: &Clause) {
        if !self.config.enabled {
            return;
        }

        let clause_id = clause.id();
        let current_tier = match self.clause_tiers.get(&clause_id) {
            Some(&tier) => tier,
            None => return, // Clause not tracked
        };

        let new_tier = self.classify_clause(clause);

        // Check for promotion based on activity
        if current_tier == ClauseTier::Tier2
            && clause.activity() > self.config.tier1_activity_threshold
        {
            self.promote_clause(clause_id, ClauseTier::Tier1);
        } else if current_tier != new_tier {
            // Update tier based on LBD changes
            if new_tier.priority() > current_tier.priority() {
                self.promote_clause(clause_id, new_tier);
            } else if new_tier.priority() < current_tier.priority() {
                self.demote_clause(clause_id, new_tier);
            }
        }
    }

    /// Promote a clause to a higher tier
    fn promote_clause(&mut self, clause_id: ClauseId, new_tier: ClauseTier) {
        if let Some(old_tier) = self.clause_tiers.get_mut(&clause_id) {
            match *old_tier {
                ClauseTier::Core => return, // Can't promote from Core
                ClauseTier::Tier1 => self.stats.tier1_clauses -= 1,
                ClauseTier::Tier2 => self.stats.tier2_clauses -= 1,
                ClauseTier::Local => self.stats.local_clauses -= 1,
            }

            *old_tier = new_tier;
            self.stats.promotions += 1;

            match new_tier {
                ClauseTier::Core => self.stats.core_clauses += 1,
                ClauseTier::Tier1 => self.stats.tier1_clauses += 1,
                ClauseTier::Tier2 => self.stats.tier2_clauses += 1,
                ClauseTier::Local => self.stats.local_clauses += 1,
            }
        }
    }

    /// Demote a clause to a lower tier
    fn demote_clause(&mut self, clause_id: ClauseId, new_tier: ClauseTier) {
        if let Some(old_tier) = self.clause_tiers.get_mut(&clause_id) {
            match *old_tier {
                ClauseTier::Core => self.stats.core_clauses -= 1,
                ClauseTier::Tier1 => self.stats.tier1_clauses -= 1,
                ClauseTier::Tier2 => self.stats.tier2_clauses -= 1,
                ClauseTier::Local => return, // Can't demote from Local
            }

            *old_tier = new_tier;
            self.stats.demotions += 1;

            match new_tier {
                ClauseTier::Core => self.stats.core_clauses += 1,
                ClauseTier::Tier1 => self.stats.tier1_clauses += 1,
                ClauseTier::Tier2 => self.stats.tier2_clauses += 1,
                ClauseTier::Local => {
                    self.stats.local_clauses += 1;
                    self.local_queue.push_back(clause_id);
                }
            }
        }
    }

    /// Select clauses for deletion from Local tier
    /// Returns clause IDs that should be deleted
    pub fn select_for_deletion(&mut self, clauses: &HashMap<ClauseId, Clause>) -> Vec<ClauseId> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut to_delete = Vec::new();

        // Delete old clauses from Local if it's too large
        while self.stats.local_clauses > self.config.local_max_size {
            if let Some(clause_id) = self.local_queue.pop_front() {
                // Check if clause still exists and is in Local tier
                if let Some(&tier) = self.clause_tiers.get(&clause_id)
                    && tier == ClauseTier::Local
                    && clauses.contains_key(&clause_id)
                {
                    to_delete.push(clause_id);
                    self.remove_clause(clause_id);
                    self.stats.local_deleted += 1;
                }
            } else {
                break;
            }
        }

        to_delete
    }

    /// Check if a clause is protected from deletion
    pub fn is_protected(&self, clause_id: ClauseId) -> bool {
        self.clause_tiers
            .get(&clause_id)
            .is_some_and(|tier| tier.is_protected())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Literal;

    fn make_clause(lbd: u32, learned: bool, id: ClauseId) -> Clause {
        let mut clause = Clause::new(
            vec![Literal::positive(1), Literal::positive(2)],
            2,
            learned,
            id,
        );
        clause.set_lbd(lbd);
        clause
    }

    #[test]
    fn test_clause_tier_priority() {
        assert!(ClauseTier::Core.priority() > ClauseTier::Tier1.priority());
        assert!(ClauseTier::Tier1.priority() > ClauseTier::Tier2.priority());
        assert!(ClauseTier::Tier2.priority() > ClauseTier::Local.priority());
    }

    #[test]
    fn test_tier_manager_new() {
        let config = ClauseTierConfig::default();
        let manager = ClauseTierManager::new(config);
        assert_eq!(manager.stats().total_clauses(), 0);
    }

    #[test]
    fn test_classify_input_clause() {
        let config = ClauseTierConfig::default();
        let manager = ClauseTierManager::new(config);

        let clause = make_clause(10, false, 0);
        assert_eq!(manager.classify_clause(&clause), ClauseTier::Core);
    }

    #[test]
    fn test_classify_learned_clause() {
        let config = ClauseTierConfig::default();
        let manager = ClauseTierManager::new(config);

        let core_clause = make_clause(2, true, 0);
        assert_eq!(manager.classify_clause(&core_clause), ClauseTier::Core);

        let tier1_clause = make_clause(5, true, 1);
        assert_eq!(manager.classify_clause(&tier1_clause), ClauseTier::Tier1);

        let tier2_clause = make_clause(10, true, 2);
        assert_eq!(manager.classify_clause(&tier2_clause), ClauseTier::Tier2);

        let local_clause = make_clause(20, true, 3);
        assert_eq!(manager.classify_clause(&local_clause), ClauseTier::Local);
    }

    #[test]
    fn test_add_remove_clause() {
        let config = ClauseTierConfig::default();
        let mut manager = ClauseTierManager::new(config);

        let clause = make_clause(5, true, 0);
        manager.add_clause(&clause);

        assert_eq!(manager.stats().tier1_clauses, 1);
        assert_eq!(manager.get_tier(0), Some(ClauseTier::Tier1));

        manager.remove_clause(0);
        assert_eq!(manager.stats().tier1_clauses, 0);
        assert_eq!(manager.get_tier(0), None);
    }

    #[test]
    fn test_tier_protected() {
        assert!(ClauseTier::Core.is_protected());
        assert!(ClauseTier::Tier1.is_protected());
        assert!(!ClauseTier::Tier2.is_protected());
        assert!(!ClauseTier::Local.is_protected());
    }

    #[test]
    fn test_local_queue_deletion() {
        let config = ClauseTierConfig {
            local_max_size: 2,
            ..Default::default()
        };
        let mut manager = ClauseTierManager::new(config);

        // Add 3 local clauses
        let c1 = make_clause(20, true, 0);
        let c2 = make_clause(20, true, 1);
        let c3 = make_clause(20, true, 2);

        manager.add_clause(&c1);
        manager.add_clause(&c2);
        manager.add_clause(&c3);

        let mut clauses = HashMap::new();
        clauses.insert(0, c1);
        clauses.insert(1, c2);
        clauses.insert(2, c3);

        // Should delete oldest clause (id 0)
        let to_delete = manager.select_for_deletion(&clauses);
        assert_eq!(to_delete.len(), 1);
        assert_eq!(to_delete[0], 0);
    }

    #[test]
    fn test_update_tier_promotion() {
        let config = ClauseTierConfig::default();
        let mut manager = ClauseTierManager::new(config);

        let mut clause = make_clause(10, true, 0);
        manager.add_clause(&clause);

        assert_eq!(manager.get_tier(0), Some(ClauseTier::Tier2));

        // Improve LBD
        clause.set_lbd(3);
        manager.update_clause_tier(&clause);

        assert_eq!(manager.get_tier(0), Some(ClauseTier::Tier1));
        assert_eq!(manager.stats().promotions, 1);
    }
}
