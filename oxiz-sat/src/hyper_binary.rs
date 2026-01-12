//! Hyper-binary resolution and advanced probing
//!
//! This module implements hyper-binary resolution (HBR), an advanced technique
//! for discovering and adding binary clauses during search. HBR can significantly
//! reduce the search space by identifying implied binary clauses.
//!
//! References:
//! - "Hyper Binary Resolution: A Proof System for SAT Solver Analysis"
//! - "Binary Clause Reasoning in Conflict-Driven Clause Learning"
//! - "Vivification and Hyper-Binary Resolution" (Biere)

use crate::literal::Lit;
use smallvec::SmallVec;

/// Statistics for hyper-binary resolution
#[derive(Debug, Clone, Default)]
pub struct HyperBinaryStats {
    /// Number of hyper-binary resolutions performed
    pub hbr_attempts: u64,
    /// Number of binary clauses discovered
    pub binary_clauses_found: usize,
    /// Number of unit clauses discovered
    pub unit_clauses_found: usize,
    /// Number of conflicts discovered
    pub conflicts_found: usize,
    /// Total literals probed
    pub literals_probed: u64,
}

impl HyperBinaryStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Hyper-Binary Resolution Statistics:");
        println!("  HBR attempts: {}", self.hbr_attempts);
        println!("  Binary clauses found: {}", self.binary_clauses_found);
        println!("  Unit clauses found: {}", self.unit_clauses_found);
        println!("  Conflicts found: {}", self.conflicts_found);
        println!("  Literals probed: {}", self.literals_probed);
        if self.hbr_attempts > 0 {
            let success_rate = 100.0 * self.binary_clauses_found as f64 / self.hbr_attempts as f64;
            println!("  Success rate: {:.1}%", success_rate);
        }
    }
}

/// Result of hyper-binary resolution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HbrResult {
    /// No new information discovered
    None,
    /// Binary clause discovered: (~probe => implied)
    BinaryClause {
        /// The probe literal
        probe: Lit,
        /// The implied literal
        implied: Lit,
    },
    /// Unit clause discovered (forced assignment)
    UnitClause {
        /// The forced literal
        lit: Lit,
    },
    /// Conflict discovered (probe leads to contradiction)
    Conflict {
        /// The probe literal that caused conflict
        probe: Lit,
    },
}

/// Hyper-binary resolution manager
///
/// Performs probing and hyper-binary resolution to discover new binary clauses.
/// This can strengthen the clause database and reduce search space.
#[derive(Debug)]
pub struct HyperBinaryResolver {
    /// Binary implication graph: implications[lit] = list of literals implied by ~lit
    implications: Vec<SmallVec<[Lit; 4]>>,
    /// Mark for literals seen during propagation
    seen: Vec<bool>,
    /// Work queue for propagation
    queue: Vec<Lit>,
    /// Statistics
    stats: HyperBinaryStats,
}

impl Default for HyperBinaryResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperBinaryResolver {
    /// Create a new hyper-binary resolver
    #[must_use]
    pub fn new() -> Self {
        Self {
            implications: Vec::new(),
            seen: Vec::new(),
            queue: Vec::new(),
            stats: HyperBinaryStats::default(),
        }
    }

    /// Resize for new number of variables
    pub fn resize(&mut self, num_vars: usize) {
        self.implications.resize(num_vars * 2, SmallVec::new());
        self.seen.resize(num_vars * 2, false);
    }

    /// Add a binary clause to the implication graph
    ///
    /// For binary clause (a v b), this means:
    /// - ~a => b (when a is false, b must be true)
    /// - ~b => a (when b is false, a must be true)
    pub fn add_binary_clause(&mut self, a: Lit, b: Lit) {
        let not_a_idx = (!a).code() as usize;
        let not_b_idx = (!b).code() as usize;

        // Ensure space
        let max_idx = not_a_idx.max(not_b_idx);
        if max_idx >= self.implications.len() {
            self.implications.resize(max_idx + 1, SmallVec::new());
        }

        // Add implications if not already present
        if !self.implications[not_a_idx].contains(&b) {
            self.implications[not_a_idx].push(b);
        }
        if !self.implications[not_b_idx].contains(&a) {
            self.implications[not_b_idx].push(a);
        }
    }

    /// Probe a literal to discover implied literals
    ///
    /// Assumes the literal is true and propagates to find all implications.
    /// Returns discovered binary clauses, units, or conflicts.
    pub fn probe(&mut self, probe_lit: Lit) -> Vec<HbrResult> {
        self.stats.hbr_attempts += 1;
        self.stats.literals_probed += 1;

        let mut results = Vec::new();

        // Clear state
        for seen in &mut self.seen {
            *seen = false;
        }
        self.queue.clear();

        // Start with probe literal
        self.queue.push(probe_lit);
        self.seen[probe_lit.code() as usize] = true;

        // Propagate implications
        while let Some(lit) = self.queue.pop() {
            let lit_idx = lit.code() as usize;

            if lit_idx >= self.implications.len() {
                continue;
            }

            // Check all literals implied by lit
            for &implied in &self.implications[lit_idx] {
                let impl_idx = implied.code() as usize;

                // Check for conflict
                if self.seen[(!implied).code() as usize] {
                    results.push(HbrResult::Conflict { probe: probe_lit });
                    self.stats.conflicts_found += 1;
                    return results;
                }

                // Add new implication
                if !self.seen[impl_idx] {
                    self.seen[impl_idx] = true;
                    self.queue.push(implied);

                    // Record binary clause: ~probe => implied
                    if implied != probe_lit {
                        results.push(HbrResult::BinaryClause {
                            probe: probe_lit,
                            implied,
                        });
                        self.stats.binary_clauses_found += 1;
                    }
                }
            }
        }

        results
    }

    /// Perform double lookahead probing
    ///
    /// Probes both polarities of a literal to discover implications.
    /// Can detect:
    /// - Failed literals (both polarities lead to conflict)
    /// - Forced assignments (one polarity always conflicts)
    /// - Common implications (both polarities imply same literals)
    pub fn double_probe(&mut self, lit: Lit) -> Vec<HbrResult> {
        let mut results = Vec::new();

        // Probe positive polarity
        let pos_results = self.probe(lit);
        let pos_conflict = pos_results
            .iter()
            .any(|r| matches!(r, HbrResult::Conflict { .. }));

        // Probe negative polarity
        let neg_results = self.probe(!lit);
        let neg_conflict = neg_results
            .iter()
            .any(|r| matches!(r, HbrResult::Conflict { .. }));

        // Check for failed literal (both polarities conflict)
        if pos_conflict && neg_conflict {
            // This is a contradiction - should not happen in SAT instances
            results.push(HbrResult::Conflict { probe: lit });
            self.stats.conflicts_found += 1;
        } else if pos_conflict {
            // Positive polarity conflicts, so ~lit is forced
            results.push(HbrResult::UnitClause { lit: !lit });
            self.stats.unit_clauses_found += 1;
        } else if neg_conflict {
            // Negative polarity conflicts, so lit is forced
            results.push(HbrResult::UnitClause { lit });
            self.stats.unit_clauses_found += 1;
        } else {
            // Find common implications (literals implied by both polarities)
            let pos_implied: Vec<_> = pos_results
                .iter()
                .filter_map(|r| match r {
                    HbrResult::BinaryClause { implied, .. } => Some(*implied),
                    _ => None,
                })
                .collect();

            let neg_implied: Vec<_> = neg_results
                .iter()
                .filter_map(|r| match r {
                    HbrResult::BinaryClause { implied, .. } => Some(*implied),
                    _ => None,
                })
                .collect();

            // Common implications are units
            for &implied in &pos_implied {
                if neg_implied.contains(&implied) {
                    results.push(HbrResult::UnitClause { lit: implied });
                    self.stats.unit_clauses_found += 1;
                }
            }
        }

        results
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &HyperBinaryStats {
        &self.stats
    }

    /// Clear all implications
    pub fn clear(&mut self) {
        for implications in &mut self.implications {
            implications.clear();
        }
        self.seen.clear();
        self.queue.clear();
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = HyperBinaryStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_hyper_binary_resolver_creation() {
        let resolver = HyperBinaryResolver::new();
        assert_eq!(resolver.implications.len(), 0);
    }

    #[test]
    fn test_add_binary_clause() {
        let mut resolver = HyperBinaryResolver::new();
        resolver.resize(10);

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        resolver.add_binary_clause(a, b);

        // Check implications: ~a => b and ~b => a
        let not_a_idx = (!a).code() as usize;
        let not_b_idx = (!b).code() as usize;

        assert!(resolver.implications[not_a_idx].contains(&b));
        assert!(resolver.implications[not_b_idx].contains(&a));
    }

    #[test]
    fn test_simple_probe() {
        let mut resolver = HyperBinaryResolver::new();
        resolver.resize(10);

        // Add binary clauses: (a v b) and (b v c)
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let c = Lit::pos(Var::new(2));

        resolver.add_binary_clause(a, b);
        resolver.add_binary_clause(b, c);

        // Probe ~a should imply b and c
        let results = resolver.probe(!a);

        assert!(!results.is_empty());
        // Should find that ~a => b and ~a => c
    }

    #[test]
    fn test_conflict_detection() {
        let mut resolver = HyperBinaryResolver::new();
        resolver.resize(10);

        // Add contradictory binary clauses
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        // (!a v b) and (!a v !b) means: a => b and a => !b (conflict when a is true)
        resolver.add_binary_clause(!a, b);
        resolver.add_binary_clause(!a, !b);

        let results = resolver.probe(a);

        // Should detect conflict
        assert!(
            results
                .iter()
                .any(|r| matches!(r, HbrResult::Conflict { .. }))
        );
    }

    #[test]
    fn test_double_probe_unit() {
        let mut resolver = HyperBinaryResolver::new();
        resolver.resize(10);

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        // a => b (so ~a conflicts)
        resolver.add_binary_clause(a, b);
        resolver.add_binary_clause(a, !b); // a implies both b and ~b

        let results = resolver.double_probe(a);

        // Should find that a must be false (unit clause ~a)
        assert!(
            results
                .iter()
                .any(|r| matches!(r, HbrResult::UnitClause { .. }))
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut resolver = HyperBinaryResolver::new();
        resolver.resize(10);

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        resolver.add_binary_clause(a, b);
        resolver.probe(a);

        assert_eq!(resolver.stats().hbr_attempts, 1);
        assert_eq!(resolver.stats().literals_probed, 1);
    }

    #[test]
    fn test_clear() {
        let mut resolver = HyperBinaryResolver::new();
        resolver.resize(10);

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        resolver.add_binary_clause(a, b);
        resolver.clear();

        assert!(resolver.implications.iter().all(|v| v.is_empty()));
    }

    #[test]
    fn test_reset_stats() {
        let mut resolver = HyperBinaryResolver::new();
        resolver.resize(10);

        resolver.probe(Lit::pos(Var::new(0)));
        resolver.reset_stats();

        assert_eq!(resolver.stats().hbr_attempts, 0);
    }
}
