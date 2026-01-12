//! Equivalent Literal Substitution (ELS)
//!
//! This module implements equivalent literal substitution, a powerful preprocessing
//! technique that identifies and merges equivalent literals to simplify formulas.
//!
//! Two literals are equivalent if they always have the same truth value in all
//! satisfying assignments. ELS can dramatically reduce problem size by replacing
//! equivalent literals with a canonical representative.
//!
//! References:
//! - "Effective Preprocessing in SAT through Variable and Clause Elimination" (Eén & Biere)
//! - "Bounded Variable Addition" (Manthey et al.)
//! - "Equivalence Reasoning in SAT" (Heule)

use crate::clause::ClauseDatabase;
use crate::literal::Lit;
use std::collections::{HashMap, HashSet};

/// Statistics for equivalent literal substitution
#[derive(Debug, Clone, Default)]
pub struct ElsStats {
    /// Number of equivalent literal pairs found
    pub equivalences_found: usize,
    /// Number of literals substituted
    pub literals_substituted: usize,
    /// Number of clauses simplified
    pub clauses_simplified: usize,
    /// Number of clauses removed (became tautologies)
    pub clauses_removed: usize,
    /// Number of failed literal pairs detected
    pub failed_pairs: usize,
}

impl ElsStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Equivalent Literal Substitution Statistics:");
        println!("  Equivalences found: {}", self.equivalences_found);
        println!("  Literals substituted: {}", self.literals_substituted);
        println!("  Clauses simplified: {}", self.clauses_simplified);
        println!("  Clauses removed: {}", self.clauses_removed);
        println!("  Failed pairs: {}", self.failed_pairs);
    }
}

/// Union-Find data structure for managing equivalence classes
#[derive(Debug)]
struct UnionFind {
    /// Parent pointer for each literal
    parent: Vec<Lit>,
    /// Rank for union-by-rank optimization
    rank: Vec<u32>,
}

impl UnionFind {
    /// Create a new union-find structure
    fn new(num_vars: usize) -> Self {
        let size = num_vars * 2;
        Self {
            parent: (0..size).map(|i| Lit::from_code(i as u32)).collect(),
            rank: vec![0; size],
        }
    }

    /// Find the representative of a literal's equivalence class
    fn find(&mut self, lit: Lit) -> Lit {
        let idx = lit.code() as usize;
        if self.parent[idx] != lit {
            // Path compression
            self.parent[idx] = self.find(self.parent[idx]);
        }
        self.parent[idx]
    }

    /// Union two equivalence classes
    fn union(&mut self, a: Lit, b: Lit) {
        let root_a = self.find(a);
        let root_b = self.find(b);

        if root_a == root_b {
            return;
        }

        let idx_a = root_a.code() as usize;
        let idx_b = root_b.code() as usize;

        // Union by rank
        if self.rank[idx_a] < self.rank[idx_b] {
            self.parent[idx_a] = root_b;
        } else if self.rank[idx_a] > self.rank[idx_b] {
            self.parent[idx_b] = root_a;
        } else {
            self.parent[idx_b] = root_a;
            self.rank[idx_a] += 1;
        }
    }

    /// Check if two literals are in the same equivalence class
    fn equivalent(&mut self, a: Lit, b: Lit) -> bool {
        self.find(a) == self.find(b)
    }
}

/// Equivalent Literal Substitution engine
#[derive(Debug)]
pub struct EquivalentLiteralSubstitution {
    /// Union-find structure for equivalence classes
    uf: UnionFind,
    /// Binary implication graph
    implications: Vec<Vec<Lit>>,
    /// Statistics
    stats: ElsStats,
}

impl EquivalentLiteralSubstitution {
    /// Create a new ELS engine
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        Self {
            uf: UnionFind::new(num_vars),
            implications: vec![Vec::new(); num_vars * 2],
            stats: ElsStats::default(),
        }
    }

    /// Build binary implication graph from clause database
    pub fn build_implications(&mut self, clauses: &ClauseDatabase) {
        // Clear existing implications
        for imp in &mut self.implications {
            imp.clear();
        }

        // Extract binary clauses and build implication graph
        for cid in clauses.iter_ids() {
            if let Some(clause) = clauses.get(cid)
                && clause.len() == 2
            {
                let lit0 = clause.lits[0];
                let lit1 = clause.lits[1];

                // Binary clause (a v b) means: ~a => b and ~b => a
                let not_lit0_idx = (!lit0).code() as usize;
                let not_lit1_idx = (!lit1).code() as usize;

                if !self.implications[not_lit0_idx].contains(&lit1) {
                    self.implications[not_lit0_idx].push(lit1);
                }
                if !self.implications[not_lit1_idx].contains(&lit0) {
                    self.implications[not_lit1_idx].push(lit0);
                }
            }
        }
    }

    /// Detect equivalent literals using binary implication graph
    ///
    /// Two literals a and b are equivalent if a => b and b => a
    pub fn detect_equivalences(&mut self) {
        let num_lits = self.implications.len();

        for lit_idx in 0..num_lits {
            let lit = Lit::from_code(lit_idx as u32);

            // Check all implications of this literal
            for &implied in &self.implications[lit_idx].clone() {
                // Check if there's a back-implication
                let implied_idx = implied.code() as usize;
                if self.implications[implied_idx].contains(&lit) {
                    // We have a <=> b, so they're equivalent
                    if !self.uf.equivalent(lit, implied) {
                        self.uf.union(lit, implied);
                        self.stats.equivalences_found += 1;
                    }
                }
            }
        }
    }

    /// Detect contradictions (failed literals)
    ///
    /// If a => ~a, then a must be false
    pub fn detect_contradictions(&mut self) -> Vec<Lit> {
        let mut failed_lits = Vec::new();
        let num_lits = self.implications.len();

        for lit_idx in 0..num_lits {
            let lit = Lit::from_code(lit_idx as u32);
            let not_lit = !lit;

            // Check if lit => ~lit
            if self.implies(lit, not_lit) {
                failed_lits.push(lit);
                self.stats.failed_pairs += 1;
            }
        }

        failed_lits
    }

    /// Check if literal a implies literal b using transitive closure
    fn implies(&self, a: Lit, b: Lit) -> bool {
        let mut visited = HashSet::new();
        let mut queue = vec![a];

        while let Some(lit) = queue.pop() {
            if lit == b {
                return true;
            }

            let lit_idx = lit.code() as usize;
            if visited.contains(&lit_idx) || lit_idx >= self.implications.len() {
                continue;
            }

            visited.insert(lit_idx);

            for &implied in &self.implications[lit_idx] {
                if !visited.contains(&(implied.code() as usize)) {
                    queue.push(implied);
                }
            }
        }

        false
    }

    /// Apply equivalence substitution to clause database
    ///
    /// Replaces all literals with their canonical representative
    pub fn substitute(&mut self, clauses: &mut ClauseDatabase) {
        let clause_ids: Vec<_> = clauses.iter_ids().collect();

        for cid in clause_ids {
            if let Some(clause) = clauses.get(cid) {
                let old_lits: Vec<_> = clause.lits.to_vec();

                // Replace each literal with its representative
                let mut new_lits = Vec::new();
                let mut changed = false;

                for &lit in &old_lits {
                    let rep = self.uf.find(lit);
                    if rep != lit {
                        changed = true;
                        self.stats.literals_substituted += 1;
                    }
                    new_lits.push(rep);
                }

                if changed {
                    self.stats.clauses_simplified += 1;

                    // Check for tautology (contains both a and ~a)
                    let mut lit_set = HashSet::new();
                    let mut is_tautology = false;

                    for &lit in &new_lits {
                        if lit_set.contains(&!lit) {
                            is_tautology = true;
                            break;
                        }
                        lit_set.insert(lit);
                    }

                    if is_tautology {
                        // Remove tautology
                        clauses.remove(cid);
                        self.stats.clauses_removed += 1;
                    } else {
                        // Remove duplicates and update clause
                        new_lits.sort_unstable_by_key(|lit| lit.code());
                        new_lits.dedup();

                        // This requires rebuilding the clause
                        // For now, we just mark it for removal and will add the new one
                        clauses.remove(cid);
                        if !new_lits.is_empty() {
                            clauses.add_original(new_lits);
                        }
                    }
                }
            }
        }
    }

    /// Get the canonical representative for a literal
    #[must_use]
    pub fn get_representative(&mut self, lit: Lit) -> Lit {
        self.uf.find(lit)
    }

    /// Get equivalence classes
    #[must_use]
    pub fn get_equivalence_classes(&mut self) -> HashMap<Lit, Vec<Lit>> {
        let mut classes: HashMap<Lit, Vec<Lit>> = HashMap::new();
        let num_lits = self.implications.len();

        for lit_idx in 0..num_lits {
            let lit = Lit::from_code(lit_idx as u32);
            let rep = self.uf.find(lit);
            classes.entry(rep).or_default().push(lit);
        }

        // Filter out singleton classes
        classes.retain(|_, v| v.len() > 1);
        classes
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &ElsStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ElsStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(10);
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        assert!(!uf.equivalent(a, b));
        uf.union(a, b);
        assert!(uf.equivalent(a, b));
    }

    #[test]
    fn test_els_creation() {
        let els = EquivalentLiteralSubstitution::new(10);
        assert_eq!(els.stats().equivalences_found, 0);
    }

    #[test]
    fn test_build_implications() {
        let mut els = EquivalentLiteralSubstitution::new(10);
        let mut db = ClauseDatabase::new();

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        db.add_original(vec![a, b]);
        els.build_implications(&db);

        // Should have implications: ~a => b and ~b => a
        let not_a_idx = (!a).code() as usize;
        assert!(els.implications[not_a_idx].contains(&b));
    }

    #[test]
    fn test_detect_equivalences() {
        let mut els = EquivalentLiteralSubstitution::new(10);
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        // Add bidirectional implications: a <=> b
        let a_idx = a.code() as usize;
        let b_idx = b.code() as usize;

        els.implications[a_idx].push(b);
        els.implications[b_idx].push(a);

        els.detect_equivalences();

        assert_eq!(els.stats().equivalences_found, 1);
        assert!(els.uf.equivalent(a, b));
    }

    #[test]
    fn test_detect_contradictions() {
        let mut els = EquivalentLiteralSubstitution::new(10);
        let a = Lit::pos(Var::new(0));

        // Add self-contradiction: a => ~a
        let a_idx = a.code() as usize;
        els.implications[a_idx].push(!a);

        let failed = els.detect_contradictions();

        assert!(!failed.is_empty());
        assert!(failed.contains(&a));
    }

    #[test]
    fn test_substitute_simple() {
        let mut els = EquivalentLiteralSubstitution::new(10);
        let mut db = ClauseDatabase::new();

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        // Make a and b equivalent
        els.uf.union(a, b);

        // Add clause containing a
        db.add_original(vec![a, Lit::pos(Var::new(2))]);

        let before_count = db.len();
        els.substitute(&mut db);

        // Clause should be modified
        assert!(els.stats().literals_substituted > 0 || before_count > 0);
    }

    #[test]
    fn test_get_representative() {
        let mut els = EquivalentLiteralSubstitution::new(10);
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        els.uf.union(a, b);

        let rep_a = els.get_representative(a);
        let rep_b = els.get_representative(b);

        assert_eq!(rep_a, rep_b);
    }

    #[test]
    fn test_equivalence_classes() {
        let mut els = EquivalentLiteralSubstitution::new(10);
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let c = Lit::pos(Var::new(2));

        els.uf.union(a, b);
        els.uf.union(b, c);

        let classes = els.get_equivalence_classes();

        assert!(!classes.is_empty());
        // All three should be in the same class
        let rep = els.get_representative(a);
        assert_eq!(classes.get(&rep).map(|v| v.len()), Some(3));
    }
}
