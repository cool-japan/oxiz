//! Clause representation for NLSAT.
//!
//! This module provides the clause data structure and clause database
//! for the NLSAT solver. Clauses are disjunctions of literals.
//!
//! Reference: Z3's `nlsat/nlsat_clause.h`

use crate::types::{BoolVar, Literal, NULL_BOOL_VAR};
use oxiz_math::polynomial::Var;
use std::fmt;

/// Clause identifier.
pub type ClauseId = u32;

/// Null clause ID.
pub const NULL_CLAUSE: ClauseId = u32::MAX;

/// A clause is a disjunction of literals.
#[derive(Clone)]
pub struct Clause {
    /// The literals in the clause.
    literals: Vec<Literal>,
    /// Maximum arithmetic variable in any atom of this clause.
    max_var: Var,
    /// Whether this clause is learned (vs. input).
    learned: bool,
    /// Activity score for clause deletion.
    activity: f64,
    /// Literal Block Distance (LBD) - number of distinct decision levels.
    /// Lower LBD indicates a more "glue" clause that is useful to keep.
    lbd: u32,
    /// Unique identifier.
    id: ClauseId,
}

impl Clause {
    /// Create a new clause.
    pub fn new(literals: Vec<Literal>, max_var: Var, learned: bool, id: ClauseId) -> Self {
        Self {
            literals,
            max_var,
            learned,
            activity: 0.0,
            lbd: u32::MAX, // Will be computed later
            id,
        }
    }

    /// Create a unit clause.
    pub fn unit(lit: Literal, max_var: Var, id: ClauseId) -> Self {
        Self::new(vec![lit], max_var, false, id)
    }

    /// Create a binary clause.
    pub fn binary(lit1: Literal, lit2: Literal, max_var: Var, id: ClauseId) -> Self {
        Self::new(vec![lit1, lit2], max_var, false, id)
    }

    /// Get the clause ID.
    #[inline]
    pub fn id(&self) -> ClauseId {
        self.id
    }

    /// Get the literals.
    #[inline]
    pub fn literals(&self) -> &[Literal] {
        &self.literals
    }

    /// Get mutable literals.
    #[inline]
    pub fn literals_mut(&mut self) -> &mut [Literal] {
        &mut self.literals
    }

    /// Get the number of literals.
    #[inline]
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if the clause is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if the clause is unit.
    #[inline]
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Check if the clause is binary.
    #[inline]
    pub fn is_binary(&self) -> bool {
        self.literals.len() == 2
    }

    /// Get the maximum variable.
    #[inline]
    pub fn max_var(&self) -> Var {
        self.max_var
    }

    /// Check if this is a learned clause.
    #[inline]
    pub fn is_learned(&self) -> bool {
        self.learned
    }

    /// Get the activity score.
    #[inline]
    pub fn activity(&self) -> f64 {
        self.activity
    }

    /// Set the activity score.
    #[inline]
    pub fn set_activity(&mut self, activity: f64) {
        self.activity = activity;
    }

    /// Increase the activity score.
    #[inline]
    pub fn bump_activity(&mut self, delta: f64) {
        self.activity += delta;
    }

    /// Get the Literal Block Distance (LBD).
    #[inline]
    pub fn lbd(&self) -> u32 {
        self.lbd
    }

    /// Set the Literal Block Distance (LBD).
    #[inline]
    pub fn set_lbd(&mut self, lbd: u32) {
        self.lbd = lbd;
    }

    /// Get a specific literal.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<Literal> {
        self.literals.get(idx).copied()
    }

    /// Swap two literals.
    #[inline]
    pub fn swap(&mut self, i: usize, j: usize) {
        self.literals.swap(i, j);
    }

    /// Check if the clause contains a literal.
    pub fn contains(&self, lit: Literal) -> bool {
        self.literals.contains(&lit)
    }

    /// Check if the clause contains a variable (in either polarity).
    pub fn contains_var(&self, var: BoolVar) -> bool {
        self.literals.iter().any(|l| l.var() == var)
    }

    /// Get the first literal (for watched literal scheme).
    #[inline]
    pub fn first(&self) -> Option<Literal> {
        self.literals.first().copied()
    }

    /// Get the second literal (for watched literal scheme).
    #[inline]
    pub fn second(&self) -> Option<Literal> {
        self.literals.get(1).copied()
    }
}

impl fmt::Debug for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Clause[{}]: ", self.id)?;
        for (i, lit) in self.literals.iter().enumerate() {
            if i > 0 {
                write!(f, " ∨ ")?;
            }
            write!(f, "{:?}", lit)?;
        }
        if self.learned {
            write!(f, " (learned)")?;
        }
        Ok(())
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Watch list entry for the two-watched-literal scheme.
#[derive(Clone, Copy, Debug)]
pub struct Watch {
    /// The clause being watched.
    pub clause_id: ClauseId,
    /// The blocker literal (optimization).
    pub blocker: Literal,
}

impl Watch {
    /// Create a new watch entry.
    #[inline]
    pub fn new(clause_id: ClauseId, blocker: Literal) -> Self {
        Self { clause_id, blocker }
    }
}

/// Clause database for the NLSAT solver.
#[derive(Clone)]
pub struct ClauseDatabase {
    /// All clauses.
    clauses: Vec<Clause>,
    /// Watch lists for each literal.
    watches: Vec<Vec<Watch>>,
    /// Number of boolean variables.
    num_bool_vars: u32,
    /// Number of learned clauses.
    num_learned: u32,
    /// Next clause ID.
    next_id: ClauseId,
    /// Activity increment for bumping.
    activity_inc: f64,
    /// Activity decay factor.
    activity_decay: f64,
}

impl ClauseDatabase {
    /// Create a new clause database.
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            watches: Vec::new(),
            num_bool_vars: 0,
            num_learned: 0,
            next_id: 0,
            activity_inc: 1.0,
            activity_decay: 0.95,
        }
    }

    /// Create with capacity.
    pub fn with_capacity(num_clauses: usize, num_bool_vars: usize) -> Self {
        let mut db = Self::new();
        db.clauses.reserve(num_clauses);
        db.ensure_bool_var(num_bool_vars as BoolVar);
        db
    }

    /// Ensure we have watch lists for a boolean variable.
    pub fn ensure_bool_var(&mut self, var: BoolVar) {
        if var == NULL_BOOL_VAR {
            return;
        }
        let needed = ((var as usize) + 1) * 2;
        if needed > self.watches.len() {
            self.watches.resize(needed, Vec::new());
            self.num_bool_vars = var + 1;
        }
    }

    /// Get the number of clauses.
    #[inline]
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Get the number of learned clauses.
    #[inline]
    pub fn num_learned(&self) -> u32 {
        self.num_learned
    }

    /// Get a clause by ID.
    pub fn get(&self, id: ClauseId) -> Option<&Clause> {
        self.clauses.iter().find(|c| c.id == id)
    }

    /// Get a mutable clause by ID.
    pub fn get_mut(&mut self, id: ClauseId) -> Option<&mut Clause> {
        self.clauses.iter_mut().find(|c| c.id == id)
    }

    /// Get all clauses.
    #[inline]
    pub fn clauses(&self) -> &[Clause] {
        &self.clauses
    }

    /// Add a clause to the database.
    /// Returns the clause ID.
    pub fn add(&mut self, literals: Vec<Literal>, max_var: Var, learned: bool) -> ClauseId {
        // Ensure watch lists exist for all literals
        for lit in &literals {
            self.ensure_bool_var(lit.var());
        }

        let id = self.next_id;
        self.next_id += 1;

        if learned {
            self.num_learned += 1;
        }

        // If clause has at least 2 literals, add watches
        if literals.len() >= 2 {
            let lit0 = literals[0];
            let lit1 = literals[1];

            // Watch the negation of the first two literals
            self.watches[lit0.negate().index() as usize].push(Watch::new(id, lit1));
            self.watches[lit1.negate().index() as usize].push(Watch::new(id, lit0));
        }

        self.clauses
            .push(Clause::new(literals, max_var, learned, id));
        id
    }

    /// Add a unit clause.
    pub fn add_unit(&mut self, lit: Literal, max_var: Var) -> ClauseId {
        self.add(vec![lit], max_var, false)
    }

    /// Get the watch list for a literal.
    pub fn watches(&self, lit: Literal) -> &[Watch] {
        self.watches
            .get(lit.index() as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get mutable watch list for a literal.
    pub fn watches_mut(&mut self, lit: Literal) -> &mut Vec<Watch> {
        let idx = lit.index() as usize;
        if idx >= self.watches.len() {
            self.watches.resize(idx + 1, Vec::new());
        }
        &mut self.watches[idx]
    }

    /// Remove a watch entry.
    pub fn remove_watch(&mut self, lit: Literal, clause_id: ClauseId) {
        if let Some(watches) = self.watches.get_mut(lit.index() as usize) {
            watches.retain(|w| w.clause_id != clause_id);
        }
    }

    /// Bump the activity of a clause.
    pub fn bump_activity(&mut self, clause_id: ClauseId) {
        let activity_inc = self.activity_inc;
        let need_rescale;
        if let Some(clause) = self.get_mut(clause_id) {
            clause.bump_activity(activity_inc);
            need_rescale = clause.activity() > 1e100;
        } else {
            return;
        }

        // Rescale if activity gets too large
        if need_rescale {
            self.rescale_activities();
        }
    }

    /// Decay all activities.
    pub fn decay_activities(&mut self) {
        self.activity_inc *= 1.0 / self.activity_decay;
    }

    /// Rescale all activities to prevent overflow.
    fn rescale_activities(&mut self) {
        for clause in &mut self.clauses {
            let new_activity = clause.activity() * 1e-100;
            clause.set_activity(new_activity);
        }
        self.activity_inc *= 1e-100;
    }

    /// Remove learned clauses with low activity.
    /// Returns the IDs of removed clauses.
    pub fn reduce_learned(&mut self, keep_fraction: f64) -> Vec<ClauseId> {
        // Collect learned clause IDs with LBD and activity
        let learned: Vec<_> = self
            .clauses
            .iter()
            .filter(|c| c.is_learned())
            .map(|c| (c.id, c.lbd(), c.activity()))
            .collect();

        // Glucose-style reduction: protect glue clauses (LBD <= 2)
        let (_glue_clauses, non_glue): (Vec<_>, Vec<_>) =
            learned.into_iter().partition(|(_, lbd, _)| *lbd <= 2);

        // For non-glue clauses, sort by LBD first (higher is worse),
        // then by activity (lower is worse)
        let mut non_glue = non_glue;
        non_glue.sort_by(|a, b| {
            // Sort by LBD descending (higher LBD = more likely to remove)
            match b.1.cmp(&a.1) {
                std::cmp::Ordering::Equal => {
                    // If LBD is equal, sort by activity ascending (lower activity = more likely to remove)
                    a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
                }
                other => other,
            }
        });

        // Keep the top fraction of non-glue clauses
        let keep_count = (non_glue.len() as f64 * keep_fraction) as usize;
        let remove: Vec<ClauseId> = non_glue
            .iter()
            .skip(keep_count)
            .map(|(id, _, _)| *id)
            .collect();

        // Remove clauses (glue clauses are always kept)
        for &id in &remove {
            self.remove(id);
        }

        remove
    }

    /// Remove a clause from the database.
    pub fn remove(&mut self, clause_id: ClauseId) {
        if let Some(pos) = self.clauses.iter().position(|c| c.id == clause_id) {
            // Get data we need before mutating
            let len = self.clauses[pos].len();
            let lit0 = if len >= 2 {
                Some(self.clauses[pos].literals[0])
            } else {
                None
            };
            let lit1 = if len >= 2 {
                Some(self.clauses[pos].literals[1])
            } else {
                None
            };
            let is_learned = self.clauses[pos].is_learned();

            // Remove watches
            if let (Some(l0), Some(l1)) = (lit0, lit1) {
                self.remove_watch(l0.negate(), clause_id);
                self.remove_watch(l1.negate(), clause_id);
            }

            if is_learned {
                self.num_learned -= 1;
            }

            self.clauses.remove(pos);
        }
    }

    /// Clear all clauses.
    pub fn clear(&mut self) {
        self.clauses.clear();
        for w in &mut self.watches {
            w.clear();
        }
        self.num_learned = 0;
        self.next_id = 0;
    }

    /// Get clauses that contain a specific variable.
    pub fn clauses_with_var(&self, var: BoolVar) -> Vec<ClauseId> {
        self.clauses
            .iter()
            .filter(|c| c.contains_var(var))
            .map(|c| c.id)
            .collect()
    }

    /// Find unit clauses.
    pub fn find_unit_clauses(&self) -> Vec<(ClauseId, Literal)> {
        self.clauses
            .iter()
            .filter(|c| c.is_unit())
            .filter_map(|c| c.first().map(|lit| (c.id, lit)))
            .collect()
    }
}

impl Default for ClauseDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clause_basic() {
        let lits = vec![
            Literal::positive(0),
            Literal::negative(1),
            Literal::positive(2),
        ];
        let clause = Clause::new(lits, 0, false, 0);

        assert_eq!(clause.len(), 3);
        assert!(!clause.is_unit());
        assert!(!clause.is_binary());
        assert!(!clause.is_learned());
        assert!(clause.contains(Literal::positive(0)));
        assert!(clause.contains(Literal::negative(1)));
        assert!(!clause.contains(Literal::positive(1)));
    }

    #[test]
    fn test_clause_unit() {
        let clause = Clause::unit(Literal::positive(5), 0, 0);
        assert!(clause.is_unit());
        assert_eq!(clause.first(), Some(Literal::positive(5)));
    }

    #[test]
    fn test_clause_database() {
        let mut db = ClauseDatabase::new();

        // Add a clause
        let id1 = db.add(vec![Literal::positive(0), Literal::negative(1)], 0, false);
        assert_eq!(db.num_clauses(), 1);

        // Add a learned clause
        let id2 = db.add(vec![Literal::positive(2), Literal::positive(3)], 0, true);
        assert_eq!(db.num_clauses(), 2);
        assert_eq!(db.num_learned(), 1);

        // Get clause
        let c1 = db.get(id1).unwrap();
        assert!(!c1.is_learned());
        assert_eq!(c1.len(), 2);

        let c2 = db.get(id2).unwrap();
        assert!(c2.is_learned());
    }

    #[test]
    fn test_clause_database_watches() {
        let mut db = ClauseDatabase::new();

        // Add a clause [a, b, c]
        let id = db.add(
            vec![
                Literal::positive(0),
                Literal::positive(1),
                Literal::positive(2),
            ],
            0,
            false,
        );

        // Watches should be on ~a and ~b
        assert_eq!(db.watches(Literal::negative(0)).len(), 1);
        assert_eq!(db.watches(Literal::negative(1)).len(), 1);
        assert_eq!(db.watches(Literal::negative(2)).len(), 0);
        assert_eq!(db.watches(Literal::negative(0))[0].clause_id, id);
    }

    #[test]
    fn test_clause_database_remove() {
        let mut db = ClauseDatabase::new();

        let id1 = db.add(vec![Literal::positive(0), Literal::negative(1)], 0, false);
        let _id2 = db.add(vec![Literal::positive(2), Literal::positive(3)], 0, true);

        assert_eq!(db.num_clauses(), 2);
        assert_eq!(db.num_learned(), 1);

        db.remove(id1);
        assert_eq!(db.num_clauses(), 1);
        assert!(db.get(id1).is_none());
    }

    // Property-based tests
    use proptest::prelude::*;

    proptest! {
        /// Property: Clause length matches number of literals
        #[test]
        fn prop_clause_length_matches_literals(lits in prop::collection::vec(0u32..100, 1..20)) {
            let literals: Vec<_> = lits.iter().enumerate()
                .map(|(i, &v)| if i % 2 == 0 { Literal::positive(v) } else { Literal::negative(v) })
                .collect();
            let clause = Clause::new(literals.clone(), 0, false, 0);

            prop_assert_eq!(clause.len(), literals.len());
            prop_assert_eq!(clause.is_empty(), literals.is_empty());
        }

        /// Property: Unit clause has exactly one literal
        #[test]
        fn prop_unit_clause_has_one_literal(var in 0u32..1000, sign: bool) {
            let lit = Literal::new(var, sign);
            let clause = Clause::unit(lit, 0, 0);

            prop_assert!(clause.is_unit());
            prop_assert_eq!(clause.len(), 1);
            prop_assert_eq!(clause.first(), Some(lit));
        }

        /// Property: Binary clause has exactly two literals
        #[test]
        fn prop_binary_clause_has_two_literals(v1 in 0u32..500, v2 in 500u32..1000, s1: bool, s2: bool) {
            let lit1 = Literal::new(v1, s1);
            let lit2 = Literal::new(v2, s2);
            let clause = Clause::binary(lit1, lit2, 0, 0);

            prop_assert!(clause.is_binary());
            prop_assert_eq!(clause.len(), 2);
        }

        /// Property: Clause contains exactly its literals
        #[test]
        fn prop_clause_contains_its_literals(lits in prop::collection::vec(0u32..50, 1..10)) {
            let literals: Vec<_> = lits.iter().enumerate()
                .map(|(i, &v)| if i % 2 == 0 { Literal::positive(v) } else { Literal::negative(v) })
                .collect();
            let clause = Clause::new(literals.clone(), 0, false, 0);

            for lit in &literals {
                prop_assert!(clause.contains(*lit));
            }
        }

        /// Property: Activity bumping increases activity
        #[test]
        fn prop_activity_bump_increases(delta in 0.1f64..100.0) {
            let mut clause = Clause::new(vec![Literal::positive(0)], 0, false, 0);
            let initial = clause.activity();

            clause.bump_activity(delta);

            prop_assert!(clause.activity() > initial);
            prop_assert!((clause.activity() - initial - delta).abs() < 1e-10);
        }

        /// Property: LBD can be set and retrieved
        #[test]
        fn prop_lbd_set_get(lbd in 0u32..100) {
            let mut clause = Clause::new(vec![Literal::positive(0)], 0, false, 0);
            clause.set_lbd(lbd);

            prop_assert_eq!(clause.lbd(), lbd);
        }

        /// Property: ClauseDatabase preserves clause count
        #[test]
        fn prop_database_preserves_count(num_clauses in 1usize..50) {
            let mut db = ClauseDatabase::new();

            for i in 0..num_clauses {
                db.add(vec![Literal::positive(i as u32)], 0, i % 2 == 0);
            }

            prop_assert_eq!(db.num_clauses(), num_clauses);
        }

        /// Property: Retrieved clauses match added clauses
        #[test]
        fn prop_database_get_matches_add(var in 0u32..100, sign: bool) {
            let mut db = ClauseDatabase::new();
            let lit = Literal::new(var, sign);
            let id = db.add(vec![lit], 0, false);

            let clause = db.get(id);
            prop_assert!(clause.is_some());

            let clause = clause.unwrap();
            prop_assert_eq!(clause.len(), 1);
            prop_assert!(clause.contains(lit));
        }
    }
}
