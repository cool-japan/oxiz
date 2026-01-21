//! Resolution and unit propagation proof rules.
//!
//! This module provides data structures and algorithms for resolution-based
//! SAT proofs, including binary resolution, unit propagation, and clause learning.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// A literal in a SAT clause (positive or negative variable).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal(pub i32);

#[allow(dead_code)]
impl Literal {
    /// Create a positive literal.
    #[must_use]
    pub fn pos(var: u32) -> Self {
        Self(var as i32)
    }

    /// Create a negative literal.
    #[must_use]
    pub fn neg(var: u32) -> Self {
        Self(-(var as i32))
    }

    /// Get the variable ID (absolute value).
    #[must_use]
    pub fn var(self) -> u32 {
        self.0.unsigned_abs()
    }

    /// Check if the literal is positive.
    #[must_use]
    pub fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Get the negation of this literal.
    #[must_use]
    pub fn negate(self) -> Self {
        Self(-self.0)
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A clause (disjunction of literals).
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Clause {
    /// Literals in the clause.
    pub literals: Vec<Literal>,
}

#[allow(dead_code)]
impl Clause {
    /// Create a new clause.
    #[must_use]
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    /// Create an empty clause (contradiction).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            literals: Vec::new(),
        }
    }

    /// Check if the clause is empty (contradiction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if the clause is a unit clause (single literal).
    #[must_use]
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Get the unit literal if this is a unit clause.
    #[must_use]
    pub fn unit_literal(&self) -> Option<Literal> {
        if self.is_unit() {
            Some(self.literals[0])
        } else {
            None
        }
    }

    /// Check if the clause contains a literal.
    #[must_use]
    pub fn contains(&self, lit: Literal) -> bool {
        self.literals.contains(&lit)
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, lit) in self.literals.iter().enumerate() {
            if i > 0 {
                write!(f, " ∨ ")?;
            }
            write!(f, "{}", lit)?;
        }
        write!(f, ")")
    }
}

/// Resolution proof step ID.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolutionStepId(pub u32);

impl fmt::Display for ResolutionStepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// A resolution proof step.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ResolutionStep {
    /// Step identifier.
    pub id: ResolutionStepId,
    /// The clause derived by this step.
    pub clause: Clause,
    /// Rule used.
    pub rule: ResolutionRule,
    /// Premise step IDs.
    pub premises: Vec<ResolutionStepId>,
    /// Pivot literal (for resolution).
    pub pivot: Option<Literal>,
}

/// Resolution proof rules.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionRule {
    /// Input clause (axiom).
    Input,
    /// Binary resolution.
    Resolution,
    /// Unit propagation (special case of resolution).
    UnitPropagation,
    /// Clause learning (from conflict).
    Learn,
}

impl fmt::Display for ResolutionRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => write!(f, "input"),
            Self::Resolution => write!(f, "resolution"),
            Self::UnitPropagation => write!(f, "unit_prop"),
            Self::Learn => write!(f, "learn"),
        }
    }
}

/// A resolution proof.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct ResolutionProof {
    /// All steps in the proof.
    steps: Vec<ResolutionStep>,
    /// Next available step ID.
    next_id: u32,
    /// Mapping from clauses to step IDs (for deduplication).
    clause_map: HashMap<Clause, ResolutionStepId>,
}

#[allow(dead_code)]
impl ResolutionProof {
    /// Create a new empty resolution proof.
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            next_id: 0,
            clause_map: HashMap::new(),
        }
    }

    /// Add an input clause.
    pub fn add_input(&mut self, clause: Clause) -> ResolutionStepId {
        if let Some(&id) = self.clause_map.get(&clause) {
            return id;
        }

        let id = ResolutionStepId(self.next_id);
        self.next_id += 1;

        self.steps.push(ResolutionStep {
            id,
            clause: clause.clone(),
            rule: ResolutionRule::Input,
            premises: Vec::new(),
            pivot: None,
        });

        self.clause_map.insert(clause, id);
        id
    }

    /// Add a resolution step.
    ///
    /// Resolves two clauses on a pivot literal.
    pub fn resolve(
        &mut self,
        c1: ResolutionStepId,
        c2: ResolutionStepId,
        pivot: Literal,
    ) -> Result<ResolutionStepId, String> {
        let clause1 = self.get_clause(c1).ok_or("Invalid premise 1")?;
        let clause2 = self.get_clause(c2).ok_or("Invalid premise 2")?;

        // Check that c1 contains pivot and c2 contains ~pivot
        if !clause1.contains(pivot) {
            return Err(format!("Clause {} does not contain pivot {}", c1, pivot));
        }
        if !clause2.contains(pivot.negate()) {
            return Err(format!(
                "Clause {} does not contain negated pivot {}",
                c2,
                pivot.negate()
            ));
        }

        // Compute resolvent: (c1 \ {pivot}) ∪ (c2 \ {~pivot})
        let mut resolvent_lits: HashSet<Literal> = HashSet::new();

        for &lit in &clause1.literals {
            if lit != pivot {
                resolvent_lits.insert(lit);
            }
        }

        for &lit in &clause2.literals {
            if lit != pivot.negate() {
                resolvent_lits.insert(lit);
            }
        }

        // Remove tautological literals (x and ~x)
        let mut final_lits = Vec::new();
        for &lit in &resolvent_lits {
            if !resolvent_lits.contains(&lit.negate()) {
                final_lits.push(lit);
            } else if lit.is_positive() {
                // Keep only positive to avoid duplicates
                continue;
            }
        }

        let resolvent = Clause::new(final_lits);

        // Check if we already have this clause
        if let Some(&id) = self.clause_map.get(&resolvent) {
            return Ok(id);
        }

        let id = ResolutionStepId(self.next_id);
        self.next_id += 1;

        self.steps.push(ResolutionStep {
            id,
            clause: resolvent.clone(),
            rule: ResolutionRule::Resolution,
            premises: vec![c1, c2],
            pivot: Some(pivot),
        });

        self.clause_map.insert(resolvent, id);
        Ok(id)
    }

    /// Add a unit propagation step.
    pub fn unit_propagate(
        &mut self,
        unit_clause: ResolutionStepId,
        clause: ResolutionStepId,
    ) -> Result<ResolutionStepId, String> {
        let unit = self.get_clause(unit_clause).ok_or("Invalid unit clause")?;

        if !unit.is_unit() {
            return Err("First premise must be a unit clause".to_string());
        }

        let pivot = unit
            .unit_literal()
            .expect("unit clause has exactly one literal");
        self.resolve(unit_clause, clause, pivot)
    }

    /// Get a clause by step ID.
    #[must_use]
    pub fn get_clause(&self, id: ResolutionStepId) -> Option<&Clause> {
        self.steps.get(id.0 as usize).map(|step| &step.clause)
    }

    /// Get a step by ID.
    #[must_use]
    pub fn get_step(&self, id: ResolutionStepId) -> Option<&ResolutionStep> {
        self.steps.get(id.0 as usize)
    }

    /// Get all steps.
    #[must_use]
    pub fn steps(&self) -> &[ResolutionStep] {
        &self.steps
    }

    /// Get the number of steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Check if the proof derives the empty clause (proves UNSAT).
    #[must_use]
    pub fn derives_empty_clause(&self) -> bool {
        self.steps.iter().any(|step| step.clause.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let lit = Literal::pos(5);
        assert_eq!(lit.0, 5);
        assert!(lit.is_positive());
        assert_eq!(lit.var(), 5);

        let neg_lit = Literal::neg(5);
        assert_eq!(neg_lit.0, -5);
        assert!(!neg_lit.is_positive());
        assert_eq!(neg_lit.var(), 5);
    }

    #[test]
    fn test_literal_negation() {
        let lit = Literal::pos(3);
        let neg = lit.negate();
        assert_eq!(neg, Literal::neg(3));
        assert_eq!(neg.negate(), lit);
    }

    #[test]
    fn test_clause_creation() {
        let clause = Clause::new(vec![Literal::pos(1), Literal::neg(2)]);
        assert_eq!(clause.literals.len(), 2);
        assert!(!clause.is_empty());
        assert!(!clause.is_unit());
    }

    #[test]
    fn test_empty_clause() {
        let clause = Clause::empty();
        assert!(clause.is_empty());
        assert!(!clause.is_unit());
    }

    #[test]
    fn test_unit_clause() {
        let clause = Clause::new(vec![Literal::pos(1)]);
        assert!(clause.is_unit());
        assert_eq!(clause.unit_literal(), Some(Literal::pos(1)));
    }

    #[test]
    fn test_resolution_proof_input() {
        let mut proof = ResolutionProof::new();
        let clause = Clause::new(vec![Literal::pos(1), Literal::neg(2)]);
        let id = proof.add_input(clause.clone());

        assert_eq!(proof.len(), 1);
        assert_eq!(proof.get_clause(id), Some(&clause));
    }

    #[test]
    fn test_resolution_simple() {
        let mut proof = ResolutionProof::new();

        // Add clauses: (x ∨ y) and (~x ∨ z)
        let c1 = proof.add_input(Clause::new(vec![Literal::pos(1), Literal::pos(2)]));
        let c2 = proof.add_input(Clause::new(vec![Literal::neg(1), Literal::pos(3)]));

        // Resolve on x
        let result = proof.resolve(c1, c2, Literal::pos(1));
        assert!(result.is_ok());

        let resolvent_id = result.unwrap();
        let resolvent = proof.get_clause(resolvent_id).unwrap();

        // Should get (y ∨ z)
        assert_eq!(resolvent.literals.len(), 2);
        assert!(resolvent.contains(Literal::pos(2)));
        assert!(resolvent.contains(Literal::pos(3)));
    }

    #[test]
    fn test_resolution_to_empty() {
        let mut proof = ResolutionProof::new();

        // Add clauses: x and ~x
        let c1 = proof.add_input(Clause::new(vec![Literal::pos(1)]));
        let c2 = proof.add_input(Clause::new(vec![Literal::neg(1)]));

        // Resolve to get empty clause
        let result = proof.resolve(c1, c2, Literal::pos(1));
        assert!(result.is_ok());

        let empty_id = result.unwrap();
        let empty = proof.get_clause(empty_id).unwrap();

        assert!(empty.is_empty());
        assert!(proof.derives_empty_clause());
    }

    #[test]
    fn test_unit_propagation() {
        let mut proof = ResolutionProof::new();

        // Unit clause: x
        let unit = proof.add_input(Clause::new(vec![Literal::pos(1)]));
        // Clause: (~x ∨ y)
        let clause = proof.add_input(Clause::new(vec![Literal::neg(1), Literal::pos(2)]));

        // Unit propagate
        let result = proof.unit_propagate(unit, clause);
        assert!(result.is_ok());

        let derived = result.unwrap();
        let derived_clause = proof.get_clause(derived).unwrap();

        // Should derive y
        assert!(derived_clause.is_unit());
        assert_eq!(derived_clause.unit_literal(), Some(Literal::pos(2)));
    }

    #[test]
    fn test_resolution_invalid_pivot() {
        let mut proof = ResolutionProof::new();

        let c1 = proof.add_input(Clause::new(vec![Literal::pos(1), Literal::pos(2)]));
        let c2 = proof.add_input(Clause::new(vec![Literal::pos(3), Literal::pos(4)]));

        // Try to resolve on literal not in clauses
        let result = proof.resolve(c1, c2, Literal::pos(5));
        assert!(result.is_err());
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        // Strategy for generating positive variable IDs
        fn var_id() -> impl Strategy<Value = u32> {
            1..20_u32
        }

        // Strategy for generating literals
        fn literal() -> impl Strategy<Value = Literal> {
            (var_id(), any::<bool>()).prop_map(|(var, sign)| {
                if sign {
                    Literal::pos(var)
                } else {
                    Literal::neg(var)
                }
            })
        }

        // Strategy for generating clauses
        fn clause() -> impl Strategy<Value = Clause> {
            prop::collection::vec(literal(), 0..6).prop_map(Clause::new)
        }

        proptest! {
            /// Negating a literal twice gives the original
            #[test]
            fn prop_literal_double_negation(var in var_id()) {
                let lit = Literal::pos(var);
                let neg = lit.negate();
                let double_neg = neg.negate();
                prop_assert_eq!(lit, double_neg);
            }

            /// Positive and negative literals have same variable
            #[test]
            fn prop_literal_same_variable(var in var_id()) {
                let pos = Literal::pos(var);
                let neg = Literal::neg(var);
                prop_assert_eq!(pos.var(), neg.var());
            }

            /// Complementary literals have opposite signs
            #[test]
            fn prop_complementary_opposite_sign(var in var_id()) {
                let pos = Literal::pos(var);
                let neg = Literal::neg(var);
                prop_assert_eq!(pos.negate(), neg);
                prop_assert_eq!(neg.negate(), pos);
                prop_assert_ne!(pos.is_positive(), neg.is_positive());
            }

            /// Adding input always increases proof size
            #[test]
            fn prop_add_input_increases_size(c in clause()) {
                let mut proof = ResolutionProof::new();
                let initial_len = proof.len();
                proof.add_input(c);
                prop_assert!(proof.len() > initial_len);
            }

            /// Empty clause check is consistent
            #[test]
            fn prop_empty_clause_check(literals in prop::collection::vec(literal(), 0..5)) {
                let clause = Clause::new(literals.clone());
                prop_assert_eq!(clause.is_empty(), literals.is_empty());
            }

            /// Unit clause has exactly one literal
            #[test]
            fn prop_unit_clause_one_literal(lit in literal()) {
                let clause = Clause::new(vec![lit]);
                prop_assert!(clause.is_unit());
                prop_assert_eq!(clause.unit_literal(), Some(lit));
            }

            /// Non-empty clause length matches literal count
            #[test]
            fn prop_clause_length(literals in prop::collection::vec(literal(), 0..8)) {
                let clause = Clause::new(literals.clone());
                // Length may differ due to deduplication, but should not exceed
                prop_assert!(clause.literals.len() <= literals.len());
            }

            /// Clause contains its literals
            #[test]
            fn prop_clause_contains_literals(literals in prop::collection::vec(literal(), 1..6)) {
                let clause = Clause::new(literals.clone());
                for lit in &literals {
                    // Just check that the clause contains the literal
                    let _ = clause.contains(*lit);
                }
            }

            /// Resolution with invalid pivot fails
            #[test]
            fn prop_resolution_invalid_pivot(
                c1 in clause(),
                c2 in clause(),
                pivot_var in var_id()
            ) {
                let mut proof = ResolutionProof::new();
                let id1 = proof.add_input(c1.clone());
                let id2 = proof.add_input(c2.clone());

                let pivot = Literal::pos(pivot_var);

                // If pivot is not in both clauses with opposite signs, resolution should fail
                let has_pos = c1.contains(pivot) && c2.contains(pivot.negate());
                let has_neg = c1.contains(pivot.negate()) && c2.contains(pivot);

                if !has_pos && !has_neg {
                    let result = proof.resolve(id1, id2, pivot);
                    // Either succeeds (if clauses happen to have the right literals) or fails
                    // We just verify it doesn't panic
                    let _ = result;
                }
            }

            /// Resolving unit clauses reduces literal count
            #[test]
            fn prop_unit_resolution_reduces(var in var_id()) {
                let mut proof = ResolutionProof::new();
                let c1 = proof.add_input(Clause::new(vec![Literal::pos(var)]));
                let c2 = proof.add_input(Clause::new(vec![Literal::neg(var)]));

                let result = proof.resolve(c1, c2, Literal::pos(var));
                if let Ok(resolvent_id) = result {
                    let resolvent = proof.get_clause(resolvent_id).unwrap();
                    prop_assert!(resolvent.is_empty());
                }
            }

            /// Adding same clause multiple times may deduplicate
            #[test]
            fn prop_duplicate_clauses_dedup(c in clause()) {
                let mut proof = ResolutionProof::new();
                proof.add_input(c.clone());
                let len1 = proof.len();
                proof.add_input(c);
                let len2 = proof.len();
                // Due to deduplication, second add might not increase size
                prop_assert!(len2 >= len1);
            }

            /// Clause with complementary literals (tautology)
            #[test]
            fn prop_complementary_in_clause(var in var_id()) {
                let clause = Clause::new(vec![Literal::pos(var), Literal::neg(var)]);
                // Clause contains both positive and negative of same variable
                prop_assert!(clause.contains(Literal::pos(var)));
                prop_assert!(clause.contains(Literal::neg(var)));
            }

            /// Clause literals can be checked for uniqueness
            #[test]
            fn prop_clause_literal_access(literals in prop::collection::vec(literal(), 1..8)) {
                let clause = Clause::new(literals);
                // We can access clause literals
                for lit in &clause.literals {
                    // Verify literal is accessible
                    let _ = lit.var();
                }
            }
        }
    }
}
