//! CNF (Conjunctive Normal Form) transformation with proofs.
//!
//! This module provides algorithms for converting arbitrary Boolean formulas
//! to CNF with proof generation using Tseitin transformation.

use crate::resolution::{Clause, Literal};
use std::collections::HashMap;
use std::fmt;

/// A Boolean formula variable.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(pub u32);

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A Boolean formula.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Formula {
    /// Variable.
    Var(Var),
    /// Negation.
    Not(Box<Formula>),
    /// Conjunction (AND).
    And(Vec<Formula>),
    /// Disjunction (OR).
    Or(Vec<Formula>),
    /// Implication.
    Implies(Box<Formula>, Box<Formula>),
    /// Equivalence (IFF).
    Iff(Box<Formula>, Box<Formula>),
    /// Exclusive OR (XOR).
    Xor(Box<Formula>, Box<Formula>),
}

#[allow(dead_code)]
impl Formula {
    /// Create a variable formula.
    #[must_use]
    pub fn var(v: u32) -> Self {
        Self::Var(Var(v))
    }

    /// Create a negation.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(f: Formula) -> Self {
        Self::Not(Box::new(f))
    }

    /// Create a conjunction.
    #[must_use]
    pub fn and(formulas: Vec<Formula>) -> Self {
        Self::And(formulas)
    }

    /// Create a disjunction.
    #[must_use]
    pub fn or(formulas: Vec<Formula>) -> Self {
        Self::Or(formulas)
    }

    /// Create an implication.
    #[must_use]
    pub fn implies(a: Formula, b: Formula) -> Self {
        Self::Implies(Box::new(a), Box::new(b))
    }

    /// Create an equivalence.
    #[must_use]
    pub fn iff(a: Formula, b: Formula) -> Self {
        Self::Iff(Box::new(a), Box::new(b))
    }

    /// Create an XOR.
    #[must_use]
    pub fn xor(a: Formula, b: Formula) -> Self {
        Self::Xor(Box::new(a), Box::new(b))
    }
}

/// CNF transformation context.
#[allow(dead_code)]
pub struct CnfTransformer {
    /// Next available variable ID.
    next_var: u32,
    /// Mapping from subformulas to their Tseitin variables.
    #[allow(dead_code)]
    tseitin_vars: HashMap<String, Var>,
    /// Generated clauses.
    clauses: Vec<Clause>,
}

#[allow(dead_code)]
impl CnfTransformer {
    /// Create a new CNF transformer.
    #[must_use]
    pub fn new(first_var: u32) -> Self {
        Self {
            next_var: first_var,
            tseitin_vars: HashMap::new(),
            clauses: Vec::new(),
        }
    }

    /// Allocate a fresh Tseitin variable.
    fn fresh_var(&mut self) -> Var {
        let v = Var(self.next_var);
        self.next_var += 1;
        v
    }

    /// Transform a formula to CNF using Tseitin transformation.
    pub fn transform(&mut self, formula: &Formula) -> Var {
        match formula {
            Formula::Var(v) => *v,

            Formula::Not(f) => {
                let sub_var = self.transform(f);
                let result_var = self.fresh_var();

                // result_var <=> ~sub_var
                // (~result_var ∨ ~sub_var) ∧ (result_var ∨ sub_var)
                self.clauses.push(Clause::new(vec![
                    Literal::neg(result_var.0),
                    Literal::neg(sub_var.0),
                ]));
                self.clauses.push(Clause::new(vec![
                    Literal::pos(result_var.0),
                    Literal::pos(sub_var.0),
                ]));

                result_var
            }

            Formula::And(formulas) => {
                let sub_vars: Vec<Var> = formulas.iter().map(|f| self.transform(f)).collect();
                let result_var = self.fresh_var();

                // result_var <=> (v1 ∧ v2 ∧ ... ∧ vn)
                // (result_var ∨ ~v1 ∨ ~v2 ∨ ... ∨ ~vn)
                let mut clause_lits = vec![Literal::pos(result_var.0)];
                for &v in &sub_vars {
                    clause_lits.push(Literal::neg(v.0));
                }
                self.clauses.push(Clause::new(clause_lits));

                // (~result_var ∨ v1) ∧ (~result_var ∨ v2) ∧ ... ∧ (~result_var ∨ vn)
                for &v in &sub_vars {
                    self.clauses.push(Clause::new(vec![
                        Literal::neg(result_var.0),
                        Literal::pos(v.0),
                    ]));
                }

                result_var
            }

            Formula::Or(formulas) => {
                let sub_vars: Vec<Var> = formulas.iter().map(|f| self.transform(f)).collect();
                let result_var = self.fresh_var();

                // result_var <=> (v1 ∨ v2 ∨ ... ∨ vn)
                // (~result_var ∨ v1 ∨ v2 ∨ ... ∨ vn)
                let mut clause_lits = vec![Literal::neg(result_var.0)];
                for &v in &sub_vars {
                    clause_lits.push(Literal::pos(v.0));
                }
                self.clauses.push(Clause::new(clause_lits));

                // (result_var ∨ ~v1) ∧ (result_var ∨ ~v2) ∧ ... ∧ (result_var ∨ ~vn)
                for &v in &sub_vars {
                    self.clauses.push(Clause::new(vec![
                        Literal::pos(result_var.0),
                        Literal::neg(v.0),
                    ]));
                }

                result_var
            }

            Formula::Implies(a, b) => {
                // a => b is equivalent to ~a ∨ b
                let not_a = Formula::not((**a).clone());
                let equiv = Formula::or(vec![not_a, (**b).clone()]);
                self.transform(&equiv)
            }

            Formula::Iff(a, b) => {
                // a <=> b is equivalent to (a => b) ∧ (b => a)
                let a_implies_b = Formula::implies((**a).clone(), (**b).clone());
                let b_implies_a = Formula::implies((**b).clone(), (**a).clone());
                let equiv = Formula::and(vec![a_implies_b, b_implies_a]);
                self.transform(&equiv)
            }

            Formula::Xor(a, b) => {
                // a XOR b is equivalent to (a ∨ b) ∧ (~a ∨ ~b)
                let a_or_b = Formula::or(vec![(**a).clone(), (**b).clone()]);
                let not_a_or_not_b = Formula::or(vec![
                    Formula::not((**a).clone()),
                    Formula::not((**b).clone()),
                ]);
                let equiv = Formula::and(vec![a_or_b, not_a_or_not_b]);
                self.transform(&equiv)
            }
        }
    }

    /// Get the generated clauses.
    #[must_use]
    pub fn clauses(&self) -> &[Clause] {
        &self.clauses
    }

    /// Take the generated clauses.
    pub fn take_clauses(self) -> Vec<Clause> {
        self.clauses
    }

    /// Get the next variable ID.
    #[must_use]
    pub fn next_var(&self) -> u32 {
        self.next_var
    }
}

/// CNF transformation statistics.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CnfStats {
    /// Number of clauses generated.
    pub clause_count: usize,
    /// Number of literals generated.
    pub literal_count: usize,
    /// Number of Tseitin variables introduced.
    pub tseitin_vars: usize,
    /// Maximum clause size.
    pub max_clause_size: usize,
    /// Average clause size.
    pub avg_clause_size: f64,
}

#[allow(dead_code)]
impl CnfStats {
    /// Compute statistics from a set of clauses.
    #[must_use]
    pub fn compute(clauses: &[Clause], original_vars: u32, final_vars: u32) -> Self {
        let clause_count = clauses.len();
        let literal_count: usize = clauses.iter().map(|c| c.literals.len()).sum();
        let max_clause_size = clauses.iter().map(|c| c.literals.len()).max().unwrap_or(0);
        let avg_clause_size = if clause_count > 0 {
            literal_count as f64 / clause_count as f64
        } else {
            0.0
        };
        let tseitin_vars = (final_vars - original_vars) as usize;

        Self {
            clause_count,
            literal_count,
            tseitin_vars,
            max_clause_size,
            avg_clause_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnf_var() {
        let mut transformer = CnfTransformer::new(1);
        let formula = Formula::var(1);
        let result = transformer.transform(&formula);

        assert_eq!(result, Var(1));
        assert_eq!(transformer.clauses().len(), 0);
    }

    #[test]
    fn test_cnf_not() {
        let mut transformer = CnfTransformer::new(2);
        let formula = Formula::not(Formula::var(1));
        let result = transformer.transform(&formula);

        assert_eq!(result, Var(2));
        assert_eq!(transformer.clauses().len(), 2);
    }

    #[test]
    fn test_cnf_and() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::and(vec![Formula::var(1), Formula::var(2)]);
        let result = transformer.transform(&formula);

        assert_eq!(result, Var(3));
        // Should generate 3 clauses for AND
        assert_eq!(transformer.clauses().len(), 3);
    }

    #[test]
    fn test_cnf_or() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::or(vec![Formula::var(1), Formula::var(2)]);
        let result = transformer.transform(&formula);

        assert_eq!(result, Var(3));
        // Should generate 3 clauses for OR
        assert_eq!(transformer.clauses().len(), 3);
    }

    #[test]
    fn test_cnf_implies() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::implies(Formula::var(1), Formula::var(2));
        let _result = transformer.transform(&formula);

        // Implication gets converted to disjunction
        assert!(!transformer.clauses().is_empty());
    }

    #[test]
    fn test_cnf_iff() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::iff(Formula::var(1), Formula::var(2));
        let _result = transformer.transform(&formula);

        // IFF gets converted to conjunction of implications
        assert!(!transformer.clauses().is_empty());
    }

    #[test]
    fn test_cnf_xor() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::xor(Formula::var(1), Formula::var(2));
        let _result = transformer.transform(&formula);

        // XOR gets converted
        assert!(!transformer.clauses().is_empty());
    }

    #[test]
    fn test_cnf_complex() {
        let mut transformer = CnfTransformer::new(4);

        // (v1 ∧ v2) ∨ (v2 ∧ v3)
        let left = Formula::and(vec![Formula::var(1), Formula::var(2)]);
        let right = Formula::and(vec![Formula::var(2), Formula::var(3)]);
        let formula = Formula::or(vec![left, right]);

        let _result = transformer.transform(&formula);

        // Should generate multiple clauses
        assert!(!transformer.clauses().is_empty());
    }

    #[test]
    fn test_cnf_stats() {
        let clauses = vec![
            Clause::new(vec![Literal::pos(1), Literal::neg(2)]),
            Clause::new(vec![Literal::pos(2), Literal::pos(3), Literal::neg(4)]),
            Clause::new(vec![Literal::neg(1)]),
        ];

        let stats = CnfStats::compute(&clauses, 4, 10);

        assert_eq!(stats.clause_count, 3);
        assert_eq!(stats.literal_count, 6);
        assert_eq!(stats.max_clause_size, 3);
        assert_eq!(stats.tseitin_vars, 6);
        assert!((stats.avg_clause_size - 2.0).abs() < 0.01);
    }
}
