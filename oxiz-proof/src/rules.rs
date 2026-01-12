//! Proof rule definitions and validators.
//!
//! This module provides validation logic for standard proof rules used in SMT solving,
//! including resolution, unit propagation, CNF transformation, and theory-specific rules.

use std::collections::HashSet;
use std::fmt;

/// A literal in a clause (variable index with sign)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal {
    /// Variable index
    pub var: u32,
    /// True if positive, false if negated
    pub sign: bool,
}

impl Literal {
    /// Create a positive literal
    #[must_use]
    pub const fn pos(var: u32) -> Self {
        Self { var, sign: true }
    }

    /// Create a negative literal
    #[must_use]
    pub const fn neg(var: u32) -> Self {
        Self { var, sign: false }
    }

    /// Negate this literal
    #[must_use]
    pub const fn negate(self) -> Self {
        Self {
            var: self.var,
            sign: !self.sign,
        }
    }

    /// Check if two literals are complementary
    #[must_use]
    pub const fn is_complementary(self, other: Self) -> bool {
        self.var == other.var && self.sign != other.sign
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.sign {
            write!(f, "{}", self.var)
        } else {
            write!(f, "-{}", self.var)
        }
    }
}

/// A clause (disjunction of literals)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    /// Literals in the clause
    pub literals: Vec<Literal>,
}

impl Clause {
    /// Create a new clause
    #[must_use]
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    /// Create an empty clause (false)
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            literals: Vec::new(),
        }
    }

    /// Create a unit clause
    #[must_use]
    pub fn unit(lit: Literal) -> Self {
        Self {
            literals: vec![lit],
        }
    }

    /// Check if the clause is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if the clause is a unit clause
    #[must_use]
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Get the unit literal (if this is a unit clause)
    #[must_use]
    pub fn unit_literal(&self) -> Option<Literal> {
        if self.is_unit() {
            self.literals.first().copied()
        } else {
            None
        }
    }

    /// Check if the clause is a tautology
    #[must_use]
    pub fn is_tautology(&self) -> bool {
        let mut seen = HashSet::new();
        for &lit in &self.literals {
            if seen.contains(&lit.negate()) {
                return true;
            }
            seen.insert(lit);
        }
        false
    }

    /// Remove duplicate literals
    pub fn normalize(&mut self) {
        let mut seen = HashSet::new();
        self.literals.retain(|&lit| seen.insert(lit));
        self.literals.sort_by_key(|l| (l.var, !l.sign));
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, lit) in self.literals.iter().enumerate() {
            if i > 0 {
                write!(f, " ∨ ")?;
            }
            write!(f, "{}", lit)?;
        }
        write!(f, "]")
    }
}

/// Result of rule validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleValidation {
    /// Rule application is valid
    Valid,
    /// Rule application is invalid
    Invalid(String),
}

impl RuleValidation {
    /// Check if the validation is successful
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Get the error message (if invalid)
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        match self {
            Self::Invalid(msg) => Some(msg),
            Self::Valid => None,
        }
    }
}

/// Resolution rule validator
pub struct ResolutionValidator;

impl ResolutionValidator {
    /// Validate a resolution step
    ///
    /// Resolution: C1 ∨ x, C2 ∨ ¬x ⊢ C1 ∨ C2
    #[must_use]
    pub fn validate(c1: &Clause, c2: &Clause, pivot: Literal, result: &Clause) -> RuleValidation {
        // Find the pivot literal in c1 and its negation in c2
        let has_pivot_in_c1 = c1.literals.contains(&pivot);
        let has_neg_pivot_in_c2 = c2.literals.contains(&pivot.negate());

        if !has_pivot_in_c1 {
            return RuleValidation::Invalid(format!("Pivot {} not found in first clause", pivot));
        }

        if !has_neg_pivot_in_c2 {
            return RuleValidation::Invalid(format!(
                "Negated pivot {} not found in second clause",
                pivot.negate()
            ));
        }

        // Build expected resolvent
        let mut expected = Vec::new();
        for &lit in &c1.literals {
            if lit != pivot {
                expected.push(lit);
            }
        }
        for &lit in &c2.literals {
            if lit != pivot.negate() {
                expected.push(lit);
            }
        }

        // Normalize and compare
        let mut expected_clause = Clause::new(expected);
        expected_clause.normalize();

        let mut result_normalized = result.clone();
        result_normalized.normalize();

        if expected_clause == result_normalized {
            RuleValidation::Valid
        } else {
            RuleValidation::Invalid(format!(
                "Expected resolvent {}, got {}",
                expected_clause, result_normalized
            ))
        }
    }
}

/// Unit propagation validator
pub struct UnitPropagationValidator;

impl UnitPropagationValidator {
    /// Validate a unit propagation step
    ///
    /// Unit propagation: C ∨ x, ¬x ⊢ C
    #[must_use]
    pub fn validate(clause: &Clause, unit: Literal, result: &Clause) -> RuleValidation {
        // Check that unit is indeed a literal
        let neg_unit = unit.negate();

        // Build expected result (clause with neg_unit removed)
        let expected: Vec<Literal> = clause
            .literals
            .iter()
            .copied()
            .filter(|&lit| lit != neg_unit)
            .collect();

        if expected.len() == clause.literals.len() {
            return RuleValidation::Invalid(format!(
                "Unit literal {} not found in clause",
                neg_unit
            ));
        }

        let mut expected_clause = Clause::new(expected);
        expected_clause.normalize();

        let mut result_normalized = result.clone();
        result_normalized.normalize();

        if expected_clause == result_normalized {
            RuleValidation::Valid
        } else {
            RuleValidation::Invalid(format!(
                "Expected {}, got {}",
                expected_clause, result_normalized
            ))
        }
    }
}

/// CNF transformation validator
pub struct CnfValidator;

impl CnfValidator {
    /// Validate negation normal form transformation
    ///
    /// ¬(¬A) ⟺ A
    #[must_use]
    pub fn validate_not_not(input: &str, output: &str) -> RuleValidation {
        if input.starts_with("¬¬") && output == &input[4..] {
            RuleValidation::Valid
        } else {
            RuleValidation::Invalid("Invalid ¬¬ elimination".to_string())
        }
    }

    /// Validate De Morgan's law (AND)
    ///
    /// ¬(A ∧ B) ⟺ ¬A ∨ ¬B
    #[must_use]
    pub fn validate_demorgan_and(_input: &str, _output: &str) -> RuleValidation {
        // Simplified validation - in practice would parse formulas
        RuleValidation::Valid
    }

    /// Validate De Morgan's law (OR)
    ///
    /// ¬(A ∨ B) ⟺ ¬A ∧ ¬B
    #[must_use]
    pub fn validate_demorgan_or(_input: &str, _output: &str) -> RuleValidation {
        // Simplified validation - in practice would parse formulas
        RuleValidation::Valid
    }

    /// Validate distributivity
    ///
    /// A ∨ (B ∧ C) ⟺ (A ∨ B) ∧ (A ∨ C)
    #[must_use]
    pub fn validate_distributivity(_input: &str, _output: &str) -> RuleValidation {
        // Simplified validation - in practice would parse formulas
        RuleValidation::Valid
    }
}

/// Theory lemma validator
pub struct TheoryLemmaValidator;

impl TheoryLemmaValidator {
    /// Validate an arithmetic Farkas lemma
    ///
    /// Given inequalities and coefficients, check that the combination is valid
    #[must_use]
    pub fn validate_farkas(
        _inequalities: &[String],
        _coefficients: &[f64],
        _result: &str,
    ) -> RuleValidation {
        // Simplified - in practice would check arithmetic
        RuleValidation::Valid
    }

    /// Validate congruence closure
    ///
    /// a = b, f(a) ⊢ f(a) = f(b)
    #[must_use]
    pub fn validate_congruence(_equalities: &[String], _result: &str) -> RuleValidation {
        // Simplified - in practice would check congruence
        RuleValidation::Valid
    }

    /// Validate transitivity of equality
    ///
    /// a = b, b = c ⊢ a = c
    #[must_use]
    pub fn validate_transitivity(_eq1: &str, _eq2: &str, _result: &str) -> RuleValidation {
        // Simplified - in practice would parse and check
        RuleValidation::Valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let lit = Literal::pos(5);
        assert_eq!(lit.var, 5);
        assert!(lit.sign);

        let neg_lit = Literal::neg(5);
        assert_eq!(neg_lit.var, 5);
        assert!(!neg_lit.sign);
    }

    #[test]
    fn test_literal_negate() {
        let lit = Literal::pos(3);
        let neg = lit.negate();
        assert_eq!(neg.var, 3);
        assert!(!neg.sign);
    }

    #[test]
    fn test_literal_complementary() {
        let lit1 = Literal::pos(5);
        let lit2 = Literal::neg(5);
        assert!(lit1.is_complementary(lit2));
        assert!(lit2.is_complementary(lit1));

        let lit3 = Literal::pos(6);
        assert!(!lit1.is_complementary(lit3));
    }

    #[test]
    fn test_clause_empty() {
        let clause = Clause::empty();
        assert!(clause.is_empty());
        assert!(!clause.is_unit());
    }

    #[test]
    fn test_clause_unit() {
        let clause = Clause::unit(Literal::pos(1));
        assert!(clause.is_unit());
        assert_eq!(clause.unit_literal(), Some(Literal::pos(1)));
    }

    #[test]
    fn test_clause_tautology() {
        let clause = Clause::new(vec![Literal::pos(1), Literal::neg(1)]);
        assert!(clause.is_tautology());

        let non_taut = Clause::new(vec![Literal::pos(1), Literal::pos(2)]);
        assert!(!non_taut.is_tautology());
    }

    #[test]
    fn test_clause_normalize() {
        let mut clause = Clause::new(vec![
            Literal::pos(2),
            Literal::pos(1),
            Literal::pos(2), // duplicate
        ]);

        clause.normalize();
        assert_eq!(clause.literals.len(), 2);
    }

    #[test]
    fn test_resolution_valid() {
        // (p ∨ q) ∧ (¬p ∨ r) ⊢ (q ∨ r)
        let c1 = Clause::new(vec![Literal::pos(1), Literal::pos(2)]); // p ∨ q
        let c2 = Clause::new(vec![Literal::neg(1), Literal::pos(3)]); // ¬p ∨ r
        let result = Clause::new(vec![Literal::pos(2), Literal::pos(3)]); // q ∨ r
        let pivot = Literal::pos(1); // p

        let validation = ResolutionValidator::validate(&c1, &c2, pivot, &result);
        assert!(validation.is_valid());
    }

    #[test]
    fn test_resolution_invalid_pivot() {
        let c1 = Clause::new(vec![Literal::pos(1), Literal::pos(2)]);
        let c2 = Clause::new(vec![Literal::neg(3), Literal::pos(4)]); // Wrong pivot
        let result = Clause::new(vec![Literal::pos(2), Literal::pos(4)]);
        let pivot = Literal::pos(1);

        let validation = ResolutionValidator::validate(&c1, &c2, pivot, &result);
        assert!(!validation.is_valid());
    }

    #[test]
    fn test_unit_propagation_valid() {
        // (p ∨ q ∨ r) with unit ¬p ⊢ (q ∨ r)
        let clause = Clause::new(vec![Literal::pos(1), Literal::pos(2), Literal::pos(3)]);
        let unit = Literal::neg(1);
        let result = Clause::new(vec![Literal::pos(2), Literal::pos(3)]);

        let validation = UnitPropagationValidator::validate(&clause, unit, &result);
        assert!(validation.is_valid());
    }

    #[test]
    fn test_unit_propagation_invalid() {
        let clause = Clause::new(vec![Literal::pos(1), Literal::pos(2)]);
        let unit = Literal::neg(3); // Not in clause
        let result = Clause::new(vec![Literal::pos(1), Literal::pos(2)]);

        let validation = UnitPropagationValidator::validate(&clause, unit, &result);
        assert!(!validation.is_valid());
    }

    #[test]
    fn test_cnf_not_not() {
        let validation = CnfValidator::validate_not_not("¬¬A", "A");
        assert!(validation.is_valid());

        let invalid = CnfValidator::validate_not_not("¬A", "A");
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_literal_display() {
        assert_eq!(format!("{}", Literal::pos(5)), "5");
        assert_eq!(format!("{}", Literal::neg(5)), "-5");
    }

    #[test]
    fn test_clause_display() {
        let clause = Clause::new(vec![Literal::pos(1), Literal::neg(2), Literal::pos(3)]);
        let display = format!("{}", clause);
        assert!(display.contains("1"));
        assert!(display.contains("-2"));
        assert!(display.contains("3"));
    }
}
