//! Quantifier instantiation proof generation.
//!
//! This module provides infrastructure for generating proofs of quantifier
//! instantiation steps, which are crucial for SMT solving with quantifiers.

use crate::proof::{Proof, ProofNodeId};
use rustc_hash::FxHashMap;
use std::fmt;

/// A quantified variable.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantVar {
    /// Variable name.
    pub name: String,
    /// Sort (type) of the variable.
    pub sort: String,
}

#[allow(dead_code)]
impl QuantVar {
    /// Create a new quantified variable.
    #[must_use]
    pub fn new(name: impl Into<String>, sort: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sort: sort.into(),
        }
    }
}

impl fmt::Display for QuantVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {})", self.name, self.sort)
    }
}

/// A quantified formula.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantifiedFormula {
    /// Universal quantification: ∀x. φ(x)
    Forall {
        /// Bound variables.
        vars: Vec<QuantVar>,
        /// Body of the formula.
        body: String,
    },
    /// Existential quantification: ∃x. φ(x)
    Exists {
        /// Bound variables.
        vars: Vec<QuantVar>,
        /// Body of the formula.
        body: String,
    },
}

#[allow(dead_code)]
impl QuantifiedFormula {
    /// Create a universal quantification.
    #[must_use]
    pub fn forall(vars: Vec<QuantVar>, body: impl Into<String>) -> Self {
        Self::Forall {
            vars,
            body: body.into(),
        }
    }

    /// Create an existential quantification.
    #[must_use]
    pub fn exists(vars: Vec<QuantVar>, body: impl Into<String>) -> Self {
        Self::Exists {
            vars,
            body: body.into(),
        }
    }

    /// Get the bound variables.
    #[must_use]
    pub fn vars(&self) -> &[QuantVar] {
        match self {
            Self::Forall { vars, .. } | Self::Exists { vars, .. } => vars,
        }
    }

    /// Get the body of the formula.
    #[must_use]
    pub fn body(&self) -> &str {
        match self {
            Self::Forall { body, .. } | Self::Exists { body, .. } => body,
        }
    }

    /// Check if this is a universal quantification.
    #[must_use]
    pub fn is_forall(&self) -> bool {
        matches!(self, Self::Forall { .. })
    }

    /// Check if this is an existential quantification.
    #[must_use]
    pub fn is_exists(&self) -> bool {
        matches!(self, Self::Exists { .. })
    }
}

impl fmt::Display for QuantifiedFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forall { vars, body } => {
                write!(f, "(forall (")?;
                for (i, var) in vars.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", var)?;
                }
                write!(f, ") {})", body)
            }
            Self::Exists { vars, body } => {
                write!(f, "(exists (")?;
                for (i, var) in vars.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", var)?;
                }
                write!(f, ") {})", body)
            }
        }
    }
}

/// A substitution mapping variables to terms.
pub type Substitution = FxHashMap<String, String>;

/// An instantiation of a quantified formula.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Instantiation {
    /// The original quantified formula.
    pub formula: QuantifiedFormula,
    /// Substitution for bound variables.
    pub substitution: Substitution,
    /// The instantiated formula (with substitution applied).
    pub instantiated: String,
}

#[allow(dead_code)]
impl Instantiation {
    /// Create a new instantiation.
    #[must_use]
    pub fn new(
        formula: QuantifiedFormula,
        substitution: Substitution,
        instantiated: impl Into<String>,
    ) -> Self {
        Self {
            formula,
            substitution,
            instantiated: instantiated.into(),
        }
    }

    /// Apply the substitution to get the instantiated formula.
    ///
    /// This is a simple string-based substitution. A real implementation
    /// would use proper term substitution.
    #[must_use]
    pub fn apply_substitution(&self) -> String {
        let mut result = self.formula.body().to_string();
        for (var, term) in &self.substitution {
            // Simple string replacement (not semantically correct for all cases)
            result = result.replace(var, term);
        }
        result
    }
}

/// Proof recorder for quantifier instantiation.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct QuantifierProofRecorder {
    /// Recorded instantiations.
    instantiations: Vec<Instantiation>,
    /// Map from instantiation to proof node ID.
    inst_to_node: FxHashMap<String, ProofNodeId>,
}

#[allow(dead_code)]
impl QuantifierProofRecorder {
    /// Create a new quantifier proof recorder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instantiations: Vec::new(),
            inst_to_node: FxHashMap::default(),
        }
    }

    /// Record a forall instantiation.
    ///
    /// Given a formula ∀x. φ(x) and a term t, records the instantiation φ(t).
    pub fn record_forall_inst(
        &mut self,
        proof: &mut Proof,
        formula: QuantifiedFormula,
        substitution: Substitution,
        instantiated: impl Into<String>,
    ) -> ProofNodeId {
        let instantiated = instantiated.into();

        // Check if we already have this instantiation
        if let Some(&node_id) = self.inst_to_node.get(&instantiated) {
            return node_id;
        }

        let inst = Instantiation::new(formula, substitution, instantiated.clone());

        // Create a proof node for the instantiation
        let node_id = proof.add_inference(
            "forall_inst",
            Vec::new(),
            format!("(=> {} {})", inst.formula, inst.instantiated),
        );

        self.instantiations.push(inst);
        self.inst_to_node.insert(instantiated, node_id);

        node_id
    }

    /// Record an exists introduction.
    ///
    /// Given a formula ∃x. φ(x) and a witness term t where φ(t) holds,
    /// records the derivation of ∃x. φ(x).
    pub fn record_exists_intro(
        &mut self,
        proof: &mut Proof,
        formula: QuantifiedFormula,
        witness: Substitution,
        premise: ProofNodeId,
    ) -> ProofNodeId {
        let instantiated_body = formula.body().to_string();
        // Apply witness substitution
        let mut result = instantiated_body;
        for (var, term) in &witness {
            result = result.replace(var, term);
        }

        proof.add_inference("exists_intro", vec![premise], format!("{}", formula))
    }

    /// Record a skolemization step.
    ///
    /// Replaces an existential quantifier with a Skolem function/constant.
    pub fn record_skolemization(
        &mut self,
        proof: &mut Proof,
        formula: QuantifiedFormula,
        skolem_terms: Vec<String>,
    ) -> ProofNodeId {
        proof.add_inference(
            "skolem",
            Vec::new(),
            format!("(skolem {} {:?})", formula, skolem_terms),
        )
    }

    /// Get the number of recorded instantiations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.instantiations.len()
    }

    /// Check if there are no recorded instantiations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instantiations.is_empty()
    }

    /// Get all recorded instantiations.
    #[must_use]
    pub fn instantiations(&self) -> &[Instantiation] {
        &self.instantiations
    }

    /// Clear all recorded instantiations.
    pub fn clear(&mut self) {
        self.instantiations.clear();
        self.inst_to_node.clear();
    }
}

/// E-matching pattern for quantifier instantiation.
///
/// E-matching is a technique for finding instantiations of quantified formulas
/// by pattern matching against the ground terms in the current context.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EMatchPattern {
    /// Pattern to match (with variables).
    pub pattern: String,
    /// Variables in the pattern.
    pub vars: Vec<String>,
}

#[allow(dead_code)]
impl EMatchPattern {
    /// Create a new e-matching pattern.
    #[must_use]
    pub fn new(pattern: impl Into<String>, vars: Vec<String>) -> Self {
        Self {
            pattern: pattern.into(),
            vars,
        }
    }

    /// Check if this pattern matches a ground term.
    ///
    /// Returns the substitution if it matches, None otherwise.
    /// This is a placeholder - real e-matching is much more sophisticated.
    #[allow(dead_code)]
    #[must_use]
    pub fn matches(&self, _term: &str) -> Option<Substitution> {
        // Real implementation would:
        // 1. Parse both pattern and term into ASTs
        // 2. Perform structural matching
        // 3. Build substitution for pattern variables
        // For now, return None (no match)
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_var_creation() {
        let var = QuantVar::new("x", "Int");
        assert_eq!(var.name, "x");
        assert_eq!(var.sort, "Int");
        assert_eq!(format!("{}", var), "(x Int)");
    }

    #[test]
    fn test_forall_formula() {
        let vars = vec![QuantVar::new("x", "Int")];
        let formula = QuantifiedFormula::forall(vars, "(> x 0)");

        assert!(formula.is_forall());
        assert!(!formula.is_exists());
        assert_eq!(formula.body(), "(> x 0)");
        assert_eq!(formula.vars().len(), 1);

        let display = format!("{}", formula);
        assert!(display.contains("forall"));
        assert!(display.contains("x Int"));
    }

    #[test]
    fn test_exists_formula() {
        let vars = vec![QuantVar::new("y", "Real")];
        let formula = QuantifiedFormula::exists(vars, "(= y 5.0)");

        assert!(!formula.is_forall());
        assert!(formula.is_exists());
        assert_eq!(formula.body(), "(= y 5.0)");

        let display = format!("{}", formula);
        assert!(display.contains("exists"));
    }

    #[test]
    fn test_instantiation_creation() {
        let vars = vec![QuantVar::new("x", "Int")];
        let formula = QuantifiedFormula::forall(vars, "(> x 0)");

        let mut sub = Substitution::default();
        sub.insert("x".to_string(), "5".to_string());

        let inst = Instantiation::new(formula, sub, "(> 5 0)");
        assert_eq!(inst.instantiated, "(> 5 0)");
    }

    #[test]
    fn test_instantiation_apply() {
        let vars = vec![QuantVar::new("x", "Int")];
        let formula = QuantifiedFormula::forall(vars, "(> x 0)");

        let mut sub = Substitution::default();
        sub.insert("x".to_string(), "42".to_string());

        let inst = Instantiation::new(formula, sub, "(> 42 0)");
        let result = inst.apply_substitution();
        assert!(result.contains("42"));
    }

    #[test]
    fn test_quantifier_recorder_forall() {
        let mut recorder = QuantifierProofRecorder::new();
        let mut proof = Proof::new();
        proof.add_axiom("true");

        let vars = vec![QuantVar::new("x", "Int")];
        let formula = QuantifiedFormula::forall(vars, "(> x 0)");

        let mut sub = Substitution::default();
        sub.insert("x".to_string(), "5".to_string());

        let _node = recorder.record_forall_inst(&mut proof, formula, sub, "(> 5 0)");

        assert_eq!(recorder.len(), 1);
        assert!(!recorder.is_empty());
    }

    #[test]
    fn test_quantifier_recorder_exists() {
        let mut recorder = QuantifierProofRecorder::new();
        let mut proof = Proof::new();
        proof.add_axiom("(> 5 0)");
        let root = proof.root().unwrap();

        let vars = vec![QuantVar::new("x", "Int")];
        let formula = QuantifiedFormula::exists(vars, "(> x 0)");

        let mut witness = Substitution::default();
        witness.insert("x".to_string(), "5".to_string());

        let _node = recorder.record_exists_intro(&mut proof, formula, witness, root);

        assert_eq!(proof.node_count(), 2);
    }

    #[test]
    fn test_quantifier_recorder_skolem() {
        let mut recorder = QuantifierProofRecorder::new();
        let mut proof = Proof::new();
        proof.add_axiom("true");

        let vars = vec![QuantVar::new("x", "Int")];
        let formula = QuantifiedFormula::exists(vars, "(> x 0)");

        let _node = recorder.record_skolemization(&mut proof, formula, vec!["sk_x".to_string()]);

        assert_eq!(proof.node_count(), 2);
    }

    #[test]
    fn test_quantifier_recorder_dedup() {
        let mut recorder = QuantifierProofRecorder::new();
        let mut proof = Proof::new();
        proof.add_axiom("true");

        let vars = vec![QuantVar::new("x", "Int")];
        let formula = QuantifiedFormula::forall(vars.clone(), "(> x 0)");

        let mut sub = Substitution::default();
        sub.insert("x".to_string(), "5".to_string());

        let node1 =
            recorder.record_forall_inst(&mut proof, formula.clone(), sub.clone(), "(> 5 0)");
        let formula2 = QuantifiedFormula::forall(vars, "(> x 0)");
        let node2 = recorder.record_forall_inst(&mut proof, formula2, sub, "(> 5 0)");

        assert_eq!(node1, node2);
        assert_eq!(recorder.len(), 1);
    }

    #[test]
    fn test_quantifier_recorder_clear() {
        let mut recorder = QuantifierProofRecorder::new();
        let mut proof = Proof::new();
        proof.add_axiom("true");

        let vars = vec![QuantVar::new("x", "Int")];
        let formula = QuantifiedFormula::forall(vars, "(> x 0)");

        let mut sub = Substitution::default();
        sub.insert("x".to_string(), "5".to_string());

        recorder.record_forall_inst(&mut proof, formula, sub, "(> 5 0)");
        assert_eq!(recorder.len(), 1);

        recorder.clear();
        assert_eq!(recorder.len(), 0);
        assert!(recorder.is_empty());
    }

    #[test]
    fn test_ematch_pattern_creation() {
        let pattern = EMatchPattern::new("(f x)", vec!["x".to_string()]);
        assert_eq!(pattern.pattern, "(f x)");
        assert_eq!(pattern.vars.len(), 1);
    }

    #[test]
    fn test_multiple_vars() {
        let vars = vec![QuantVar::new("x", "Int"), QuantVar::new("y", "Int")];
        let formula = QuantifiedFormula::forall(vars, "(> (+ x y) 0)");

        assert_eq!(formula.vars().len(), 2);

        let display = format!("{}", formula);
        assert!(display.contains("(x Int)"));
        assert!(display.contains("(y Int)"));
    }
}
