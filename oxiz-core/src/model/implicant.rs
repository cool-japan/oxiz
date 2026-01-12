//! Prime Implicant Extraction
//!
//! Extracts minimal satisfying assignments from models.

use super::{Model, ModelEvaluator, Value};
use crate::ast::{TermId, TermKind, TermManager};
use std::collections::{HashMap, HashSet};

/// Configuration for implicant extraction
#[derive(Debug, Clone)]
pub struct ImplicantConfig {
    /// Maximum iterations for minimization
    pub max_iterations: u32,
    /// Try removing literals in order
    pub ordered_removal: bool,
    /// Use caching during evaluation
    pub use_cache: bool,
}

impl Default for ImplicantConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            ordered_removal: true,
            use_cache: true,
        }
    }
}

/// A prime implicant: minimal set of literals that implies the formula
#[derive(Debug, Clone)]
pub struct PrimeImplicant {
    /// Literals in the implicant (term -> value)
    pub literals: HashMap<TermId, Value>,
    /// Original formula
    pub formula: TermId,
}

impl PrimeImplicant {
    /// Create a new prime implicant
    pub fn new(literals: HashMap<TermId, Value>, formula: TermId) -> Self {
        Self { literals, formula }
    }

    /// Number of literals
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Get value of a term
    pub fn get(&self, term: TermId) -> Option<&Value> {
        self.literals.get(&term)
    }

    /// Iterate over literals
    pub fn iter(&self) -> impl Iterator<Item = (&TermId, &Value)> {
        self.literals.iter()
    }

    /// Convert to model
    pub fn to_model(&self) -> Model {
        let mut model = Model::new();
        for (&term, value) in &self.literals {
            model.assign(term, value.clone());
        }
        model
    }
}

/// Implicant extractor
#[derive(Debug)]
pub struct ImplicantExtractor {
    config: ImplicantConfig,
    /// Statistics
    stats: ImplicantStats,
}

/// Statistics for implicant extraction
#[derive(Debug, Clone, Default)]
pub struct ImplicantStats {
    /// Number of extractions performed
    pub extractions: u64,
    /// Number of literals removed during minimization
    pub literals_removed: u64,
    /// Number of literals in original models
    pub original_literals: u64,
    /// Number of literals in final implicants
    pub final_literals: u64,
}

impl ImplicantStats {
    /// Compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.original_literals == 0 {
            0.0
        } else {
            1.0 - (self.final_literals as f64 / self.original_literals as f64)
        }
    }
}

impl ImplicantExtractor {
    /// Create a new implicant extractor
    pub fn new() -> Self {
        Self {
            config: ImplicantConfig::default(),
            stats: ImplicantStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ImplicantConfig) -> Self {
        Self {
            config,
            stats: ImplicantStats::default(),
        }
    }

    /// Extract a prime implicant from a model
    pub fn extract(
        &mut self,
        model: &Model,
        formula: TermId,
        manager: &TermManager,
    ) -> Option<PrimeImplicant> {
        self.stats.extractions += 1;

        // Collect relevant literals from the model
        let relevant = self.collect_relevant_literals(model, formula, manager);
        self.stats.original_literals += relevant.len() as u64;

        // Minimize the literal set
        let minimized = self.minimize(relevant, formula, manager);
        self.stats.final_literals += minimized.len() as u64;
        self.stats.literals_removed += (model.len() as u64).saturating_sub(minimized.len() as u64);

        Some(PrimeImplicant::new(minimized, formula))
    }

    /// Collect literals that are relevant to the formula
    fn collect_relevant_literals(
        &self,
        model: &Model,
        formula: TermId,
        manager: &TermManager,
    ) -> HashMap<TermId, Value> {
        let mut relevant = HashMap::new();
        let mut visited = HashSet::new();
        let mut worklist = vec![formula];

        while let Some(term) = worklist.pop() {
            if visited.contains(&term) {
                continue;
            }
            visited.insert(term);

            // Get the term from manager
            if let Some(t) = manager.get(term) {
                match &t.kind {
                    TermKind::Var(_) => {
                        if let Some(value) = model.get(term) {
                            relevant.insert(term, value.clone());
                        }
                    }
                    TermKind::Ite(cond, then_branch, else_branch) => {
                        // For ITE, only follow the taken branch
                        worklist.push(*cond);
                        if let Some(Value::Bool(true)) = model.get(*cond) {
                            worklist.push(*then_branch);
                        } else {
                            worklist.push(*else_branch);
                        }
                    }
                    _ => {
                        // Add children based on term kind
                        self.add_children_to_worklist(&t.kind, &mut worklist);
                    }
                }
            }
        }

        relevant
    }

    /// Add children of a term to the worklist
    fn add_children_to_worklist(&self, kind: &TermKind, worklist: &mut Vec<TermId>) {
        match kind {
            TermKind::Not(a) | TermKind::Neg(a) | TermKind::BvNot(a) => {
                worklist.push(*a);
            }
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::Distinct(args) => {
                for arg in args.iter() {
                    worklist.push(*arg);
                }
            }
            TermKind::Xor(a, b)
            | TermKind::Implies(a, b)
            | TermKind::Eq(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::BvAnd(a, b)
            | TermKind::BvOr(a, b)
            | TermKind::BvXor(a, b)
            | TermKind::BvAdd(a, b)
            | TermKind::BvSub(a, b)
            | TermKind::BvMul(a, b)
            | TermKind::BvConcat(a, b) => {
                worklist.push(*a);
                worklist.push(*b);
            }
            TermKind::Ite(a, b, c) => {
                worklist.push(*a);
                worklist.push(*b);
                worklist.push(*c);
            }
            _ => {}
        }
    }

    /// Minimize a set of literals while preserving satisfiability
    fn minimize(
        &self,
        mut literals: HashMap<TermId, Value>,
        formula: TermId,
        manager: &TermManager,
    ) -> HashMap<TermId, Value> {
        let mut iterations = 0;
        let mut changed = true;

        while changed && iterations < self.config.max_iterations {
            changed = false;
            iterations += 1;

            // Try removing each literal
            let terms: Vec<_> = literals.keys().copied().collect();
            for term in terms {
                if let Some(value) = literals.remove(&term) {
                    // Check if formula is still satisfied without this literal
                    if !self.check_satisfied(&literals, formula, manager) {
                        // Need this literal, put it back
                        literals.insert(term, value);
                    } else {
                        // Can remove this literal
                        changed = true;
                    }
                }
            }
        }

        literals
    }

    /// Check if formula is satisfied under given literals
    fn check_satisfied(
        &self,
        literals: &HashMap<TermId, Value>,
        formula: TermId,
        manager: &TermManager,
    ) -> bool {
        let mut model = Model::new();
        for (&term, value) in literals {
            model.assign(term, value.clone());
        }

        let mut evaluator = ModelEvaluator::new(&model);
        matches!(
            evaluator.eval(formula, manager),
            super::EvalResult::Ok(Value::Bool(true))
        )
    }

    /// Get statistics
    pub fn stats(&self) -> &ImplicantStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ImplicantStats::default();
    }
}

impl Default for ImplicantExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicant_config() {
        let config = ImplicantConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert!(config.ordered_removal);
        assert!(config.use_cache);
    }

    #[test]
    fn test_prime_implicant() {
        let mut literals = HashMap::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);
        literals.insert(t1, Value::Bool(true));
        literals.insert(t2, Value::Int(42));

        let formula = TermId::from(100u32);
        let implicant = PrimeImplicant::new(literals, formula);

        assert_eq!(implicant.len(), 2);
        assert!(!implicant.is_empty());
        assert_eq!(implicant.get(t1), Some(&Value::Bool(true)));
        assert_eq!(implicant.get(t2), Some(&Value::Int(42)));
    }

    #[test]
    fn test_implicant_to_model() {
        let mut literals = HashMap::new();
        let t1 = TermId::from(1u32);
        literals.insert(t1, Value::Bool(true));

        let formula = TermId::from(100u32);
        let implicant = PrimeImplicant::new(literals, formula);

        let model = implicant.to_model();
        assert_eq!(model.get(t1), Some(&Value::Bool(true)));
    }

    #[test]
    fn test_extractor_creation() {
        let extractor = ImplicantExtractor::new();
        assert_eq!(extractor.stats().extractions, 0);
    }

    #[test]
    fn test_stats_compression_ratio() {
        let mut stats = ImplicantStats::default();
        stats.original_literals = 10;
        stats.final_literals = 3;

        assert!((stats.compression_ratio() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_empty_compression_ratio() {
        let stats = ImplicantStats::default();
        assert_eq!(stats.compression_ratio(), 0.0);
    }
}
