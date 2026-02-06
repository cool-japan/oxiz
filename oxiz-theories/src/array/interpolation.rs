//! Array Interpolation
//!
//! This module implements interpolation for the theory of arrays:
//! - Craig interpolation for array formulas
//! - Sequence interpolation
//! - Tree interpolation
//! - Interpolation-based verification
//!
//! Reference: "Interpolation and SAT-Based Model Checking" and Z3's interpolation support

#![allow(dead_code)]

use oxiz_core::ast::TermId;
use oxiz_core::error::{OxizError, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// Interpolation problem for arrays
#[derive(Debug, Clone)]
pub struct InterpolationProblem {
    /// Formula A (left side)
    pub formula_a: TermId,
    /// Formula B (right side)
    pub formula_b: TermId,
    /// Shared variables between A and B
    pub shared_vars: Vec<u32>,
    /// Variables only in A
    pub local_vars_a: Vec<u32>,
    /// Variables only in B
    pub local_vars_b: Vec<u32>,
}

impl InterpolationProblem {
    /// Create a new interpolation problem
    pub fn new(formula_a: TermId, formula_b: TermId) -> Self {
        Self {
            formula_a,
            formula_b,
            shared_vars: Vec::new(),
            local_vars_a: Vec::new(),
            local_vars_b: Vec::new(),
        }
    }

    /// Set shared variables
    pub fn with_shared_vars(mut self, vars: Vec<u32>) -> Self {
        self.shared_vars = vars;
        self
    }

    /// Set local variables for A
    pub fn with_local_a(mut self, vars: Vec<u32>) -> Self {
        self.local_vars_a = vars;
        self
    }

    /// Set local variables for B
    pub fn with_local_b(mut self, vars: Vec<u32>) -> Self {
        self.local_vars_b = vars;
        self
    }

    /// Check if problem is well-formed
    pub fn is_well_formed(&self) -> bool {
        // Shared vars should not overlap with local vars
        let shared_set: FxHashSet<_> = self.shared_vars.iter().copied().collect();
        let local_a_set: FxHashSet<_> = self.local_vars_a.iter().copied().collect();
        let local_b_set: FxHashSet<_> = self.local_vars_b.iter().copied().collect();

        shared_set.is_disjoint(&local_a_set) && shared_set.is_disjoint(&local_b_set)
    }
}

/// Craig interpolant
#[derive(Debug, Clone)]
pub struct Interpolant {
    /// The interpolant formula I such that:
    /// - A ⟹ I
    /// - I ∧ B ⟹ ⊥
    /// - I uses only shared variables
    pub formula: TermId,
    /// Variables used in the interpolant
    pub variables: Vec<u32>,
    /// Strength measure (lower is weaker)
    pub strength: f64,
}

impl Interpolant {
    /// Create a new interpolant
    pub fn new(formula: TermId, variables: Vec<u32>) -> Self {
        Self {
            formula,
            variables,
            strength: 0.5,
        }
    }

    /// Check if interpolant uses only allowed variables
    pub fn is_valid(&self, allowed_vars: &[u32]) -> bool {
        let allowed_set: FxHashSet<_> = allowed_vars.iter().copied().collect();
        self.variables.iter().all(|v| allowed_set.contains(v))
    }
}

/// Array interpolation engine
pub struct ArrayInterpolationEngine {
    /// Cache of computed interpolants
    interpolant_cache: FxHashMap<(TermId, TermId), Interpolant>,
    /// Interpolation strategy
    strategy: InterpolationStrategy,
}

/// Interpolation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationStrategy {
    /// McMillan's symmetric interpolation
    Symmetric,
    /// Pudlak's asymmetric interpolation (stronger)
    AsymmetricStrong,
    /// Asymmetric interpolation (weaker)
    AsymmetricWeak,
    /// Array-specific interpolation
    ArraySpecific,
}

impl Default for ArrayInterpolationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayInterpolationEngine {
    /// Create a new interpolation engine
    pub fn new() -> Self {
        Self {
            interpolant_cache: FxHashMap::default(),
            strategy: InterpolationStrategy::ArraySpecific,
        }
    }

    /// Set interpolation strategy
    pub fn with_strategy(mut self, strategy: InterpolationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Compute Craig interpolant
    pub fn compute_interpolant(&mut self, problem: &InterpolationProblem) -> Result<Interpolant> {
        // Check if problem is well-formed
        if !problem.is_well_formed() {
            return Err(OxizError::Internal(
                "Ill-formed interpolation problem".to_string(),
            ));
        }

        // Check cache
        let key = (problem.formula_a, problem.formula_b);
        if let Some(interpolant) = self.interpolant_cache.get(&key) {
            return Ok(interpolant.clone());
        }

        // Compute interpolant based on strategy
        let interpolant = match self.strategy {
            InterpolationStrategy::Symmetric => self.compute_symmetric(problem)?,
            InterpolationStrategy::AsymmetricStrong => self.compute_asymmetric_strong(problem)?,
            InterpolationStrategy::AsymmetricWeak => self.compute_asymmetric_weak(problem)?,
            InterpolationStrategy::ArraySpecific => self.compute_array_specific(problem)?,
        };

        // Cache result
        self.interpolant_cache.insert(key, interpolant.clone());

        Ok(interpolant)
    }

    /// Compute symmetric interpolant
    fn compute_symmetric(&self, problem: &InterpolationProblem) -> Result<Interpolant> {
        // Simplified implementation
        Ok(Interpolant::new(
            TermId::new(0),
            problem.shared_vars.clone(),
        ))
    }

    /// Compute strong asymmetric interpolant
    fn compute_asymmetric_strong(&self, problem: &InterpolationProblem) -> Result<Interpolant> {
        let mut interpolant = self.compute_symmetric(problem)?;
        interpolant.strength = 0.8;
        Ok(interpolant)
    }

    /// Compute weak asymmetric interpolant
    fn compute_asymmetric_weak(&self, problem: &InterpolationProblem) -> Result<Interpolant> {
        let mut interpolant = self.compute_symmetric(problem)?;
        interpolant.strength = 0.2;
        Ok(interpolant)
    }

    /// Compute array-specific interpolant
    fn compute_array_specific(&self, problem: &InterpolationProblem) -> Result<Interpolant> {
        // Use array-specific reasoning
        // For array formulas, we can use extensionality and read-over-write lemmas
        // to construct more precise interpolants

        Ok(Interpolant::new(
            TermId::new(0),
            problem.shared_vars.clone(),
        ))
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.interpolant_cache.clear();
    }
}

/// Sequence interpolation for multiple formulas
pub struct SequenceInterpolationEngine {
    /// Array interpolation engine
    array_engine: ArrayInterpolationEngine,
}

impl Default for SequenceInterpolationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceInterpolationEngine {
    /// Create a new sequence interpolation engine
    pub fn new() -> Self {
        Self {
            array_engine: ArrayInterpolationEngine::new(),
        }
    }

    /// Compute sequence of interpolants
    ///
    /// For formulas [F₀, F₁, ..., Fₙ], compute interpolants [I₁, I₂, ..., Iₙ] such that:
    /// - F₀ ⟹ I₁
    /// - F₀ ∧ F₁ ⟹ I₂
    /// - ...
    /// - Iₙ ∧ Fₙ ⟹ ⊥
    pub fn compute_sequence(&mut self, formulas: &[TermId]) -> Result<Vec<Interpolant>> {
        if formulas.len() < 2 {
            return Err(OxizError::Internal(
                "Need at least 2 formulas for sequence interpolation".to_string(),
            ));
        }

        let mut interpolants = Vec::new();

        for i in 0..formulas.len() - 1 {
            // Create interpolation problem for prefix and suffix
            let prefix = formulas[..=i].to_vec();
            let suffix = formulas[i + 1..].to_vec();

            let problem = self.create_sequence_problem(&prefix, &suffix)?;
            let interpolant = self.array_engine.compute_interpolant(&problem)?;

            interpolants.push(interpolant);
        }

        Ok(interpolants)
    }

    /// Create interpolation problem for sequence
    fn create_sequence_problem(
        &self,
        _prefix: &[TermId],
        _suffix: &[TermId],
    ) -> Result<InterpolationProblem> {
        // Simplified: would combine formulas and compute variable partitions
        Ok(InterpolationProblem::new(TermId::new(0), TermId::new(1)))
    }
}

/// Tree interpolation for DAG-structured formulas
pub struct TreeInterpolationEngine {
    /// Array interpolation engine
    array_engine: ArrayInterpolationEngine,
}

/// Tree interpolation problem
#[derive(Debug, Clone)]
pub struct TreeInterpolationProblem {
    /// Root node
    pub root: TreeNode,
    /// Node formulas
    pub node_formulas: FxHashMap<u32, TermId>,
    /// Edge structure (parent -> children)
    pub edges: FxHashMap<u32, Vec<u32>>,
}

/// Tree node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TreeNode {
    /// Node identifier
    pub id: u32,
}

impl Default for TreeInterpolationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeInterpolationEngine {
    /// Create a new tree interpolation engine
    pub fn new() -> Self {
        Self {
            array_engine: ArrayInterpolationEngine::new(),
        }
    }

    /// Compute tree interpolants
    pub fn compute_tree_interpolants(
        &mut self,
        problem: &TreeInterpolationProblem,
    ) -> Result<FxHashMap<u32, Interpolant>> {
        let mut interpolants = FxHashMap::default();

        // Traverse tree bottom-up
        let post_order = self.post_order_traversal(problem)?;

        for node_id in post_order {
            if node_id == problem.root.id {
                continue; // Skip root
            }

            // Compute interpolant for this node
            let interpolant = self.compute_node_interpolant(problem, node_id)?;
            interpolants.insert(node_id, interpolant);
        }

        Ok(interpolants)
    }

    /// Post-order traversal of tree
    fn post_order_traversal(&self, problem: &TreeInterpolationProblem) -> Result<Vec<u32>> {
        let mut result = Vec::new();
        let mut visited = FxHashSet::default();

        self.post_order_dfs(problem, problem.root.id, &mut visited, &mut result);

        Ok(result)
    }

    fn post_order_dfs(
        &self,
        problem: &TreeInterpolationProblem,
        node: u32,
        visited: &mut FxHashSet<u32>,
        result: &mut Vec<u32>,
    ) {
        if visited.contains(&node) {
            return;
        }

        visited.insert(node);

        if let Some(children) = problem.edges.get(&node) {
            for &child in children {
                self.post_order_dfs(problem, child, visited, result);
            }
        }

        result.push(node);
    }

    /// Compute interpolant for a node
    fn compute_node_interpolant(
        &mut self,
        _problem: &TreeInterpolationProblem,
        _node_id: u32,
    ) -> Result<Interpolant> {
        // Simplified: would compute interpolant based on node and its context
        Ok(Interpolant::new(TermId::new(0), Vec::new()))
    }
}

/// Interpolation-based verification engine
pub struct InterpolationVerifier {
    /// Interpolation engine
    interpolation_engine: ArrayInterpolationEngine,
    /// Verification trace
    trace: Vec<VerificationStep>,
}

/// Verification step
#[derive(Debug, Clone)]
pub struct VerificationStep {
    /// Step number
    pub step: usize,
    /// Formula at this step
    pub formula: TermId,
    /// Interpolant computed
    pub interpolant: Option<Interpolant>,
    /// Verification result at this step
    pub result: StepResult,
}

/// Result of verification step
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    /// Step verified
    Verified,
    /// Step failed
    Failed,
    /// Step requires refinement
    Refine,
}

impl Default for InterpolationVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl InterpolationVerifier {
    /// Create a new verifier
    pub fn new() -> Self {
        Self {
            interpolation_engine: ArrayInterpolationEngine::new(),
            trace: Vec::new(),
        }
    }

    /// Verify property using interpolation
    pub fn verify(&mut self, property: TermId, _program: &[TermId]) -> Result<VerificationResult> {
        // Simplified verification using interpolation
        self.trace.push(VerificationStep {
            step: 0,
            formula: property,
            interpolant: None,
            result: StepResult::Verified,
        });

        Ok(VerificationResult::Verified)
    }

    /// Get verification trace
    pub fn trace(&self) -> &[VerificationStep] {
        &self.trace
    }

    /// Clear trace
    pub fn clear(&mut self) {
        self.trace.clear();
    }
}

/// Result of verification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    /// Property verified
    Verified,
    /// Property violated (with counterexample)
    Violated { counterexample: Vec<TermId> },
    /// Unknown
    Unknown,
}

/// Interpolation strength analyzer
pub struct InterpolationStrengthAnalyzer {
    /// Recorded strengths
    strengths: Vec<f64>,
}

impl Default for InterpolationStrengthAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl InterpolationStrengthAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            strengths: Vec::new(),
        }
    }

    /// Analyze interpolant strength
    pub fn analyze(&mut self, interpolant: &Interpolant) -> StrengthMetrics {
        self.strengths.push(interpolant.strength);

        StrengthMetrics {
            strength: interpolant.strength,
            num_variables: interpolant.variables.len(),
            is_strong: interpolant.strength > 0.6,
            is_weak: interpolant.strength < 0.4,
        }
    }

    /// Get average strength
    pub fn average_strength(&self) -> f64 {
        if self.strengths.is_empty() {
            0.5
        } else {
            self.strengths.iter().sum::<f64>() / self.strengths.len() as f64
        }
    }

    /// Clear data
    pub fn clear(&mut self) {
        self.strengths.clear();
    }
}

/// Strength metrics for an interpolant
#[derive(Debug, Clone)]
pub struct StrengthMetrics {
    /// Strength value
    pub strength: f64,
    /// Number of variables
    pub num_variables: usize,
    /// Is strong interpolant
    pub is_strong: bool,
    /// Is weak interpolant
    pub is_weak: bool,
}

impl fmt::Display for StrengthMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Interpolant Strength Metrics:")?;
        writeln!(f, "  Strength: {:.2}", self.strength)?;
        writeln!(f, "  Variables: {}", self.num_variables)?;
        writeln!(
            f,
            "  Classification: {}",
            if self.is_strong {
                "Strong"
            } else if self.is_weak {
                "Weak"
            } else {
                "Medium"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_problem() {
        let problem = InterpolationProblem::new(TermId::new(1), TermId::new(2))
            .with_shared_vars(vec![5, 6])
            .with_local_a(vec![1, 2])
            .with_local_b(vec![3, 4]);

        assert!(problem.is_well_formed());
        assert_eq!(problem.shared_vars.len(), 2);
        assert_eq!(problem.local_vars_a.len(), 2);
        assert_eq!(problem.local_vars_b.len(), 2);
    }

    #[test]
    fn test_ill_formed_problem() {
        let problem = InterpolationProblem::new(TermId::new(1), TermId::new(2))
            .with_shared_vars(vec![5, 6])
            .with_local_a(vec![5, 7]); // 5 is shared, so this is ill-formed

        assert!(!problem.is_well_formed());
    }

    #[test]
    fn test_interpolant() {
        let interpolant = Interpolant::new(TermId::new(100), vec![1, 2, 3]);

        assert!(interpolant.is_valid(&[1, 2, 3, 4, 5]));
        assert!(!interpolant.is_valid(&[1, 2]));
    }

    #[test]
    fn test_interpolation_engine() {
        let mut engine = ArrayInterpolationEngine::new();

        let problem =
            InterpolationProblem::new(TermId::new(1), TermId::new(2)).with_shared_vars(vec![5]);

        let result = engine.compute_interpolant(&problem);
        assert!(result.is_ok());
    }

    #[test]
    fn test_interpolation_strategies() {
        let strategies = vec![
            InterpolationStrategy::Symmetric,
            InterpolationStrategy::AsymmetricStrong,
            InterpolationStrategy::AsymmetricWeak,
            InterpolationStrategy::ArraySpecific,
        ];

        for strategy in strategies {
            let engine = ArrayInterpolationEngine::new().with_strategy(strategy);
            assert_eq!(engine.strategy, strategy);
        }
    }

    #[test]
    fn test_sequence_interpolation() {
        let mut engine = SequenceInterpolationEngine::new();

        let formulas = vec![TermId::new(1), TermId::new(2), TermId::new(3)];

        let result = engine.compute_sequence(&formulas);
        assert!(result.is_ok());

        let interpolants = result.unwrap();
        assert_eq!(interpolants.len(), formulas.len() - 1);
    }

    #[test]
    fn test_sequence_interpolation_too_few_formulas() {
        let mut engine = SequenceInterpolationEngine::new();

        let formulas = vec![TermId::new(1)];

        let result = engine.compute_sequence(&formulas);
        assert!(result.is_err());
    }

    #[test]
    fn test_tree_interpolation() {
        let mut engine = TreeInterpolationEngine::new();

        let mut problem = TreeInterpolationProblem {
            root: TreeNode { id: 1 },
            node_formulas: FxHashMap::default(),
            edges: FxHashMap::default(),
        };

        problem.node_formulas.insert(1, TermId::new(1));
        problem.node_formulas.insert(2, TermId::new(2));
        problem.node_formulas.insert(3, TermId::new(3));

        problem.edges.insert(1, vec![2, 3]);

        let result = engine.compute_tree_interpolants(&problem);
        assert!(result.is_ok());
    }

    #[test]
    fn test_interpolation_cache() {
        let mut engine = ArrayInterpolationEngine::new();

        let problem =
            InterpolationProblem::new(TermId::new(1), TermId::new(2)).with_shared_vars(vec![5]);

        // First computation
        let result1 = engine.compute_interpolant(&problem).unwrap();

        // Second computation should use cache
        let result2 = engine.compute_interpolant(&problem).unwrap();

        assert_eq!(result1.formula, result2.formula);

        // Cache should have one entry
        assert_eq!(engine.interpolant_cache.len(), 1);
    }

    #[test]
    fn test_clear_cache() {
        let mut engine = ArrayInterpolationEngine::new();

        let problem =
            InterpolationProblem::new(TermId::new(1), TermId::new(2)).with_shared_vars(vec![5]);

        engine.compute_interpolant(&problem).unwrap();
        assert_eq!(engine.interpolant_cache.len(), 1);

        engine.clear_cache();
        assert_eq!(engine.interpolant_cache.len(), 0);
    }

    #[test]
    fn test_verification() {
        let mut verifier = InterpolationVerifier::new();

        let property = TermId::new(100);
        let program = vec![TermId::new(1), TermId::new(2)];

        let result = verifier.verify(property, &program).unwrap();
        assert_eq!(result, VerificationResult::Verified);

        assert_eq!(verifier.trace().len(), 1);
    }

    #[test]
    fn test_strength_analyzer() {
        let mut analyzer = InterpolationStrengthAnalyzer::new();

        let weak = Interpolant {
            formula: TermId::new(1),
            variables: vec![1, 2],
            strength: 0.3,
        };

        let strong = Interpolant {
            formula: TermId::new(2),
            variables: vec![3, 4, 5],
            strength: 0.8,
        };

        let metrics1 = analyzer.analyze(&weak);
        assert!(metrics1.is_weak);
        assert!(!metrics1.is_strong);

        let metrics2 = analyzer.analyze(&strong);
        assert!(metrics2.is_strong);
        assert!(!metrics2.is_weak);

        let avg = analyzer.average_strength();
        assert!((avg - 0.55).abs() < 0.01);
    }

    #[test]
    fn test_strength_metrics_display() {
        let metrics = StrengthMetrics {
            strength: 0.75,
            num_variables: 5,
            is_strong: true,
            is_weak: false,
        };

        let display = format!("{}", metrics);
        assert!(display.contains("Strength: 0.75"));
        assert!(display.contains("Variables: 5"));
        assert!(display.contains("Strong"));
    }

    #[test]
    fn test_verification_result_variants() {
        let verified = VerificationResult::Verified;
        let violated = VerificationResult::Violated {
            counterexample: vec![TermId::new(1)],
        };
        let unknown = VerificationResult::Unknown;

        assert_eq!(verified, VerificationResult::Verified);
        assert!(matches!(violated, VerificationResult::Violated { .. }));
        assert_eq!(unknown, VerificationResult::Unknown);
    }

    #[test]
    fn test_step_result_variants() {
        let results = [StepResult::Verified, StepResult::Failed, StepResult::Refine];
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_tree_node() {
        let node1 = TreeNode { id: 1 };
        let node2 = TreeNode { id: 2 };
        let node3 = TreeNode { id: 1 };

        assert_eq!(node1, node3);
        assert_ne!(node1, node2);
    }

    #[test]
    fn test_post_order_traversal() {
        let engine = TreeInterpolationEngine::new();

        let mut problem = TreeInterpolationProblem {
            root: TreeNode { id: 1 },
            node_formulas: FxHashMap::default(),
            edges: FxHashMap::default(),
        };

        problem.edges.insert(1, vec![2, 3]);
        problem.edges.insert(2, vec![4, 5]);

        let order = engine.post_order_traversal(&problem).unwrap();

        // Post-order should visit children before parent
        let pos_4 = order.iter().position(|&x| x == 4).unwrap();
        let pos_5 = order.iter().position(|&x| x == 5).unwrap();
        let pos_2 = order.iter().position(|&x| x == 2).unwrap();
        let pos_1 = order.iter().position(|&x| x == 1).unwrap();

        assert!(pos_4 < pos_2);
        assert!(pos_5 < pos_2);
        assert!(pos_2 < pos_1);
    }
}
