//! Array Quantifier Elimination
//!
//! This module implements quantifier elimination techniques for the theory of arrays,
//! including:
//! - Index quantifier elimination using Skolemization
//! - Element quantifier patterns and instantiation
//! - Model-based projection for arrays
//! - Decidable array fragments (e.g., Bradley-Manna-Sipma)
//!
//! Reference: Z3's qe_array_plugin.cpp and mbp_arrays.cpp

#![allow(missing_docs)]
#![allow(dead_code)]

use oxiz_core::ast::TermId;
use oxiz_core::error::{OxizError, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// Quantifier type in array context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantifierType {
    /// Universal quantifier (forall)
    Forall,
    /// Existential quantifier (exists)
    Exists,
}

impl fmt::Display for QuantifierType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantifierType::Forall => write!(f, "∀"),
            QuantifierType::Exists => write!(f, "∃"),
        }
    }
}

/// Quantified variable in array formulas
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantifiedVar {
    /// Variable identifier
    pub var_id: u32,
    /// Variable name
    pub name: String,
    /// Sort of the variable
    pub sort: u32,
    /// Quantifier type
    pub qtype: QuantifierType,
}

impl QuantifiedVar {
    /// Create a new quantified variable
    pub fn new(var_id: u32, name: String, sort: u32, qtype: QuantifierType) -> Self {
        Self {
            var_id,
            name,
            sort,
            qtype,
        }
    }

    /// Check if this is universally quantified
    pub fn is_universal(&self) -> bool {
        self.qtype == QuantifierType::Forall
    }

    /// Check if this is existentially quantified
    pub fn is_existential(&self) -> bool {
        self.qtype == QuantifierType::Exists
    }
}

/// Array quantifier pattern for instantiation
#[derive(Debug, Clone)]
pub enum ArrayQuantifierPattern {
    /// Select pattern: ∀i. select(A, i) = f(i)
    SelectPattern {
        array_var: u32,
        index_var: u32,
        value_expr: TermId,
    },
    /// Store pattern: ∃A. A = store(B, i, v)
    StorePattern {
        array_var: u32,
        base_array: u32,
        index: u32,
        value: u32,
    },
    /// Extensionality pattern: (∀i. select(A, i) = select(B, i)) → A = B
    ExtensionalityPattern {
        array1: u32,
        array2: u32,
        index_var: u32,
    },
    /// Read-over-write pattern: select(store(A, i, v), i) = v
    ReadOverWritePattern { array: u32, index: u32, value: u32 },
}

/// Quantifier elimination context
pub struct QuantifierEliminationContext {
    /// Quantified variables
    quantified_vars: Vec<QuantifiedVar>,
    /// Free variables
    free_vars: FxHashSet<u32>,
    /// Skolem functions/constants introduced
    skolem_functions: FxHashMap<u32, SkolemTerm>,
    /// Patterns for quantifier instantiation
    patterns: Vec<ArrayQuantifierPattern>,
    /// Counter for generating fresh variables
    fresh_counter: u32,
}

/// Skolem term replacing a quantified variable
#[derive(Debug, Clone)]
pub struct SkolemTerm {
    /// Skolem identifier
    pub skolem_id: u32,
    /// Original quantified variable
    pub original_var: u32,
    /// Dependencies (free variables this Skolem depends on)
    pub dependencies: Vec<u32>,
    /// Term representing the Skolem
    pub term: TermId,
}

impl Default for QuantifierEliminationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantifierEliminationContext {
    /// Create a new quantifier elimination context
    pub fn new() -> Self {
        Self {
            quantified_vars: Vec::new(),
            free_vars: FxHashSet::default(),
            skolem_functions: FxHashMap::default(),
            patterns: Vec::new(),
            fresh_counter: 0,
        }
    }

    /// Register a quantified variable
    pub fn register_quantified_var(&mut self, var: QuantifiedVar) {
        self.quantified_vars.push(var);
    }

    /// Register a free variable
    pub fn register_free_var(&mut self, var_id: u32) {
        self.free_vars.insert(var_id);
    }

    /// Generate a fresh variable ID
    pub fn fresh_var(&mut self, _prefix: &str) -> u32 {
        let id = self.fresh_counter;
        self.fresh_counter += 1;
        id
    }

    /// Skolemize an existentially quantified variable
    ///
    /// For ∃x. φ(x, y₁, ..., yₙ), replace x with a Skolem function f(y₁, ..., yₙ)
    /// where y₁, ..., yₙ are free variables in φ
    pub fn skolemize(&mut self, var: &QuantifiedVar, dependencies: Vec<u32>) -> Result<SkolemTerm> {
        if !var.is_existential() {
            return Err(OxizError::Internal(
                "Can only Skolemize existential quantifiers".to_string(),
            ));
        }

        let skolem_id = self.fresh_var(&format!("sk_{}", var.name));
        let skolem = SkolemTerm {
            skolem_id,
            original_var: var.var_id,
            dependencies: dependencies.clone(),
            term: TermId::new(skolem_id),
        };

        self.skolem_functions.insert(var.var_id, skolem.clone());
        Ok(skolem)
    }

    /// Find dependencies of a quantified variable (free variables in the formula)
    pub fn find_dependencies(&self, _var_id: u32) -> Vec<u32> {
        // Return all free variables as dependencies
        self.free_vars.iter().copied().collect()
    }

    /// Add a quantifier instantiation pattern
    pub fn add_pattern(&mut self, pattern: ArrayQuantifierPattern) {
        self.patterns.push(pattern);
    }

    /// Get all patterns
    pub fn get_patterns(&self) -> &[ArrayQuantifierPattern] {
        &self.patterns
    }

    /// Clear the context
    pub fn clear(&mut self) {
        self.quantified_vars.clear();
        self.free_vars.clear();
        self.skolem_functions.clear();
        self.patterns.clear();
        self.fresh_counter = 0;
    }
}

/// Array quantifier eliminator
pub struct ArrayQuantifierEliminator {
    /// Elimination context
    context: QuantifierEliminationContext,
    /// Eliminated formulas
    eliminated_formulas: Vec<TermId>,
    /// Model-based projection cache
    mbp_cache: FxHashMap<u32, ModelProjection>,
}

/// Model-based projection result
#[derive(Debug, Clone)]
pub struct ModelProjection {
    /// Variable being eliminated
    pub var: u32,
    /// Substitution term
    pub substitution: TermId,
    /// Additional constraints
    pub constraints: Vec<TermId>,
}

impl Default for ArrayQuantifierEliminator {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayQuantifierEliminator {
    /// Create a new quantifier eliminator
    pub fn new() -> Self {
        Self {
            context: QuantifierEliminationContext::new(),
            eliminated_formulas: Vec::new(),
            mbp_cache: FxHashMap::default(),
        }
    }

    /// Eliminate existential quantifiers over arrays
    ///
    /// For ∃A. φ(A), use the following strategies:
    /// 1. If φ contains (A = store(B, i, v)), substitute A with store(B, i, v)
    /// 2. If φ contains select(A, i) = t, introduce fresh array B and substitute A with store(B, i, t)
    /// 3. For complex cases, use model-based projection
    pub fn eliminate_existential_array(
        &mut self,
        array_var: u32,
        formula: TermId,
    ) -> Result<EliminationResult> {
        // Try to find a simple substitution
        if let Some(substitution) = self.find_array_equality(array_var, formula) {
            return Ok(EliminationResult {
                eliminated_var: array_var,
                result_formula: formula,
                substitution: Some(substitution),
                auxiliary_vars: Vec::new(),
                auxiliary_constraints: Vec::new(),
            });
        }

        // Try to solve using select equations
        if let Some(result) = self.solve_select_equation(array_var, formula)? {
            return Ok(result);
        }

        // Fall back to model-based projection
        self.model_based_projection(array_var, formula)
    }

    /// Find an equality A = t in the formula
    fn find_array_equality(&self, _array_var: u32, _formula: TermId) -> Option<TermId> {
        // Placeholder: would analyze formula to find equalities
        None
    }

    /// Solve select equations: select(A, i) = t
    ///
    /// Replace A with store(B, i, t) where B is fresh
    fn solve_select_equation(
        &mut self,
        array_var: u32,
        _formula: TermId,
    ) -> Result<Option<EliminationResult>> {
        // Simplified implementation
        let fresh_array = self.context.fresh_var("arr");

        Ok(Some(EliminationResult {
            eliminated_var: array_var,
            result_formula: TermId::new(0), // Placeholder
            substitution: Some(TermId::new(fresh_array)),
            auxiliary_vars: vec![fresh_array],
            auxiliary_constraints: Vec::new(),
        }))
    }

    /// Model-based projection for array quantifier elimination
    fn model_based_projection(
        &mut self,
        array_var: u32,
        formula: TermId,
    ) -> Result<EliminationResult> {
        // Check cache first
        if let Some(cached) = self.mbp_cache.get(&array_var) {
            return Ok(EliminationResult {
                eliminated_var: array_var,
                result_formula: formula,
                substitution: Some(cached.substitution),
                auxiliary_vars: Vec::new(),
                auxiliary_constraints: cached.constraints.clone(),
            });
        }

        // Perform model-based projection
        let fresh_array = self.context.fresh_var("mbp_arr");
        let projection = ModelProjection {
            var: array_var,
            substitution: TermId::new(fresh_array),
            constraints: Vec::new(),
        };

        self.mbp_cache.insert(array_var, projection.clone());

        Ok(EliminationResult {
            eliminated_var: array_var,
            result_formula: formula,
            substitution: Some(projection.substitution),
            auxiliary_vars: vec![fresh_array],
            auxiliary_constraints: projection.constraints,
        })
    }

    /// Eliminate universal quantifiers using instantiation
    ///
    /// For ∀i. φ(i), find relevant instantiation terms and generate instances
    pub fn eliminate_universal_index(
        &mut self,
        index_var: u32,
        _formula: TermId,
    ) -> Result<Vec<TermId>> {
        // Collect relevant index terms for instantiation
        let instantiation_terms = self.collect_index_terms(index_var);

        // Generate instances
        let mut instances = Vec::new();
        for term in instantiation_terms {
            let _instance = self.instantiate_formula(index_var, term);
            instances.push(term); // Placeholder
        }

        Ok(instances)
    }

    /// Collect relevant index terms for instantiation
    fn collect_index_terms(&self, _index_var: u32) -> Vec<TermId> {
        // Would collect index terms from select and store operations
        Vec::new()
    }

    /// Instantiate a formula with a specific term
    fn instantiate_formula(&self, _var: u32, _term: TermId) -> TermId {
        // Would perform substitution
        TermId::new(0)
    }

    /// Handle extensionality quantifiers
    ///
    /// For (∀i. select(A, i) = select(B, i)) → A = B
    pub fn eliminate_extensionality(
        &mut self,
        index_var: u32,
        array1: u32,
        array2: u32,
    ) -> Result<TermId> {
        // Add extensionality axiom
        self.context
            .add_pattern(ArrayQuantifierPattern::ExtensionalityPattern {
                array1,
                array2,
                index_var,
            });

        // Return the conclusion A = B
        Ok(TermId::new(0)) // Placeholder
    }

    /// Get elimination context
    pub fn context(&self) -> &QuantifierEliminationContext {
        &self.context
    }

    /// Get mutable elimination context
    pub fn context_mut(&mut self) -> &mut QuantifierEliminationContext {
        &mut self.context
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.context.clear();
        self.eliminated_formulas.clear();
        self.mbp_cache.clear();
    }
}

/// Result of quantifier elimination
#[derive(Debug, Clone)]
pub struct EliminationResult {
    /// The variable that was eliminated
    pub eliminated_var: u32,
    /// The resulting formula (after elimination)
    pub result_formula: TermId,
    /// Substitution term for the eliminated variable
    pub substitution: Option<TermId>,
    /// Auxiliary variables introduced
    pub auxiliary_vars: Vec<u32>,
    /// Auxiliary constraints added
    pub auxiliary_constraints: Vec<TermId>,
}

/// Decidable fragment analyzer for arrays
pub struct ArrayFragmentAnalyzer {
    /// Fragment classification cache
    fragment_cache: FxHashMap<TermId, ArrayFragment>,
}

/// Types of array formula fragments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrayFragment {
    /// Bradley-Manna-Sipma fragment (quantifier-free with limited nesting)
    BradleyMannaSipma,
    /// Array Property Fragment (APF)
    ArrayProperty,
    /// Monadic fragment
    Monadic,
    /// Full theory (undecidable in general)
    Full,
}

impl Default for ArrayFragmentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayFragmentAnalyzer {
    /// Create a new fragment analyzer
    pub fn new() -> Self {
        Self {
            fragment_cache: FxHashMap::default(),
        }
    }

    /// Classify a formula into a fragment
    pub fn classify(&mut self, formula: TermId) -> ArrayFragment {
        if let Some(&fragment) = self.fragment_cache.get(&formula) {
            return fragment;
        }

        // Analyze the formula structure
        let fragment = self.analyze_formula(formula);
        self.fragment_cache.insert(formula, fragment);
        fragment
    }

    /// Analyze formula structure
    fn analyze_formula(&self, _formula: TermId) -> ArrayFragment {
        // Simplified: would analyze formula structure
        // Check for:
        // - Quantifier alternation
        // - Nesting depth of selects/stores
        // - Index variable dependencies
        ArrayFragment::BradleyMannaSipma
    }

    /// Check if a formula is in the Bradley-Manna-Sipma fragment
    ///
    /// BMS fragment characteristics:
    /// - Quantifier-free or limited quantification
    /// - Bounded array updates
    /// - Simple index expressions
    pub fn is_bradley_manna_sipma(&mut self, formula: TermId) -> bool {
        matches!(self.classify(formula), ArrayFragment::BradleyMannaSipma)
    }

    /// Check if a formula is in a decidable fragment
    pub fn is_decidable(&mut self, formula: TermId) -> bool {
        !matches!(self.classify(formula), ArrayFragment::Full)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.fragment_cache.clear();
    }
}

/// Quantifier instantiation strategy
pub struct QuantifierInstantiationStrategy {
    /// E-matching patterns
    e_matching_patterns: Vec<EMatchingPattern>,
    /// Trigger terms for instantiation
    trigger_terms: FxHashMap<u32, Vec<TermId>>,
    /// Instantiation heuristic
    heuristic: InstantiationHeuristic,
}

/// E-matching pattern for quantifier instantiation
#[derive(Debug, Clone)]
pub struct EMatchingPattern {
    /// Quantified variables bound by this pattern
    pub bound_vars: Vec<u32>,
    /// Pattern term
    pub pattern: TermId,
    /// Body formula
    pub body: TermId,
}

/// Instantiation heuristic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstantiationHeuristic {
    /// Conservative: only instantiate when necessary
    Conservative,
    /// Aggressive: instantiate eagerly
    Aggressive,
    /// Model-based: use model information to guide instantiation
    ModelBased,
}

impl Default for QuantifierInstantiationStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantifierInstantiationStrategy {
    /// Create a new instantiation strategy
    pub fn new() -> Self {
        Self {
            e_matching_patterns: Vec::new(),
            trigger_terms: FxHashMap::default(),
            heuristic: InstantiationHeuristic::Conservative,
        }
    }

    /// Add an e-matching pattern
    pub fn add_pattern(&mut self, pattern: EMatchingPattern) {
        self.e_matching_patterns.push(pattern);
    }

    /// Register a trigger term
    pub fn register_trigger(&mut self, var: u32, term: TermId) {
        self.trigger_terms.entry(var).or_default().push(term);
    }

    /// Set instantiation heuristic
    pub fn set_heuristic(&mut self, heuristic: InstantiationHeuristic) {
        self.heuristic = heuristic;
    }

    /// Get instantiation heuristic
    pub fn get_heuristic(&self) -> InstantiationHeuristic {
        self.heuristic
    }

    /// Find instantiations for a pattern
    pub fn find_instantiations(&self, pattern: &EMatchingPattern) -> Vec<Vec<TermId>> {
        let mut instantiations = Vec::new();

        // For each bound variable, collect possible instantiation terms
        for &var in &pattern.bound_vars {
            if let Some(terms) = self.trigger_terms.get(&var) {
                if instantiations.is_empty() {
                    instantiations = terms.iter().map(|t| vec![*t]).collect();
                } else {
                    // Cartesian product for multiple variables
                    let mut new_instantiations = Vec::new();
                    for inst in &instantiations {
                        for term in terms {
                            let mut new_inst = inst.clone();
                            new_inst.push(*term);
                            new_instantiations.push(new_inst);
                        }
                    }
                    instantiations = new_instantiations;
                }
            }
        }

        // Filter based on heuristic
        self.filter_instantiations(instantiations)
    }

    /// Filter instantiations based on heuristic
    fn filter_instantiations(&self, instantiations: Vec<Vec<TermId>>) -> Vec<Vec<TermId>> {
        match self.heuristic {
            InstantiationHeuristic::Conservative => {
                // Limit number of instantiations
                instantiations.into_iter().take(10).collect()
            }
            InstantiationHeuristic::Aggressive => {
                // Use all instantiations
                instantiations
            }
            InstantiationHeuristic::ModelBased => {
                // Would filter based on model information
                instantiations.into_iter().take(50).collect()
            }
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.e_matching_patterns.clear();
        self.trigger_terms.clear();
    }
}

/// Index term collector for quantifier instantiation
pub struct IndexTermCollector {
    /// Collected index terms
    index_terms: FxHashSet<TermId>,
    /// Index terms by sort
    terms_by_sort: FxHashMap<u32, FxHashSet<TermId>>,
}

impl Default for IndexTermCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexTermCollector {
    /// Create a new index term collector
    pub fn new() -> Self {
        Self {
            index_terms: FxHashSet::default(),
            terms_by_sort: FxHashMap::default(),
        }
    }

    /// Add an index term
    pub fn add_term(&mut self, term: TermId, sort: u32) {
        self.index_terms.insert(term);
        self.terms_by_sort.entry(sort).or_default().insert(term);
    }

    /// Get all index terms
    pub fn get_all_terms(&self) -> Vec<TermId> {
        self.index_terms.iter().copied().collect()
    }

    /// Get index terms of a specific sort
    pub fn get_terms_of_sort(&self, sort: u32) -> Vec<TermId> {
        self.terms_by_sort
            .get(&sort)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Get number of collected terms
    pub fn num_terms(&self) -> usize {
        self.index_terms.len()
    }

    /// Clear all terms
    pub fn clear(&mut self) {
        self.index_terms.clear();
        self.terms_by_sort.clear();
    }
}

/// Array formula simplifier with quantifier elimination
pub struct ArrayFormulaSimplifier {
    /// Quantifier eliminator
    eliminator: ArrayQuantifierEliminator,
    /// Fragment analyzer
    fragment_analyzer: ArrayFragmentAnalyzer,
    /// Simplification cache
    simplification_cache: FxHashMap<TermId, TermId>,
}

impl Default for ArrayFormulaSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayFormulaSimplifier {
    /// Create a new simplifier
    pub fn new() -> Self {
        Self {
            eliminator: ArrayQuantifierEliminator::new(),
            fragment_analyzer: ArrayFragmentAnalyzer::new(),
            simplification_cache: FxHashMap::default(),
        }
    }

    /// Simplify a formula with quantifiers
    pub fn simplify(&mut self, formula: TermId) -> Result<TermId> {
        // Check cache
        if let Some(&simplified) = self.simplification_cache.get(&formula) {
            return Ok(simplified);
        }

        // Check if formula is in a decidable fragment
        let fragment = self.fragment_analyzer.classify(formula);

        let result = match fragment {
            ArrayFragment::BradleyMannaSipma => {
                // Use specialized techniques for BMS fragment
                self.simplify_bms(formula)?
            }
            ArrayFragment::ArrayProperty => {
                // Use array property techniques
                self.simplify_apf(formula)?
            }
            ArrayFragment::Monadic => {
                // Use monadic simplification
                self.simplify_monadic(formula)?
            }
            ArrayFragment::Full => {
                // Use general quantifier elimination
                formula // Placeholder
            }
        };

        self.simplification_cache.insert(formula, result);
        Ok(result)
    }

    /// Simplify Bradley-Manna-Sipma fragment
    fn simplify_bms(&mut self, formula: TermId) -> Result<TermId> {
        // BMS-specific simplification
        Ok(formula)
    }

    /// Simplify Array Property Fragment
    fn simplify_apf(&mut self, formula: TermId) -> Result<TermId> {
        // APF-specific simplification
        Ok(formula)
    }

    /// Simplify monadic fragment
    fn simplify_monadic(&mut self, formula: TermId) -> Result<TermId> {
        // Monadic-specific simplification
        Ok(formula)
    }

    /// Get the eliminator
    pub fn eliminator(&self) -> &ArrayQuantifierEliminator {
        &self.eliminator
    }

    /// Get mutable eliminator
    pub fn eliminator_mut(&mut self) -> &mut ArrayQuantifierEliminator {
        &mut self.eliminator
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.eliminator.clear();
        self.fragment_analyzer.clear();
        self.simplification_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantified_var() {
        let var = QuantifiedVar::new(1, "x".to_string(), 10, QuantifierType::Forall);
        assert!(var.is_universal());
        assert!(!var.is_existential());
    }

    #[test]
    fn test_qe_context() {
        let mut ctx = QuantifierEliminationContext::new();
        ctx.register_free_var(1);
        ctx.register_free_var(2);

        let fresh = ctx.fresh_var("test");
        assert_eq!(fresh, 0);

        let fresh2 = ctx.fresh_var("test");
        assert_eq!(fresh2, 1);
    }

    #[test]
    fn test_skolemization() {
        let mut ctx = QuantifierEliminationContext::new();
        ctx.register_free_var(5);
        ctx.register_free_var(6);

        let var = QuantifiedVar::new(10, "x".to_string(), 1, QuantifierType::Exists);
        let deps = ctx.find_dependencies(10);
        let skolem = ctx.skolemize(&var, deps).unwrap();

        assert_eq!(skolem.original_var, 10);
        assert!(!skolem.dependencies.is_empty());
    }

    #[test]
    fn test_cannot_skolemize_universal() {
        let mut ctx = QuantifierEliminationContext::new();
        let var = QuantifiedVar::new(10, "x".to_string(), 1, QuantifierType::Forall);
        let result = ctx.skolemize(&var, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fragment_analyzer() {
        let mut analyzer = ArrayFragmentAnalyzer::new();
        let formula = TermId::new(100);

        let fragment = analyzer.classify(formula);
        assert_eq!(fragment, ArrayFragment::BradleyMannaSipma);

        // Test cache
        let fragment2 = analyzer.classify(formula);
        assert_eq!(fragment, fragment2);
    }

    #[test]
    fn test_instantiation_strategy() {
        let mut strategy = QuantifierInstantiationStrategy::new();
        strategy.set_heuristic(InstantiationHeuristic::Conservative);

        strategy.register_trigger(1, TermId::new(10));
        strategy.register_trigger(1, TermId::new(11));
        strategy.register_trigger(1, TermId::new(12));

        let pattern = EMatchingPattern {
            bound_vars: vec![1],
            pattern: TermId::new(100),
            body: TermId::new(200),
        };

        let insts = strategy.find_instantiations(&pattern);
        assert!(!insts.is_empty());
    }

    #[test]
    fn test_index_term_collector() {
        let mut collector = IndexTermCollector::new();

        collector.add_term(TermId::new(1), 10);
        collector.add_term(TermId::new(2), 10);
        collector.add_term(TermId::new(3), 20);

        assert_eq!(collector.num_terms(), 3);

        let terms_10 = collector.get_terms_of_sort(10);
        assert_eq!(terms_10.len(), 2);

        let terms_20 = collector.get_terms_of_sort(20);
        assert_eq!(terms_20.len(), 1);
    }

    #[test]
    fn test_array_eliminator() {
        let mut elim = ArrayQuantifierEliminator::new();
        let formula = TermId::new(1000);

        let result = elim.eliminate_existential_array(100, formula);
        assert!(result.is_ok());
    }

    #[test]
    fn test_elimination_result() {
        let result = EliminationResult {
            eliminated_var: 10,
            result_formula: TermId::new(100),
            substitution: Some(TermId::new(200)),
            auxiliary_vars: vec![20, 21],
            auxiliary_constraints: vec![TermId::new(300)],
        };

        assert_eq!(result.eliminated_var, 10);
        assert_eq!(result.auxiliary_vars.len(), 2);
        assert_eq!(result.auxiliary_constraints.len(), 1);
    }

    #[test]
    fn test_formula_simplifier() {
        let mut simplifier = ArrayFormulaSimplifier::new();
        let formula = TermId::new(1000);

        let simplified = simplifier.simplify(formula).unwrap();
        assert_eq!(simplified, formula);

        // Test cache
        let simplified2 = simplifier.simplify(formula).unwrap();
        assert_eq!(simplified, simplified2);
    }

    #[test]
    fn test_quantifier_pattern() {
        let pattern = ArrayQuantifierPattern::SelectPattern {
            array_var: 1,
            index_var: 2,
            value_expr: TermId::new(100),
        };

        if let ArrayQuantifierPattern::SelectPattern {
            array_var,
            index_var,
            ..
        } = pattern
        {
            assert_eq!(array_var, 1);
            assert_eq!(index_var, 2);
        } else {
            panic!("Wrong pattern type");
        }
    }

    #[test]
    fn test_model_projection() {
        let projection = ModelProjection {
            var: 100,
            substitution: TermId::new(200),
            constraints: vec![TermId::new(300), TermId::new(301)],
        };

        assert_eq!(projection.var, 100);
        assert_eq!(projection.constraints.len(), 2);
    }

    #[test]
    fn test_multiple_trigger_vars() {
        let mut strategy = QuantifierInstantiationStrategy::new();

        strategy.register_trigger(1, TermId::new(10));
        strategy.register_trigger(1, TermId::new(11));
        strategy.register_trigger(2, TermId::new(20));
        strategy.register_trigger(2, TermId::new(21));

        let pattern = EMatchingPattern {
            bound_vars: vec![1, 2],
            pattern: TermId::new(100),
            body: TermId::new(200),
        };

        let insts = strategy.find_instantiations(&pattern);
        // Should have 2 * 2 = 4 combinations
        assert_eq!(insts.len(), 4);
    }

    #[test]
    fn test_clear_operations() {
        let mut ctx = QuantifierEliminationContext::new();
        ctx.register_free_var(1);
        ctx.fresh_var("test");
        ctx.clear();

        let fresh = ctx.fresh_var("after_clear");
        assert_eq!(fresh, 0); // Counter should be reset
    }
}
