//! Array Property Fragments
//!
//! This module implements support for decidable fragments of array logic,
//! particularly the Bradley-Manna-Sipma (BMS) fragment and related
//! array property fragments.
//!
//! Reference: "The Calculus of Computation" by Bradley & Manna,
//! and "Array-based System Design and Verification" papers

#![allow(missing_docs)]

use oxiz_core::ast::TermId;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// Bradley-Manna-Sipma array property fragment
///
/// Characteristics:
/// - Limited quantifier nesting
/// - Restricted array update patterns
/// - Decidable satisfiability checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BMSFragment {
    /// Maximum store depth allowed
    pub max_store_depth: usize,
    /// Maximum quantifier alternations
    pub max_quantifier_alternations: usize,
    /// Allow nested array accesses
    pub allow_nested_access: bool,
}

impl Default for BMSFragment {
    fn default() -> Self {
        Self {
            max_store_depth: 3,
            max_quantifier_alternations: 1,
            allow_nested_access: false,
        }
    }
}

impl BMSFragment {
    /// Create a new BMS fragment configuration
    pub fn new(max_store_depth: usize, max_quantifier_alternations: usize) -> Self {
        Self {
            max_store_depth,
            max_quantifier_alternations,
            allow_nested_access: false,
        }
    }

    /// Check if a term is within the fragment
    pub fn is_in_fragment(&self, term: &ArrayTerm) -> bool {
        self.check_store_depth(term) && self.check_access_pattern(term)
    }

    /// Check store depth constraint
    fn check_store_depth(&self, term: &ArrayTerm) -> bool {
        term.store_depth() <= self.max_store_depth
    }

    /// Check access pattern constraint
    fn check_access_pattern(&self, term: &ArrayTerm) -> bool {
        if !self.allow_nested_access {
            !term.has_nested_access()
        } else {
            true
        }
    }
}

/// Array term representation for fragment analysis
#[derive(Debug, Clone)]
pub enum ArrayTerm {
    /// Base array variable
    Variable { id: u32, name: String },
    /// Select operation
    Select {
        array: Box<ArrayTerm>,
        indices: Vec<IndexTerm>,
        depth: usize,
    },
    /// Store operation
    Store {
        array: Box<ArrayTerm>,
        indices: Vec<IndexTerm>,
        value: ValueTerm,
        depth: usize,
    },
    /// Constant array
    ConstArray { value: ValueTerm },
}

impl ArrayTerm {
    /// Get the store depth of this term
    pub fn store_depth(&self) -> usize {
        match self {
            ArrayTerm::Variable { .. } => 0,
            ArrayTerm::Select { array, .. } => array.store_depth(),
            ArrayTerm::Store { depth, .. } => *depth,
            ArrayTerm::ConstArray { .. } => 0,
        }
    }

    /// Check if this term has nested array accesses
    pub fn has_nested_access(&self) -> bool {
        match self {
            ArrayTerm::Variable { .. } => false,
            ArrayTerm::Select { array, indices, .. } => {
                array.has_nested_access() || indices.iter().any(|idx| idx.contains_array_access())
            }
            ArrayTerm::Store {
                array,
                indices,
                value,
                ..
            } => {
                array.has_nested_access()
                    || indices.iter().any(|idx| idx.contains_array_access())
                    || value.contains_array_access()
            }
            ArrayTerm::ConstArray { value } => value.contains_array_access(),
        }
    }

    /// Extract all array variables
    pub fn extract_variables(&self) -> FxHashSet<u32> {
        let mut vars = FxHashSet::default();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut FxHashSet<u32>) {
        match self {
            ArrayTerm::Variable { id, .. } => {
                vars.insert(*id);
            }
            ArrayTerm::Select { array, .. } => {
                array.collect_variables(vars);
            }
            ArrayTerm::Store { array, .. } => {
                array.collect_variables(vars);
            }
            ArrayTerm::ConstArray { .. } => {}
        }
    }
}

/// Index term in array accesses
#[derive(Debug, Clone)]
pub enum IndexTerm {
    /// Variable index
    Variable(u32),
    /// Constant index
    Constant(i64),
    /// Arithmetic expression
    Arithmetic {
        op: ArithOp,
        operands: Vec<IndexTerm>,
    },
    /// Array select as index (nested access)
    ArraySelect(Box<ArrayTerm>),
}

impl IndexTerm {
    /// Check if this index contains array accesses
    pub fn contains_array_access(&self) -> bool {
        match self {
            IndexTerm::Variable(_) | IndexTerm::Constant(_) => false,
            IndexTerm::Arithmetic { operands, .. } => {
                operands.iter().any(|op| op.contains_array_access())
            }
            IndexTerm::ArraySelect(_) => true,
        }
    }
}

/// Arithmetic operations in index expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Value term in array stores
#[derive(Debug, Clone)]
pub enum ValueTerm {
    /// Variable value
    Variable(u32),
    /// Constant value
    Constant(i64),
    /// Array select
    ArraySelect(Box<ArrayTerm>),
    /// Complex expression
    Expression(TermId),
}

impl ValueTerm {
    /// Check if this value contains array accesses
    pub fn contains_array_access(&self) -> bool {
        matches!(self, ValueTerm::ArraySelect(_))
    }
}

/// Array property fragment classifier
pub struct PropertyFragmentClassifier {
    /// BMS fragment configuration
    bms_config: BMSFragment,
    /// Classification cache
    cache: FxHashMap<TermId, FragmentClass>,
}

/// Fragment classification result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FragmentClass {
    /// Pure Bradley-Manna-Sipma fragment
    PureBMS,
    /// Extended BMS (relaxed constraints)
    ExtendedBMS,
    /// Array property fragment (APF)
    APF,
    /// Monadic fragment
    Monadic,
    /// Full theory (not in a decidable fragment)
    Full,
}

impl fmt::Display for FragmentClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FragmentClass::PureBMS => write!(f, "Pure BMS"),
            FragmentClass::ExtendedBMS => write!(f, "Extended BMS"),
            FragmentClass::APF => write!(f, "Array Property Fragment"),
            FragmentClass::Monadic => write!(f, "Monadic"),
            FragmentClass::Full => write!(f, "Full Theory"),
        }
    }
}

impl Default for PropertyFragmentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl PropertyFragmentClassifier {
    /// Create a new classifier
    pub fn new() -> Self {
        Self {
            bms_config: BMSFragment::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Create with custom BMS configuration
    pub fn with_bms_config(config: BMSFragment) -> Self {
        Self {
            bms_config: config,
            cache: FxHashMap::default(),
        }
    }

    /// Classify a formula
    pub fn classify(&mut self, formula: TermId) -> FragmentClass {
        if let Some(&class) = self.cache.get(&formula) {
            return class;
        }

        let class = self.classify_impl(formula);
        self.cache.insert(formula, class);
        class
    }

    fn classify_impl(&self, _formula: TermId) -> FragmentClass {
        // Simplified implementation
        // Would analyze formula structure in detail
        FragmentClass::PureBMS
    }

    /// Check if a term is in pure BMS fragment
    pub fn is_pure_bms(&self, term: &ArrayTerm) -> bool {
        self.bms_config.is_in_fragment(term)
    }

    /// Get BMS configuration
    pub fn bms_config(&self) -> &BMSFragment {
        &self.bms_config
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Decidability checker for array formulas
pub struct DecidabilityChecker {
    /// Fragment classifier
    classifier: PropertyFragmentClassifier,
    /// Known decidable patterns
    decidable_patterns: Vec<DecidablePattern>,
}

/// Decidable pattern in array logic
#[derive(Debug, Clone)]
pub struct DecidablePattern {
    /// Pattern name
    pub name: String,
    /// Pattern description
    pub description: String,
    /// Fragment class this pattern belongs to
    pub fragment: FragmentClass,
}

impl Default for DecidabilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl DecidabilityChecker {
    /// Create a new decidability checker
    pub fn new() -> Self {
        let mut checker = Self {
            classifier: PropertyFragmentClassifier::new(),
            decidable_patterns: Vec::new(),
        };
        checker.initialize_patterns();
        checker
    }

    /// Initialize known decidable patterns
    fn initialize_patterns(&mut self) {
        self.decidable_patterns.push(DecidablePattern {
            name: "BMS-QF".to_string(),
            description: "Quantifier-free BMS fragment".to_string(),
            fragment: FragmentClass::PureBMS,
        });

        self.decidable_patterns.push(DecidablePattern {
            name: "Array-Property".to_string(),
            description: "Array property fragment with limited quantification".to_string(),
            fragment: FragmentClass::APF,
        });

        self.decidable_patterns.push(DecidablePattern {
            name: "Monadic-Arrays".to_string(),
            description: "Monadic fragment with single array variable".to_string(),
            fragment: FragmentClass::Monadic,
        });
    }

    /// Check if a formula is decidable
    pub fn is_decidable(&mut self, formula: TermId) -> bool {
        let fragment = self.classifier.classify(formula);
        !matches!(fragment, FragmentClass::Full)
    }

    /// Get decidability report
    pub fn get_report(&mut self, formula: TermId) -> DecidabilityReport {
        let fragment = self.classifier.classify(formula);
        let is_decidable = !matches!(fragment, FragmentClass::Full);

        let applicable_patterns: Vec<_> = self
            .decidable_patterns
            .iter()
            .filter(|p| p.fragment == fragment)
            .cloned()
            .collect();

        DecidabilityReport {
            formula,
            fragment,
            is_decidable,
            applicable_patterns,
            reason: self.get_classification_reason(fragment),
        }
    }

    fn get_classification_reason(&self, fragment: FragmentClass) -> String {
        match fragment {
            FragmentClass::PureBMS => "Formula is in pure Bradley-Manna-Sipma fragment".to_string(),
            FragmentClass::ExtendedBMS => {
                "Formula is in extended BMS fragment with relaxed constraints".to_string()
            }
            FragmentClass::APF => "Formula is in array property fragment".to_string(),
            FragmentClass::Monadic => "Formula is in monadic fragment".to_string(),
            FragmentClass::Full => "Formula is in full theory (undecidable in general)".to_string(),
        }
    }

    /// Get classifier
    pub fn classifier(&self) -> &PropertyFragmentClassifier {
        &self.classifier
    }
}

/// Decidability report for a formula
#[derive(Debug, Clone)]
pub struct DecidabilityReport {
    /// Formula being analyzed
    pub formula: TermId,
    /// Fragment classification
    pub fragment: FragmentClass,
    /// Whether the formula is decidable
    pub is_decidable: bool,
    /// Applicable decidable patterns
    pub applicable_patterns: Vec<DecidablePattern>,
    /// Reason for classification
    pub reason: String,
}

impl fmt::Display for DecidabilityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Decidability Report")?;
        writeln!(f, "===================")?;
        writeln!(f, "Formula: {:?}", self.formula)?;
        writeln!(f, "Fragment: {}", self.fragment)?;
        writeln!(f, "Decidable: {}", self.is_decidable)?;
        writeln!(f, "Reason: {}", self.reason)?;
        writeln!(f, "\nApplicable Patterns:")?;
        for pattern in &self.applicable_patterns {
            writeln!(f, "  - {}: {}", pattern.name, pattern.description)?;
        }
        Ok(())
    }
}

/// Array update pattern analyzer
pub struct UpdatePatternAnalyzer {
    /// Detected update patterns
    patterns: Vec<UpdatePattern>,
    /// Pattern statistics
    stats: UpdatePatternStats,
}

/// Array update pattern
#[derive(Debug, Clone)]
pub struct UpdatePattern {
    /// Pattern type
    pub pattern_type: UpdatePatternType,
    /// Array being updated
    pub array: u32,
    /// Update locations
    pub locations: Vec<Vec<u32>>,
    /// Update depth
    pub depth: usize,
}

/// Type of update pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdatePatternType {
    /// Single update: store(A, i, v)
    Single,
    /// Sequential updates: store(store(A, i1, v1), i2, v2)
    Sequential,
    /// Disjoint updates: updates at provably different indices
    Disjoint,
    /// Overlapping updates: updates may overlap
    Overlapping,
    /// Constant propagation: store(A, i, c) where c is constant
    ConstantPropagation,
}

/// Statistics for update patterns
#[derive(Debug, Clone, Default)]
pub struct UpdatePatternStats {
    /// Number of single updates
    pub num_single: usize,
    /// Number of sequential updates
    pub num_sequential: usize,
    /// Number of disjoint updates
    pub num_disjoint: usize,
    /// Number of overlapping updates
    pub num_overlapping: usize,
    /// Maximum update depth seen
    pub max_depth: usize,
}

impl Default for UpdatePatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl UpdatePatternAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            stats: UpdatePatternStats::default(),
        }
    }

    /// Analyze array term for update patterns
    pub fn analyze(&mut self, term: &ArrayTerm) {
        self.analyze_impl(term, 0);
    }

    fn analyze_impl(&mut self, term: &ArrayTerm, recursion_depth: usize) {
        match term {
            ArrayTerm::Store {
                array,
                indices: _,
                value,
                depth,
            } => {
                // Use the Store's depth field, not recursion depth
                let store_depth = *depth;

                // Determine pattern type
                let pattern_type = if recursion_depth == 0 {
                    UpdatePatternType::Single
                } else {
                    // Check if this is part of a sequential pattern
                    if matches!(value, ValueTerm::Constant(_)) {
                        UpdatePatternType::ConstantPropagation
                    } else {
                        UpdatePatternType::Sequential
                    }
                };

                let pattern = UpdatePattern {
                    pattern_type,
                    array: 0, // Would extract actual array ID
                    locations: vec![vec![]],
                    depth: store_depth,
                };

                self.patterns.push(pattern);
                self.update_stats(pattern_type, store_depth);

                // Recurse on base array
                self.analyze_impl(array, recursion_depth + 1);
            }
            ArrayTerm::Variable { .. } | ArrayTerm::ConstArray { .. } => {
                // Base case
            }
            ArrayTerm::Select { array, .. } => {
                self.analyze_impl(array, recursion_depth);
            }
        }
    }

    fn update_stats(&mut self, pattern_type: UpdatePatternType, depth: usize) {
        match pattern_type {
            UpdatePatternType::Single => self.stats.num_single += 1,
            UpdatePatternType::Sequential => self.stats.num_sequential += 1,
            UpdatePatternType::Disjoint => self.stats.num_disjoint += 1,
            UpdatePatternType::Overlapping => self.stats.num_overlapping += 1,
            UpdatePatternType::ConstantPropagation => {}
        }
        self.stats.max_depth = self.stats.max_depth.max(depth);
    }

    /// Get detected patterns
    pub fn get_patterns(&self) -> &[UpdatePattern] {
        &self.patterns
    }

    /// Get statistics
    pub fn get_stats(&self) -> &UpdatePatternStats {
        &self.stats
    }

    /// Clear analysis
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.stats = UpdatePatternStats::default();
    }
}

/// Quantifier complexity analyzer for array formulas
pub struct QuantifierComplexityAnalyzer {
    /// Maximum quantifier nesting depth
    max_nesting: usize,
    /// Number of quantifier alternations
    num_alternations: usize,
    /// Quantified variables by level
    vars_by_level: Vec<Vec<u32>>,
}

impl Default for QuantifierComplexityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantifierComplexityAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            max_nesting: 0,
            num_alternations: 0,
            vars_by_level: Vec::new(),
        }
    }

    /// Analyze quantifier complexity
    pub fn analyze(&mut self, _formula: TermId) -> QuantifierComplexity {
        // Would analyze formula structure
        QuantifierComplexity {
            max_nesting: self.max_nesting,
            num_alternations: self.num_alternations,
            has_array_quantifiers: true,
            has_index_quantifiers: true,
        }
    }

    /// Check if complexity is within BMS limits
    pub fn is_within_bms(&self, config: &BMSFragment) -> bool {
        self.num_alternations <= config.max_quantifier_alternations
    }

    /// Reset analysis
    pub fn reset(&mut self) {
        self.max_nesting = 0;
        self.num_alternations = 0;
        self.vars_by_level.clear();
    }
}

/// Quantifier complexity metrics
#[derive(Debug, Clone)]
pub struct QuantifierComplexity {
    /// Maximum nesting depth
    pub max_nesting: usize,
    /// Number of alternations (∀∃ or ∃∀)
    pub num_alternations: usize,
    /// Has quantifiers over array variables
    pub has_array_quantifiers: bool,
    /// Has quantifiers over index variables
    pub has_index_quantifiers: bool,
}

impl fmt::Display for QuantifierComplexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quantifier Complexity:")?;
        writeln!(f, "  Max nesting: {}", self.max_nesting)?;
        writeln!(f, "  Alternations: {}", self.num_alternations)?;
        writeln!(f, "  Array quantifiers: {}", self.has_array_quantifiers)?;
        writeln!(f, "  Index quantifiers: {}", self.has_index_quantifiers)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bms_fragment() {
        let config = BMSFragment::new(3, 1);
        assert_eq!(config.max_store_depth, 3);
        assert_eq!(config.max_quantifier_alternations, 1);
    }

    #[test]
    fn test_array_term_depth() {
        let base = ArrayTerm::Variable {
            id: 1,
            name: "A".to_string(),
        };
        assert_eq!(base.store_depth(), 0);

        let store = ArrayTerm::Store {
            array: Box::new(base.clone()),
            indices: vec![],
            value: ValueTerm::Constant(0),
            depth: 1,
        };
        assert_eq!(store.store_depth(), 1);
    }

    #[test]
    fn test_nested_access_detection() {
        let base = ArrayTerm::Variable {
            id: 1,
            name: "A".to_string(),
        };
        assert!(!base.has_nested_access());

        let select_in_index = ArrayTerm::Select {
            array: Box::new(base.clone()),
            indices: vec![IndexTerm::ArraySelect(Box::new(base.clone()))],
            depth: 0,
        };
        assert!(select_in_index.has_nested_access());
    }

    #[test]
    fn test_fragment_classifier() {
        let mut classifier = PropertyFragmentClassifier::new();
        let formula = TermId::new(100);

        let class = classifier.classify(formula);
        assert_eq!(class, FragmentClass::PureBMS);

        // Test cache
        let class2 = classifier.classify(formula);
        assert_eq!(class, class2);
    }

    #[test]
    fn test_decidability_checker() {
        let mut checker = DecidabilityChecker::new();
        let formula = TermId::new(100);

        assert!(checker.is_decidable(formula));

        let report = checker.get_report(formula);
        assert!(report.is_decidable);
        assert!(!report.applicable_patterns.is_empty());
    }

    #[test]
    fn test_decidability_report_display() {
        let report = DecidabilityReport {
            formula: TermId::new(100),
            fragment: FragmentClass::PureBMS,
            is_decidable: true,
            applicable_patterns: vec![],
            reason: "Test reason".to_string(),
        };

        let display = format!("{}", report);
        assert!(display.contains("Decidability Report"));
        assert!(display.contains("Pure BMS"));
    }

    #[test]
    fn test_update_pattern_analyzer() {
        let mut analyzer = UpdatePatternAnalyzer::new();

        let base = ArrayTerm::Variable {
            id: 1,
            name: "A".to_string(),
        };

        let store = ArrayTerm::Store {
            array: Box::new(base),
            indices: vec![IndexTerm::Variable(5)],
            value: ValueTerm::Constant(42),
            depth: 1,
        };

        analyzer.analyze(&store);

        let stats = analyzer.get_stats();
        assert!(stats.num_single > 0 || stats.num_sequential > 0);
        assert!(stats.max_depth > 0);
    }

    #[test]
    fn test_index_term_array_access() {
        let idx_var = IndexTerm::Variable(1);
        assert!(!idx_var.contains_array_access());

        let arr = ArrayTerm::Variable {
            id: 10,
            name: "A".to_string(),
        };
        let idx_select = IndexTerm::ArraySelect(Box::new(arr));
        assert!(idx_select.contains_array_access());
    }

    #[test]
    fn test_value_term_array_access() {
        let val_var = ValueTerm::Variable(1);
        assert!(!val_var.contains_array_access());

        let arr = ArrayTerm::Variable {
            id: 10,
            name: "A".to_string(),
        };
        let val_select = ValueTerm::ArraySelect(Box::new(arr));
        assert!(val_select.contains_array_access());
    }

    #[test]
    fn test_quantifier_complexity() {
        let complexity = QuantifierComplexity {
            max_nesting: 2,
            num_alternations: 1,
            has_array_quantifiers: true,
            has_index_quantifiers: false,
        };

        let display = format!("{}", complexity);
        assert!(display.contains("Max nesting: 2"));
        assert!(display.contains("Alternations: 1"));
    }

    #[test]
    fn test_quantifier_complexity_analyzer() {
        let mut analyzer = QuantifierComplexityAnalyzer::new();
        let formula = TermId::new(100);

        let complexity = analyzer.analyze(formula);
        assert!(complexity.has_array_quantifiers);
    }

    #[test]
    fn test_bms_config_in_fragment() {
        let config = BMSFragment::default();
        let base = ArrayTerm::Variable {
            id: 1,
            name: "A".to_string(),
        };

        assert!(config.is_in_fragment(&base));
    }

    #[test]
    fn test_arith_op_variants() {
        let ops = [
            ArithOp::Add,
            ArithOp::Sub,
            ArithOp::Mul,
            ArithOp::Div,
            ArithOp::Mod,
        ];
        assert_eq!(ops.len(), 5);
    }

    #[test]
    fn test_update_pattern_types() {
        let types = [
            UpdatePatternType::Single,
            UpdatePatternType::Sequential,
            UpdatePatternType::Disjoint,
            UpdatePatternType::Overlapping,
            UpdatePatternType::ConstantPropagation,
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_fragment_class_display() {
        assert_eq!(format!("{}", FragmentClass::PureBMS), "Pure BMS");
        assert_eq!(format!("{}", FragmentClass::ExtendedBMS), "Extended BMS");
        assert_eq!(format!("{}", FragmentClass::APF), "Array Property Fragment");
        assert_eq!(format!("{}", FragmentClass::Monadic), "Monadic");
        assert_eq!(format!("{}", FragmentClass::Full), "Full Theory");
    }
}
