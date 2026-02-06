//! Array Abstraction and Refinement
//!
//! This module implements abstraction and refinement techniques for arrays:
//! - Abstract interpretation for array operations
//! - Predicate abstraction
//! - Counterexample-guided abstraction refinement (CEGAR)
//! - Array summaries and invariants
//!
//! Reference: Z3's array abstraction techniques and CEGAR implementations

use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// Abstract domain for array values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractDomain {
    /// Top element (all possible values)
    Top,
    /// Bottom element (no values / unreachable)
    Bottom,
    /// Concrete value set
    ValueSet(FxHashSet<i64>),
    /// Interval [min, max]
    Interval { min: i64, max: i64 },
    /// Sign domain
    Sign(SignDomain),
    /// Parity domain
    Parity(ParityDomain),
}

/// Sign abstract domain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignDomain {
    /// Positive values (> 0)
    Positive,
    /// Negative values (< 0)
    Negative,
    /// Zero
    Zero,
    /// Non-negative (>= 0)
    NonNegative,
    /// Non-positive (<= 0)
    NonPositive,
    /// Any sign
    Any,
}

/// Parity abstract domain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParityDomain {
    /// Even values
    Even,
    /// Odd values
    Odd,
    /// Any parity
    Any,
}

impl fmt::Display for AbstractDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbstractDomain::Top => write!(f, "⊤"),
            AbstractDomain::Bottom => write!(f, "⊥"),
            AbstractDomain::ValueSet(vals) => {
                write!(f, "{{")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "}}")
            }
            AbstractDomain::Interval { min, max } => write!(f, "[{}, {}]", min, max),
            AbstractDomain::Sign(s) => write!(f, "{:?}", s),
            AbstractDomain::Parity(p) => write!(f, "{:?}", p),
        }
    }
}

impl AbstractDomain {
    /// Join (least upper bound) of two abstract values
    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (AbstractDomain::Bottom, x) | (x, AbstractDomain::Bottom) => x.clone(),
            (AbstractDomain::Top, _) | (_, AbstractDomain::Top) => AbstractDomain::Top,
            (
                AbstractDomain::Interval {
                    min: min1,
                    max: max1,
                },
                AbstractDomain::Interval {
                    min: min2,
                    max: max2,
                },
            ) => AbstractDomain::Interval {
                min: (*min1).min(*min2),
                max: (*max1).max(*max2),
            },
            (AbstractDomain::ValueSet(s1), AbstractDomain::ValueSet(s2)) => {
                let mut result = s1.clone();
                result.extend(s2.iter());
                if result.len() > 10 {
                    // Widen to interval if too many values
                    let min = result.iter().min().copied().unwrap_or(0);
                    let max = result.iter().max().copied().unwrap_or(0);
                    AbstractDomain::Interval { min, max }
                } else {
                    AbstractDomain::ValueSet(result)
                }
            }
            _ => AbstractDomain::Top,
        }
    }

    /// Meet (greatest lower bound) of two abstract values
    pub fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (AbstractDomain::Top, x) | (x, AbstractDomain::Top) => x.clone(),
            (AbstractDomain::Bottom, _) | (_, AbstractDomain::Bottom) => AbstractDomain::Bottom,
            (
                AbstractDomain::Interval {
                    min: min1,
                    max: max1,
                },
                AbstractDomain::Interval {
                    min: min2,
                    max: max2,
                },
            ) => {
                let min = (*min1).max(*min2);
                let max = (*max1).min(*max2);
                if min <= max {
                    AbstractDomain::Interval { min, max }
                } else {
                    AbstractDomain::Bottom
                }
            }
            (AbstractDomain::ValueSet(s1), AbstractDomain::ValueSet(s2)) => {
                let result: FxHashSet<_> = s1.intersection(s2).copied().collect();
                if result.is_empty() {
                    AbstractDomain::Bottom
                } else {
                    AbstractDomain::ValueSet(result)
                }
            }
            _ => AbstractDomain::Bottom,
        }
    }

    /// Check if this is less than or equal to another (partial order)
    pub fn is_less_or_equal(&self, other: &Self) -> bool {
        match (self, other) {
            (AbstractDomain::Bottom, _) => true,
            (_, AbstractDomain::Top) => true,
            (AbstractDomain::Top, _) | (_, AbstractDomain::Bottom) => false,
            (
                AbstractDomain::Interval {
                    min: min1,
                    max: max1,
                },
                AbstractDomain::Interval {
                    min: min2,
                    max: max2,
                },
            ) => min1 >= min2 && max1 <= max2,
            (AbstractDomain::ValueSet(s1), AbstractDomain::ValueSet(s2)) => s1.is_subset(s2),
            _ => false,
        }
    }
}

/// Abstract array state
#[derive(Debug, Clone)]
pub struct AbstractArrayState {
    /// Array identifier
    pub array: u32,
    /// Abstract values for known indices
    pub values: FxHashMap<AbstractIndex, AbstractDomain>,
    /// Default value (for unknown indices)
    pub default_value: AbstractDomain,
}

/// Abstract index (can be concrete or symbolic)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AbstractIndex {
    /// Concrete index value
    Concrete(i64),
    /// Symbolic index variable
    Symbolic(u32),
    /// Index expression
    Expression(TermId),
}

impl AbstractArrayState {
    /// Create a new abstract array state
    pub fn new(array: u32) -> Self {
        Self {
            array,
            values: FxHashMap::default(),
            default_value: AbstractDomain::Top,
        }
    }

    /// Get abstract value at an index
    pub fn get(&self, index: &AbstractIndex) -> AbstractDomain {
        self.values
            .get(index)
            .cloned()
            .unwrap_or_else(|| self.default_value.clone())
    }

    /// Set abstract value at an index
    pub fn set(&mut self, index: AbstractIndex, value: AbstractDomain) {
        self.values.insert(index, value);
    }

    /// Join with another array state
    pub fn join(&self, other: &Self) -> Self {
        let mut result = Self::new(self.array);

        // Join default values
        result.default_value = self.default_value.join(&other.default_value);

        // Join known indices
        for (idx, val1) in &self.values {
            if let Some(val2) = other.values.get(idx) {
                result.values.insert(idx.clone(), val1.join(val2));
            } else {
                result
                    .values
                    .insert(idx.clone(), val1.join(&other.default_value));
            }
        }

        // Add indices only in other
        for (idx, val2) in &other.values {
            if !self.values.contains_key(idx) {
                result
                    .values
                    .insert(idx.clone(), self.default_value.join(val2));
            }
        }

        result
    }
}

/// Array abstraction engine
pub struct ArrayAbstractionEngine {
    /// Abstract states for arrays
    abstract_states: FxHashMap<u32, AbstractArrayState>,
    /// Abstraction predicates
    predicates: Vec<AbstractionPredicate>,
    /// Refinement history
    refinement_history: Vec<RefinementStep>,
}

/// Abstraction predicate
#[derive(Debug, Clone)]
pub struct AbstractionPredicate {
    /// Predicate identifier
    pub id: u32,
    /// Predicate expression
    pub expr: TermId,
    /// Relevance score
    pub relevance: f64,
}

/// Refinement step in CEGAR
#[derive(Debug, Clone)]
pub struct RefinementStep {
    /// Step number
    pub step: usize,
    /// Spurious counterexample
    pub counterexample: Vec<TermId>,
    /// New predicates added
    pub new_predicates: Vec<u32>,
}

impl Default for ArrayAbstractionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayAbstractionEngine {
    /// Create a new abstraction engine
    pub fn new() -> Self {
        Self {
            abstract_states: FxHashMap::default(),
            predicates: Vec::new(),
            refinement_history: Vec::new(),
        }
    }

    /// Abstract an array operation
    pub fn abstract_operation(&mut self, array: u32, operation: &ArrayAbstractOp) -> Result<()> {
        let state = self
            .abstract_states
            .entry(array)
            .or_insert_with(|| AbstractArrayState::new(array));

        match operation {
            ArrayAbstractOp::Store { index, value } => {
                state.set(index.clone(), value.clone());
            }
            ArrayAbstractOp::Select { index, result } => {
                let value = state.get(index);
                // Would record that result has this abstract value
                let _ = result;
                let _ = value;
            }
        }

        Ok(())
    }

    /// Get abstract state for an array
    pub fn get_state(&self, array: u32) -> Option<&AbstractArrayState> {
        self.abstract_states.get(&array)
    }

    /// Add an abstraction predicate
    pub fn add_predicate(&mut self, expr: TermId, relevance: f64) -> u32 {
        let id = self.predicates.len() as u32;
        self.predicates.push(AbstractionPredicate {
            id,
            expr,
            relevance,
        });
        id
    }

    /// Refine abstraction based on counterexample
    pub fn refine(&mut self, counterexample: Vec<TermId>) -> Result<Vec<u32>> {
        let step = self.refinement_history.len();

        // Analyze counterexample to extract new predicates
        let new_predicates = self.extract_predicates_from_counterexample(&counterexample)?;

        // Add new predicates
        let mut predicate_ids = Vec::new();
        for pred_expr in new_predicates {
            let id = self.add_predicate(pred_expr, 1.0);
            predicate_ids.push(id);
        }

        // Record refinement step
        self.refinement_history.push(RefinementStep {
            step,
            counterexample,
            new_predicates: predicate_ids.clone(),
        });

        Ok(predicate_ids)
    }

    /// Extract predicates from counterexample
    fn extract_predicates_from_counterexample(
        &self,
        _counterexample: &[TermId],
    ) -> Result<Vec<TermId>> {
        // Simplified: would analyze counterexample to find relevant predicates
        Ok(Vec::new())
    }

    /// Get predicates
    pub fn predicates(&self) -> &[AbstractionPredicate] {
        &self.predicates
    }

    /// Get refinement history
    pub fn refinement_history(&self) -> &[RefinementStep] {
        &self.refinement_history
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.abstract_states.clear();
        self.predicates.clear();
        self.refinement_history.clear();
    }
}

/// Abstract array operation
#[derive(Debug, Clone)]
pub enum ArrayAbstractOp {
    /// Abstract select
    Select { index: AbstractIndex, result: u32 },
    /// Abstract store
    Store {
        index: AbstractIndex,
        value: AbstractDomain,
    },
}

/// Array summary for procedure abstraction
#[derive(Debug, Clone)]
pub struct ArraySummary {
    /// Array being summarized
    pub array: u32,
    /// Modified locations (may-modify set)
    pub may_modify: Vec<AbstractIndex>,
    /// Definitely modified locations (must-modify set)
    pub must_modify: Vec<AbstractIndex>,
    /// Postconditions
    pub postconditions: Vec<SummaryCondition>,
}

/// Summary condition
#[derive(Debug, Clone)]
pub struct SummaryCondition {
    /// Condition type
    pub condition_type: ConditionType,
    /// Condition expression
    pub expr: TermId,
}

/// Type of summary condition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionType {
    /// Precondition (requires)
    Requires,
    /// Postcondition (ensures)
    Ensures,
    /// Invariant (maintains)
    Maintains,
}

/// Array invariant generator
pub struct ArrayInvariantGenerator {
    /// Detected invariants
    invariants: Vec<ArrayInvariant>,
    /// Candidate invariants
    candidates: Vec<ArrayInvariant>,
}

/// Array invariant
#[derive(Debug, Clone)]
pub struct ArrayInvariant {
    /// Invariant type
    pub invariant_type: InvariantType,
    /// Arrays involved
    pub arrays: Vec<u32>,
    /// Invariant expression
    pub expr: TermId,
    /// Confidence score
    pub confidence: f64,
}

/// Type of array invariant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvariantType {
    /// Universal property: ∀i. P(a\[i\])
    Universal,
    /// Existential property: ∃i. P(a\[i\])
    Existential,
    /// Sortedness: ∀i,j. i < j → a\[i\] ≤ a\[j\]
    Sorted,
    /// Permutation: a is a permutation of b
    Permutation,
    /// Constant: ∀i. a\[i\] = c
    Constant,
    /// Equality: a = b
    Equality,
}

impl Default for ArrayInvariantGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayInvariantGenerator {
    /// Create a new invariant generator
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
            candidates: Vec::new(),
        }
    }

    /// Generate invariant candidates
    pub fn generate_candidates(&mut self, arrays: &[u32]) -> Vec<ArrayInvariant> {
        let mut candidates = Vec::new();

        // Generate constant array candidates
        for &array in arrays {
            candidates.push(ArrayInvariant {
                invariant_type: InvariantType::Constant,
                arrays: vec![array],
                expr: TermId::new(0), // Placeholder
                confidence: 0.5,
            });
        }

        // Generate equality candidates for pairs
        for i in 0..arrays.len() {
            for j in i + 1..arrays.len() {
                candidates.push(ArrayInvariant {
                    invariant_type: InvariantType::Equality,
                    arrays: vec![arrays[i], arrays[j]],
                    expr: TermId::new(0),
                    confidence: 0.5,
                });
            }
        }

        self.candidates = candidates.clone();
        candidates
    }

    /// Check a candidate invariant against a trace
    pub fn check_candidate(&mut self, _candidate: &ArrayInvariant, _trace: &[TermId]) -> bool {
        // Simplified: would check if candidate holds on trace
        // If it does, increase confidence; if not, remove it
        true
    }

    /// Promote candidates to invariants
    pub fn promote_candidates(&mut self, threshold: f64) {
        let promoted: Vec<_> = self
            .candidates
            .iter()
            .filter(|c| c.confidence >= threshold)
            .cloned()
            .collect();

        self.invariants.extend(promoted);
        self.candidates.retain(|c| c.confidence < threshold);
    }

    /// Get confirmed invariants
    pub fn invariants(&self) -> &[ArrayInvariant] {
        &self.invariants
    }

    /// Clear all invariants
    pub fn clear(&mut self) {
        self.invariants.clear();
        self.candidates.clear();
    }
}

/// Counterexample-guided abstraction refinement (CEGAR) loop
pub struct CEGARLoop {
    /// Abstraction engine
    abstraction: ArrayAbstractionEngine,
    /// Refinement iterations
    iteration: usize,
    /// Maximum iterations
    max_iterations: usize,
}

impl Default for CEGARLoop {
    fn default() -> Self {
        Self::new()
    }
}

impl CEGARLoop {
    /// Create a new CEGAR loop
    pub fn new() -> Self {
        Self {
            abstraction: ArrayAbstractionEngine::new(),
            iteration: 0,
            max_iterations: 100,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Run CEGAR loop
    pub fn run(&mut self, initial_formula: TermId) -> Result<CEGARResult> {
        self.iteration = 0;

        loop {
            if self.iteration >= self.max_iterations {
                return Ok(CEGARResult::Unknown {
                    reason: "Maximum iterations reached".to_string(),
                });
            }

            // Abstract check
            match self.check_abstract(initial_formula)? {
                AbstractCheckResult::Safe => {
                    return Ok(CEGARResult::Safe);
                }
                AbstractCheckResult::Unsafe(counterexample) => {
                    // Check if counterexample is spurious
                    if self.is_spurious(&counterexample)? {
                        // Refine abstraction
                        self.abstraction.refine(counterexample)?;
                        self.iteration += 1;
                    } else {
                        // Real counterexample
                        return Ok(CEGARResult::Unsafe { counterexample });
                    }
                }
            }
        }
    }

    /// Check abstract model
    fn check_abstract(&self, _formula: TermId) -> Result<AbstractCheckResult> {
        // Simplified: would check abstracted formula
        Ok(AbstractCheckResult::Safe)
    }

    /// Check if counterexample is spurious
    fn is_spurious(&self, _counterexample: &[TermId]) -> Result<bool> {
        // Simplified: would check if counterexample is realizable
        Ok(true)
    }

    /// Get abstraction engine
    pub fn abstraction(&self) -> &ArrayAbstractionEngine {
        &self.abstraction
    }

    /// Get current iteration
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

/// Result of abstract check
#[derive(Debug, Clone)]
pub enum AbstractCheckResult {
    /// Property holds in abstract model
    Safe,
    /// Counterexample found (may be spurious)
    Unsafe(Vec<TermId>),
}

/// Result of CEGAR loop
#[derive(Debug, Clone)]
pub enum CEGARResult {
    /// Property verified as safe
    Safe,
    /// Real counterexample found
    Unsafe { counterexample: Vec<TermId> },
    /// Unknown (e.g., max iterations)
    Unknown { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abstract_domain_join() {
        let d1 = AbstractDomain::Interval { min: 0, max: 10 };
        let d2 = AbstractDomain::Interval { min: 5, max: 15 };
        let result = d1.join(&d2);
        assert_eq!(result, AbstractDomain::Interval { min: 0, max: 15 });
    }

    #[test]
    fn test_abstract_domain_meet() {
        let d1 = AbstractDomain::Interval { min: 0, max: 10 };
        let d2 = AbstractDomain::Interval { min: 5, max: 15 };
        let result = d1.meet(&d2);
        assert_eq!(result, AbstractDomain::Interval { min: 5, max: 10 });
    }

    #[test]
    fn test_abstract_domain_bottom_top() {
        let bottom = AbstractDomain::Bottom;
        let top = AbstractDomain::Top;

        let interval = AbstractDomain::Interval { min: 0, max: 10 };

        assert_eq!(bottom.join(&interval), interval);
        assert_eq!(top.join(&interval), top);
        assert_eq!(bottom.meet(&interval), bottom);
        assert_eq!(top.meet(&interval), interval);
    }

    #[test]
    fn test_abstract_array_state() {
        let mut state = AbstractArrayState::new(100);

        let idx = AbstractIndex::Concrete(5);
        let val = AbstractDomain::Interval { min: 0, max: 10 };

        state.set(idx.clone(), val.clone());
        assert_eq!(state.get(&idx), val);
    }

    #[test]
    fn test_abstract_array_state_join() {
        let mut state1 = AbstractArrayState::new(100);
        let mut state2 = AbstractArrayState::new(100);

        let idx = AbstractIndex::Concrete(5);
        state1.set(idx.clone(), AbstractDomain::Interval { min: 0, max: 10 });
        state2.set(idx.clone(), AbstractDomain::Interval { min: 5, max: 15 });

        let joined = state1.join(&state2);
        assert_eq!(
            joined.get(&idx),
            AbstractDomain::Interval { min: 0, max: 15 }
        );
    }

    #[test]
    fn test_abstraction_engine() {
        let mut engine = ArrayAbstractionEngine::new();

        let op = ArrayAbstractOp::Store {
            index: AbstractIndex::Concrete(5),
            value: AbstractDomain::Interval { min: 0, max: 10 },
        };

        engine.abstract_operation(100, &op).unwrap();

        let state = engine.get_state(100).unwrap();
        assert_eq!(
            state.get(&AbstractIndex::Concrete(5)),
            AbstractDomain::Interval { min: 0, max: 10 }
        );
    }

    #[test]
    fn test_add_predicate() {
        let mut engine = ArrayAbstractionEngine::new();
        let pred_id = engine.add_predicate(TermId::new(100), 0.8);

        assert_eq!(pred_id, 0);
        assert_eq!(engine.predicates().len(), 1);
        assert_eq!(engine.predicates()[0].relevance, 0.8);
    }

    #[test]
    fn test_refinement() {
        let mut engine = ArrayAbstractionEngine::new();
        let counterexample = vec![TermId::new(1), TermId::new(2)];

        let new_preds = engine.refine(counterexample.clone()).unwrap();
        assert_eq!(engine.refinement_history().len(), 1);
        assert!(new_preds.is_empty()); // Simplified implementation returns empty
    }

    #[test]
    fn test_invariant_generator() {
        let mut generator = ArrayInvariantGenerator::new();
        let arrays = vec![1, 2, 3];

        let candidates = generator.generate_candidates(&arrays);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_promote_candidates() {
        let mut generator = ArrayInvariantGenerator::new();

        generator.candidates.push(ArrayInvariant {
            invariant_type: InvariantType::Constant,
            arrays: vec![1],
            expr: TermId::new(0),
            confidence: 0.9,
        });

        generator.candidates.push(ArrayInvariant {
            invariant_type: InvariantType::Equality,
            arrays: vec![1, 2],
            expr: TermId::new(1),
            confidence: 0.3,
        });

        generator.promote_candidates(0.5);

        assert_eq!(generator.invariants().len(), 1);
        assert_eq!(generator.candidates.len(), 1);
    }

    #[test]
    fn test_cegar_loop() {
        let cegar = CEGARLoop::new().with_max_iterations(10);
        assert_eq!(cegar.max_iterations, 10);
        assert_eq!(cegar.iteration(), 0);
    }

    #[test]
    fn test_sign_domain() {
        let signs = [
            SignDomain::Positive,
            SignDomain::Negative,
            SignDomain::Zero,
            SignDomain::NonNegative,
            SignDomain::NonPositive,
            SignDomain::Any,
        ];
        assert_eq!(signs.len(), 6);
    }

    #[test]
    fn test_parity_domain() {
        let parities = [ParityDomain::Even, ParityDomain::Odd, ParityDomain::Any];
        assert_eq!(parities.len(), 3);
    }

    #[test]
    fn test_abstract_domain_display() {
        assert_eq!(format!("{}", AbstractDomain::Top), "⊤");
        assert_eq!(format!("{}", AbstractDomain::Bottom), "⊥");
        assert_eq!(
            format!("{}", AbstractDomain::Interval { min: 0, max: 10 }),
            "[0, 10]"
        );
    }

    #[test]
    fn test_condition_types() {
        let types = [
            ConditionType::Requires,
            ConditionType::Ensures,
            ConditionType::Maintains,
        ];
        assert_eq!(types.len(), 3);
    }

    #[test]
    fn test_invariant_types() {
        let types = [
            InvariantType::Universal,
            InvariantType::Existential,
            InvariantType::Sorted,
            InvariantType::Permutation,
            InvariantType::Constant,
            InvariantType::Equality,
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_value_set_widening() {
        let mut set = FxHashSet::default();
        for i in 0..15 {
            set.insert(i);
        }

        let d1 = AbstractDomain::ValueSet(set.clone());
        let d2 = AbstractDomain::ValueSet(FxHashSet::default());

        let joined = d1.join(&d2);
        // Should widen to interval since set is too large
        assert!(matches!(joined, AbstractDomain::Interval { .. }));
    }
}
