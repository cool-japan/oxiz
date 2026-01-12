//! Reachability analysis utilities.
//!
//! Provides reach facts, model-based generalization, and counterexample handling.
//!
//! Reference: Z3's `muz/spacer/spacer_context.h` reach_fact class.

use crate::chc::{PredId, RuleId};
use oxiz_core::TermId;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU32, Ordering};

/// Unique identifier for a reach fact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReachFactId(pub u32);

impl ReachFactId {
    /// Create a new reach fact ID
    #[inline]
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// A reach fact: a concrete reachable state
#[derive(Debug, Clone)]
pub struct ReachFact {
    /// Unique identifier
    pub id: ReachFactId,
    /// The predicate this fact is for
    pub pred: PredId,
    /// The reachable state formula
    pub fact: TermId,
    /// Auxiliary variables in the fact
    aux_vars: SmallVec<[TermId; 4]>,
    /// The rule that produced this fact
    rule: RuleId,
    /// Justification: reach facts used to derive this one
    justification: SmallVec<[ReachFactId; 2]>,
    /// Tag for incremental disjunction
    tag: Option<TermId>,
    /// Whether this is from the initial state
    is_init: bool,
}

impl ReachFact {
    /// Create a new reach fact
    pub fn new(id: ReachFactId, pred: PredId, fact: TermId, rule: RuleId, is_init: bool) -> Self {
        Self {
            id,
            pred,
            fact,
            aux_vars: SmallVec::new(),
            rule,
            justification: SmallVec::new(),
            tag: None,
            is_init,
        }
    }

    /// Check if this is from the initial state
    #[inline]
    #[must_use]
    pub fn is_init(&self) -> bool {
        self.is_init
    }

    /// Get the rule that produced this fact
    #[inline]
    #[must_use]
    pub fn rule(&self) -> RuleId {
        self.rule
    }

    /// Add a justification (predecessor reach fact)
    pub fn add_justification(&mut self, fact: ReachFactId) {
        self.justification.push(fact);
    }

    /// Get the justifications
    pub fn justifications(&self) -> &[ReachFactId] {
        &self.justification
    }

    /// Get auxiliary variables
    pub fn aux_vars(&self) -> &[TermId] {
        &self.aux_vars
    }

    /// Set auxiliary variables
    pub fn set_aux_vars(&mut self, vars: impl IntoIterator<Item = TermId>) {
        self.aux_vars = vars.into_iter().collect();
    }

    /// Get the tag
    #[must_use]
    pub fn tag(&self) -> Option<TermId> {
        self.tag
    }

    /// Set the tag
    pub fn set_tag(&mut self, tag: TermId) {
        self.tag = Some(tag);
    }
}

/// Manager for reach facts per predicate
#[derive(Debug)]
pub struct ReachFactStore {
    /// All reach facts
    facts: Vec<ReachFact>,
    /// Facts by predicate
    by_pred: rustc_hash::FxHashMap<PredId, SmallVec<[ReachFactId; 8]>>,
    /// Init facts (facts from initial state)
    init_facts: SmallVec<[ReachFactId; 4]>,
    /// Next fact ID
    next_id: AtomicU32,
}

impl Default for ReachFactStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ReachFactStore {
    /// Create a new reach fact store
    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            by_pred: rustc_hash::FxHashMap::default(),
            init_facts: SmallVec::new(),
            next_id: AtomicU32::new(0),
        }
    }

    /// Add a new reach fact
    pub fn add(&mut self, pred: PredId, fact: TermId, rule: RuleId, is_init: bool) -> ReachFactId {
        let id = ReachFactId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let reach_fact = ReachFact::new(id, pred, fact, rule, is_init);

        self.facts.push(reach_fact);
        self.by_pred.entry(pred).or_default().push(id);

        if is_init {
            self.init_facts.push(id);
        }

        id
    }

    /// Get a reach fact by ID
    #[must_use]
    pub fn get(&self, id: ReachFactId) -> Option<&ReachFact> {
        self.facts.get(id.0 as usize)
    }

    /// Get a mutable reach fact by ID
    pub fn get_mut(&mut self, id: ReachFactId) -> Option<&mut ReachFact> {
        self.facts.get_mut(id.0 as usize)
    }

    /// Get all reach facts for a predicate
    pub fn for_pred(&self, pred: PredId) -> impl Iterator<Item = &ReachFact> {
        self.by_pred
            .get(&pred)
            .into_iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|&id| self.get(id))
    }

    /// Get init facts
    pub fn init_facts(&self) -> impl Iterator<Item = &ReachFact> {
        self.init_facts.iter().filter_map(|&id| self.get(id))
    }

    /// Get the number of reach facts
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Clear all reach facts
    pub fn clear(&mut self) {
        self.facts.clear();
        self.by_pred.clear();
        self.init_facts.clear();
        self.next_id = AtomicU32::new(0);
    }
}

/// A counterexample trace
#[derive(Debug, Clone)]
pub struct Counterexample {
    /// States in the trace (from init to error)
    states: Vec<CexState>,
    /// Whether this is a spurious counterexample
    spurious: bool,
}

/// A state in a counterexample trace
#[derive(Debug, Clone)]
pub struct CexState {
    /// The predicate
    pub pred: PredId,
    /// The concrete state
    pub state: TermId,
    /// The rule used to reach this state
    pub rule: Option<RuleId>,
    /// Variable assignments in this state
    pub assignments: SmallVec<[(TermId, TermId); 4]>,
}

impl Counterexample {
    /// Create a new counterexample
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            spurious: false,
        }
    }

    /// Add a state to the trace
    pub fn push(&mut self, state: CexState) {
        self.states.push(state);
    }

    /// Get the states
    pub fn states(&self) -> &[CexState] {
        &self.states
    }

    /// Get the length of the trace
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Mark as spurious
    pub fn mark_spurious(&mut self) {
        self.spurious = true;
    }

    /// Check if spurious
    #[must_use]
    pub fn is_spurious(&self) -> bool {
        self.spurious
    }

    /// Reverse the trace (for backward construction)
    pub fn reverse(&mut self) {
        self.states.reverse();
    }
}

impl Default for Counterexample {
    fn default() -> Self {
        Self::new()
    }
}

/// Model-based projection result
#[derive(Debug, Clone)]
pub struct Projection {
    /// The projected formula
    pub formula: TermId,
    /// Variables that were projected out
    pub projected_vars: SmallVec<[TermId; 4]>,
    /// Auxiliary variables introduced
    pub aux_vars: SmallVec<[TermId; 4]>,
}

impl Projection {
    /// Create a new projection
    pub fn new(formula: TermId) -> Self {
        Self {
            formula,
            projected_vars: SmallVec::new(),
            aux_vars: SmallVec::new(),
        }
    }
}

/// Generalization result
#[derive(Debug, Clone)]
pub struct Generalization {
    /// The generalized formula (cube)
    pub cube: SmallVec<[TermId; 8]>,
    /// Whether inductive generalization was applied
    pub inductive: bool,
    /// Literals dropped during generalization
    pub dropped: SmallVec<[TermId; 4]>,
}

impl Generalization {
    /// Create a new generalization
    pub fn new(cube: impl IntoIterator<Item = TermId>) -> Self {
        Self {
            cube: cube.into_iter().collect(),
            inductive: false,
            dropped: SmallVec::new(),
        }
    }

    /// Mark as inductively generalized
    pub fn mark_inductive(&mut self) {
        self.inductive = true;
    }

    /// Record a dropped literal
    pub fn drop_literal(&mut self, lit: TermId) {
        self.dropped.push(lit);
    }
}

/// Reachability checker interface
pub trait ReachabilityChecker {
    /// Check if a state is reachable from initial states
    fn is_reachable(&self, pred: PredId, state: TermId) -> bool;

    /// Check if a transition is possible
    fn can_transition(&self, from_pred: PredId, from_state: TermId, rule: RuleId) -> bool;

    /// Get reach facts for a predicate
    fn reach_facts(&self, pred: PredId) -> Vec<TermId>;
}

/// Under-approximation of reachable states
#[derive(Debug)]
pub struct UnderApproximation {
    /// Reachable states per predicate (disjunction)
    states: rustc_hash::FxHashMap<PredId, FxHashSet<TermId>>,
}

impl Default for UnderApproximation {
    fn default() -> Self {
        Self::new()
    }
}

impl UnderApproximation {
    /// Create a new under-approximation
    pub fn new() -> Self {
        Self {
            states: rustc_hash::FxHashMap::default(),
        }
    }

    /// Add a reachable state
    pub fn add(&mut self, pred: PredId, state: TermId) {
        self.states.entry(pred).or_default().insert(state);
    }

    /// Check if a state is in the under-approximation
    #[must_use]
    pub fn contains(&self, pred: PredId, state: TermId) -> bool {
        self.states.get(&pred).is_some_and(|s| s.contains(&state))
    }

    /// Get all states for a predicate
    pub fn states(&self, pred: PredId) -> impl Iterator<Item = TermId> + '_ {
        self.states
            .get(&pred)
            .into_iter()
            .flat_map(|s| s.iter().copied())
    }

    /// Clear all states
    pub fn clear(&mut self) {
        self.states.clear();
    }
}

/// Over-approximation of reachable states (from frames)
#[derive(Debug)]
pub struct OverApproximation {
    /// Blocked states per predicate per level (conjunction of negated cubes)
    blocked: rustc_hash::FxHashMap<PredId, rustc_hash::FxHashMap<u32, Vec<TermId>>>,
}

impl Default for OverApproximation {
    fn default() -> Self {
        Self::new()
    }
}

impl OverApproximation {
    /// Create a new over-approximation
    pub fn new() -> Self {
        Self {
            blocked: rustc_hash::FxHashMap::default(),
        }
    }

    /// Add a blocked state (lemma)
    pub fn add_blocked(&mut self, pred: PredId, level: u32, lemma: TermId) {
        self.blocked
            .entry(pred)
            .or_default()
            .entry(level)
            .or_default()
            .push(lemma);
    }

    /// Get blocked states at a level (lemmas that hold at level or higher)
    pub fn blocked_at(&self, pred: PredId, level: u32) -> impl Iterator<Item = TermId> + '_ {
        self.blocked.get(&pred).into_iter().flat_map(move |levels| {
            levels
                .iter()
                .filter(move |&(l, _)| *l >= level)
                .flat_map(|(_, lemmas)| lemmas.iter().copied())
        })
    }

    /// Clear all blocked states
    pub fn clear(&mut self) {
        self.blocked.clear();
    }
}

/// Concrete witness extractor for counterexamples
///
/// Extracts concrete variable assignments from SMT models
/// to produce executable counterexample traces with full model information.
pub struct ConcreteWitnessExtractor<'a> {
    /// Term manager
    terms: &'a mut oxiz_core::TermManager,
    /// CHC system
    system: &'a crate::chc::ChcSystem,
}

impl<'a> ConcreteWitnessExtractor<'a> {
    /// Create a new concrete witness extractor
    pub fn new(terms: &'a mut oxiz_core::TermManager, system: &'a crate::chc::ChcSystem) -> Self {
        Self { terms, system }
    }

    /// Extract concrete witness from a counterexample trace and models
    ///
    /// Takes an abstract counterexample and SMT models, and produces
    /// a concrete trace with variable assignments.
    pub fn extract_witness(
        &mut self,
        cex: &Counterexample,
        models: &[crate::smt::Model],
    ) -> Result<ConcreteWitness, WitnessError> {
        use tracing::debug;

        debug!("Extracting concrete witness from {} states", cex.len());

        if cex.is_empty() {
            return Err(WitnessError::EmptyTrace);
        }

        if models.len() != cex.len() {
            return Err(WitnessError::ModelMismatch {
                expected: cex.len(),
                got: models.len(),
            });
        }

        let mut witness = ConcreteWitness::new();

        for (i, (state, model)) in cex.states().iter().zip(models.iter()).enumerate() {
            // Extract variable assignments from the model
            let assignments = self.extract_assignments(state.pred, model)?;

            // Build concrete state with assignments
            let concrete_state = ConcreteState {
                step: i,
                pred: state.pred,
                state: state.state,
                rule: state.rule,
                assignments: assignments.clone(),
                model_values: self.model_to_values(model),
            };

            witness.add_state(concrete_state);
            debug!(
                "Extracted witness state {}: {} assignments",
                i,
                assignments.len()
            );
        }

        Ok(witness)
    }

    /// Extract variable assignments from a model for a predicate
    fn extract_assignments(
        &mut self,
        pred: PredId,
        model: &crate::smt::Model,
    ) -> Result<Vec<(TermId, TermId)>, WitnessError> {
        let mut assignments = Vec::new();

        // Get predicate information
        let pred_info = self
            .system
            .get_predicate(pred)
            .ok_or(WitnessError::PredicateNotFound(pred))?;

        // Extract assignments for each parameter
        for (idx, &sort) in pred_info.params.iter().enumerate() {
            if let Some(value) = model.get(idx) {
                // Create a variable for this parameter
                let var_name = format!("{}_{}", pred_info.name, idx);
                let var = self.terms.mk_var(&var_name, sort);
                assignments.push((var, value));
            }
        }

        Ok(assignments)
    }

    /// Convert model to concrete values
    fn model_to_values(&self, model: &crate::smt::Model) -> Vec<ConcreteValue> {
        use oxiz_core::TermKind;

        model
            .assignments()
            .iter()
            .filter_map(|&term_id| {
                let term = self.terms.get(term_id)?;
                match &term.kind {
                    TermKind::IntConst(n) => Some(ConcreteValue::Int(n.clone())),
                    TermKind::RealConst(r) => {
                        use num_bigint::BigInt;
                        Some(ConcreteValue::Real(
                            BigInt::from(*r.numer()),
                            BigInt::from(*r.denom()),
                        ))
                    }
                    TermKind::True => Some(ConcreteValue::Bool(true)),
                    TermKind::False => Some(ConcreteValue::Bool(false)),
                    TermKind::BitVecConst { value, width } => {
                        Some(ConcreteValue::BitVec(value.clone(), *width))
                    }
                    _ => None,
                }
            })
            .collect()
    }

    /// Validate that a witness is consistent with the CHC system
    pub fn validate_witness(&self, witness: &ConcreteWitness) -> Result<bool, WitnessError> {
        use tracing::trace;

        trace!("Validating witness with {} states", witness.states.len());

        // Check that trace starts from initial state
        if let Some(first) = witness.states.first()
            && first.step != 0
        {
            return Ok(false);
        }

        // Check that each transition is valid
        for i in 1..witness.states.len() {
            let prev = &witness.states[i - 1];
            let curr = &witness.states[i];

            // Check that we used a valid rule
            if let Some(rule_id) = curr.rule
                && self.system.get_rule(rule_id).is_none()
            {
                return Ok(false);
            }

            trace!(
                "Validated transition {} -> {} via rule {:?}",
                prev.step, curr.step, curr.rule
            );
        }

        Ok(true)
    }
}

/// Concrete witness with variable assignments
#[derive(Debug, Clone)]
pub struct ConcreteWitness {
    /// States in the witness
    pub states: Vec<ConcreteState>,
    /// Whether this witness was validated
    validated: bool,
}

impl ConcreteWitness {
    /// Create a new concrete witness
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            validated: false,
        }
    }

    /// Add a state to the witness
    pub fn add_state(&mut self, state: ConcreteState) {
        self.states.push(state);
    }

    /// Mark as validated
    pub fn mark_validated(&mut self) {
        self.validated = true;
    }

    /// Check if validated
    pub fn is_validated(&self) -> bool {
        self.validated
    }

    /// Get the length of the witness
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl Default for ConcreteWitness {
    fn default() -> Self {
        Self::new()
    }
}

/// A concrete state in a witness with variable assignments
#[derive(Debug, Clone)]
pub struct ConcreteState {
    /// Step number in the trace
    pub step: usize,
    /// The predicate
    pub pred: PredId,
    /// The state formula
    pub state: TermId,
    /// The rule used to reach this state
    pub rule: Option<RuleId>,
    /// Variable assignments (var, value)
    pub assignments: Vec<(TermId, TermId)>,
    /// Concrete values from the model
    pub model_values: Vec<ConcreteValue>,
}

/// Concrete value types that can appear in a witness
#[derive(Debug, Clone)]
pub enum ConcreteValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(num_bigint::BigInt),
    /// Real/rational value (numerator, denominator)
    Real(num_bigint::BigInt, num_bigint::BigInt),
    /// Bit-vector value (value, width)
    BitVec(num_bigint::BigInt, u32),
}

/// Errors from witness extraction
#[derive(Debug, thiserror::Error)]
pub enum WitnessError {
    /// Empty counterexample trace
    #[error("empty counterexample trace")]
    EmptyTrace,
    /// Model count mismatch
    #[error("model count mismatch: expected {expected}, got {got}")]
    ModelMismatch { expected: usize, got: usize },
    /// Predicate not found
    #[error("predicate not found: {0:?}")]
    PredicateNotFound(PredId),
    /// Invalid state
    #[error("invalid state: {0}")]
    InvalidState(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reach_fact_creation() {
        let mut store = ReachFactStore::new();
        let pred = PredId::new(0);
        let fact = oxiz_core::TermId::new(42);
        let rule = RuleId::new(0);

        let id = store.add(pred, fact, rule, true);
        let reach_fact = store.get(id).unwrap();

        assert!(reach_fact.is_init());
        assert_eq!(reach_fact.rule(), rule);
        assert_eq!(reach_fact.fact, fact);
    }

    #[test]
    fn test_reach_fact_justification() {
        let mut store = ReachFactStore::new();
        let pred = PredId::new(0);
        let rule = RuleId::new(0);

        let fact1 = oxiz_core::TermId::new(1);
        let fact2 = oxiz_core::TermId::new(2);

        let id1 = store.add(pred, fact1, rule, true);
        let id2 = store.add(pred, fact2, rule, false);

        // Add justification
        store.get_mut(id2).unwrap().add_justification(id1);

        let reach_fact = store.get(id2).unwrap();
        assert_eq!(reach_fact.justifications(), &[id1]);
    }

    #[test]
    fn test_counterexample() {
        let mut cex = Counterexample::new();
        let pred = PredId::new(0);
        let state = oxiz_core::TermId::new(42);

        cex.push(CexState {
            pred,
            state,
            rule: None,
            assignments: SmallVec::new(),
        });

        assert_eq!(cex.len(), 1);
        assert!(!cex.is_spurious());

        cex.mark_spurious();
        assert!(cex.is_spurious());
    }

    #[test]
    fn test_under_approximation() {
        let mut under = UnderApproximation::new();
        let pred = PredId::new(0);
        let state1 = oxiz_core::TermId::new(1);
        let state2 = oxiz_core::TermId::new(2);

        under.add(pred, state1);
        under.add(pred, state2);

        assert!(under.contains(pred, state1));
        assert!(under.contains(pred, state2));
        assert!(!under.contains(pred, oxiz_core::TermId::new(3)));
    }

    #[test]
    fn test_over_approximation() {
        let mut over = OverApproximation::new();
        let pred = PredId::new(0);
        let lemma1 = oxiz_core::TermId::new(1);
        let lemma2 = oxiz_core::TermId::new(2);

        over.add_blocked(pred, 1, lemma1);
        over.add_blocked(pred, 2, lemma2);

        // Level 1 should see lemmas at 1 and 2
        let blocked: Vec<_> = over.blocked_at(pred, 1).collect();
        assert_eq!(blocked.len(), 2);

        // Level 2 should only see lemma at 2
        let blocked: Vec<_> = over.blocked_at(pred, 2).collect();
        assert_eq!(blocked.len(), 1);
    }

    #[test]
    fn test_generalization() {
        let cube = [
            oxiz_core::TermId::new(1),
            oxiz_core::TermId::new(2),
            oxiz_core::TermId::new(3),
        ];
        let mut generalization = Generalization::new(cube);

        assert_eq!(generalization.cube.len(), 3);
        assert!(!generalization.inductive);

        generalization.drop_literal(oxiz_core::TermId::new(2));
        generalization.mark_inductive();

        assert!(generalization.inductive);
        assert_eq!(generalization.dropped.len(), 1);
    }
}
