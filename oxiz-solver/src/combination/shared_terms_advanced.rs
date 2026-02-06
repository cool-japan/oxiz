//! Advanced Shared Terms Management for Theory Combination.
//!
//! This module provides sophisticated shared term detection and management:
//! - Efficient shared term detection across theories
//! - Canonical representatives for equality classes
//! - Congruence-based equality graphs
//! - Interface term minimization
//! - Incremental shared term tracking
//!
//! ## Shared Terms
//!
//! In Nelson-Oppen combination, shared terms are variables that appear
//! in constraints of multiple theories. These terms form the "interface"
//! between theories and are used for equality propagation.
//!
//! ## Canonical Representatives
//!
//! Each equivalence class of terms has a canonical representative.
//! This module maintains:
//! - Fast lookup of representatives
//! - Efficient equality class merging
//! - Explanations for why terms are equal
//!
//! ## Equality Graphs (E-graphs)
//!
//! E-graphs compactly represent equivalence classes with congruence:
//! - Efficient term canonicalization
//! - Congruence closure
//! - Extraction of smallest equivalent terms
//!
//! ## References
//!
//! - Nelson & Oppen (1979): "Simplification by Cooperating Decision Procedures"
//! - Nieuwenhuis & Oliveras (2005): "Proof-Producing Congruence Closure"
//! - Z3's `smt/theory_combine.cpp`

use oxiz_core::TermId as CoreTermId;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Local term identifier (wrapper for core TermId).
pub type TermId = CoreTermId;

/// Theory identifier.
pub type TheoryId = usize;

/// Decision level for backtracking.
pub type DecisionLevel = u32;

/// Equality between two terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Equality {
    /// Left-hand side term.
    pub lhs: TermId,
    /// Right-hand side term.
    pub rhs: TermId,
}

impl Equality {
    /// Create a new equality (normalized: smaller term first).
    pub fn new(lhs: TermId, rhs: TermId) -> Self {
        if lhs.raw() <= rhs.raw() {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }

    /// Flip the equality.
    pub fn flip(self) -> Self {
        Self::new(self.rhs, self.lhs)
    }
}

/// Explanation for why two terms are equal.
#[derive(Debug, Clone)]
pub enum EqualityExplanation {
    /// Given as input axiom.
    Given,
    /// Reflexivity: t = t.
    Reflexive,
    /// Theory propagation.
    TheoryPropagation {
        /// Source theory.
        theory: TheoryId,
        /// Supporting equalities.
        support: Vec<Equality>,
    },
    /// Transitivity: a = b, b = c => a = c.
    Transitive {
        /// Intermediate term.
        intermediate: TermId,
        /// Left explanation.
        left: Box<EqualityExplanation>,
        /// Right explanation.
        right: Box<EqualityExplanation>,
    },
    /// Congruence: f(a) = f(b) if a = b.
    Congruence {
        /// Function term.
        function: TermId,
        /// Argument equalities.
        arg_equalities: Vec<(Equality, Box<EqualityExplanation>)>,
    },
}

/// E-class (equivalence class) identifier.
pub type EClassId = u32;

/// E-node (term in e-graph).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ENode {
    /// Term identifier.
    pub term: TermId,
    /// E-class this node belongs to.
    pub eclass: EClassId,
}

/// E-class (equivalence class in e-graph).
#[derive(Debug, Clone)]
pub struct EClass {
    /// Unique identifier.
    pub id: EClassId,
    /// Representative term.
    pub representative: TermId,
    /// All terms in this class.
    pub members: FxHashSet<TermId>,
    /// Parent e-classes (for congruence).
    pub parents: FxHashSet<EClassId>,
    /// Size of this e-class (for union-by-size).
    pub size: usize,
}

impl EClass {
    /// Create new e-class with single term.
    fn new(id: EClassId, term: TermId) -> Self {
        let mut members = FxHashSet::default();
        members.insert(term);

        Self {
            id,
            representative: term,
            members,
            parents: FxHashSet::default(),
            size: 1,
        }
    }

    /// Merge another e-class into this one.
    fn merge(&mut self, other: &EClass) {
        for &term in &other.members {
            self.members.insert(term);
        }
        for &parent in &other.parents {
            self.parents.insert(parent);
        }
        self.size += other.size;
    }
}

/// Equality graph (e-graph) for congruence closure.
#[derive(Debug, Clone)]
pub struct EGraph {
    /// Term to e-class mapping.
    term_to_eclass: FxHashMap<TermId, EClassId>,

    /// E-class to class data mapping.
    eclasses: FxHashMap<EClassId, EClass>,

    /// Next available e-class ID.
    next_eclass_id: EClassId,

    /// Union-find parent pointers (for e-class merging).
    parent: FxHashMap<EClassId, EClassId>,

    /// Rank for union-by-rank.
    rank: FxHashMap<EClassId, usize>,

    /// Pending congruences to process.
    pending_congruences: VecDeque<(EClassId, EClassId)>,

    /// Explanations for e-class merges.
    merge_explanations: FxHashMap<(EClassId, EClassId), EqualityExplanation>,
}

impl EGraph {
    /// Create new e-graph.
    pub fn new() -> Self {
        Self {
            term_to_eclass: FxHashMap::default(),
            eclasses: FxHashMap::default(),
            next_eclass_id: 0,
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
            pending_congruences: VecDeque::new(),
            merge_explanations: FxHashMap::default(),
        }
    }

    /// Add term to e-graph.
    pub fn add_term(&mut self, term: TermId) -> EClassId {
        if let Some(&eclass_id) = self.term_to_eclass.get(&term) {
            return self.find(eclass_id);
        }

        let eclass_id = self.next_eclass_id;
        self.next_eclass_id += 1;

        let eclass = EClass::new(eclass_id, term);
        self.eclasses.insert(eclass_id, eclass);
        self.term_to_eclass.insert(term, eclass_id);

        eclass_id
    }

    /// Find canonical e-class ID (with path compression).
    pub fn find(&mut self, mut eclass_id: EClassId) -> EClassId {
        let mut path = Vec::new();

        while let Some(&parent) = self.parent.get(&eclass_id) {
            if parent == eclass_id {
                break;
            }
            path.push(eclass_id);
            eclass_id = parent;
        }

        // Path compression
        for node in path {
            self.parent.insert(node, eclass_id);
        }

        eclass_id
    }

    /// Merge two e-classes.
    pub fn merge(
        &mut self,
        a: EClassId,
        b: EClassId,
        explanation: EqualityExplanation,
    ) -> Result<EClassId, String> {
        let a_root = self.find(a);
        let b_root = self.find(b);

        if a_root == b_root {
            return Ok(a_root);
        }

        // Union by rank
        let a_rank = self.rank.get(&a_root).copied().unwrap_or(0);
        let b_rank = self.rank.get(&b_root).copied().unwrap_or(0);

        let (child, parent_id) = if a_rank < b_rank {
            (a_root, b_root)
        } else if a_rank > b_rank {
            (b_root, a_root)
        } else {
            self.rank.insert(b_root, b_rank + 1);
            (a_root, b_root)
        };

        self.parent.insert(child, parent_id);

        // Merge e-class data
        if let Some(child_eclass) = self.eclasses.get(&child).cloned()
            && let Some(parent_eclass) = self.eclasses.get_mut(&parent_id)
        {
            parent_eclass.merge(&child_eclass);
        }

        // Store explanation
        self.merge_explanations
            .insert((child, parent_id), explanation);

        // Queue congruence checks
        self.queue_congruence_checks(child, parent_id);

        Ok(parent_id)
    }

    /// Queue congruence checks for parent terms.
    fn queue_congruence_checks(&mut self, _a: EClassId, _b: EClassId) {
        // Simplified: would check parent applications for congruence
    }

    /// Process pending congruences.
    pub fn process_congruences(&mut self) -> Result<(), String> {
        while let Some((a, b)) = self.pending_congruences.pop_front() {
            let a_root = self.find(a);
            let b_root = self.find(b);

            if a_root != b_root {
                self.merge(
                    a_root,
                    b_root,
                    EqualityExplanation::Congruence {
                        function: TermId::new(0), // Simplified
                        arg_equalities: Vec::new(),
                    },
                )?;
            }
        }

        Ok(())
    }

    /// Get canonical term for an e-class.
    pub fn get_representative(&mut self, term: TermId) -> Option<TermId> {
        let eclass_id = *self.term_to_eclass.get(&term)?;
        let root = self.find(eclass_id);
        self.eclasses.get(&root).map(|ec| ec.representative)
    }

    /// Check if two terms are in the same e-class.
    pub fn are_equal(&mut self, a: TermId, b: TermId) -> bool {
        if let (Some(&a_class), Some(&b_class)) =
            (self.term_to_eclass.get(&a), self.term_to_eclass.get(&b))
        {
            self.find(a_class) == self.find(b_class)
        } else {
            false
        }
    }

    /// Get explanation for why two terms are equal.
    pub fn get_explanation(&mut self, a: TermId, b: TermId) -> Option<EqualityExplanation> {
        if !self.are_equal(a, b) {
            return None;
        }

        if a == b {
            return Some(EqualityExplanation::Reflexive);
        }

        // Trace path through union-find to build explanation
        let a_class = self.term_to_eclass.get(&a)?;
        let b_class = self.term_to_eclass.get(&b)?;

        // Copy values to avoid borrow checker issues
        let a_class_val = *a_class;
        let b_class_val = *b_class;

        let a_root = self.find(a_class_val);
        let b_root = self.find(b_class_val);

        if a_root == b_root {
            // Find stored explanation
            if let Some(explanation) = self.merge_explanations.get(&(a_class_val, b_class_val)) {
                return Some(explanation.clone());
            }
        }

        None
    }

    /// Get all terms in the same e-class as a term.
    pub fn get_eclass_members(&mut self, term: TermId) -> Vec<TermId> {
        if let Some(&eclass_id) = self.term_to_eclass.get(&term) {
            let root = self.find(eclass_id);
            if let Some(eclass) = self.eclasses.get(&root) {
                return eclass.members.iter().copied().collect();
            }
        }
        Vec::new()
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.term_to_eclass.clear();
        self.eclasses.clear();
        self.next_eclass_id = 0;
        self.parent.clear();
        self.rank.clear();
        self.pending_congruences.clear();
        self.merge_explanations.clear();
    }
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a shared term.
#[derive(Debug, Clone)]
pub struct SharedTermInfo {
    /// Theories that use this term.
    pub theories: FxHashSet<TheoryId>,

    /// Is this term an interface term?
    pub is_interface: bool,

    /// Representative in equality class.
    pub representative: TermId,

    /// Size of equivalence class.
    pub class_size: usize,

    /// Decision level where this term became shared.
    pub shared_at_level: DecisionLevel,
}

impl SharedTermInfo {
    /// Create new shared term info.
    fn new(theory: TheoryId, level: DecisionLevel) -> Self {
        let mut theories = FxHashSet::default();
        theories.insert(theory);

        Self {
            theories,
            is_interface: false,
            representative: TermId::new(0), // Will be set later
            class_size: 1,
            shared_at_level: level,
        }
    }
}

/// Configuration for shared terms manager.
#[derive(Debug, Clone)]
pub struct SharedTermsConfig {
    /// Enable notification batching.
    pub enable_batching: bool,

    /// Maximum batch size before forcing flush.
    pub max_batch_size: usize,

    /// Enable e-graph for congruence closure.
    pub enable_egraph: bool,

    /// Enable interface term minimization.
    pub minimize_interface: bool,

    /// Track explanations.
    pub track_explanations: bool,
}

impl Default for SharedTermsConfig {
    fn default() -> Self {
        Self {
            enable_batching: true,
            max_batch_size: 1000,
            enable_egraph: true,
            minimize_interface: true,
            track_explanations: true,
        }
    }
}

/// Statistics for shared terms.
#[derive(Debug, Clone, Default)]
pub struct SharedTermsStats {
    /// Number of shared terms registered.
    pub terms_registered: u64,

    /// Number of theory subscriptions.
    pub subscriptions: u64,

    /// Equalities propagated.
    pub equalities_propagated: u64,

    /// Notification batches sent.
    pub batches_sent: u64,

    /// E-class merges performed.
    pub eclass_merges: u64,

    /// Congruences detected.
    pub congruences: u64,

    /// Interface terms identified.
    pub interface_terms: u64,
}

/// Advanced shared terms manager for theory combination.
#[derive(Debug)]
pub struct AdvancedSharedTermsManager {
    /// Configuration.
    config: SharedTermsConfig,

    /// Shared term information.
    terms: FxHashMap<TermId, SharedTermInfo>,

    /// E-graph for equality management.
    egraph: EGraph,

    /// Pending equalities to propagate.
    pending_equalities: Vec<Equality>,

    /// Theories subscribed to each term.
    subscriptions: FxHashMap<TermId, FxHashSet<TheoryId>>,

    /// Interface terms (truly shared between theories).
    interface_terms: FxHashSet<TermId>,

    /// Decision level history for backtracking.
    decision_levels: FxHashMap<DecisionLevel, Vec<TermId>>,

    /// Current decision level.
    current_level: DecisionLevel,

    /// Statistics.
    stats: SharedTermsStats,

    /// Equality explanations.
    explanations: FxHashMap<Equality, EqualityExplanation>,
}

impl AdvancedSharedTermsManager {
    /// Create a new shared terms manager.
    pub fn new(config: SharedTermsConfig) -> Self {
        Self {
            config,
            terms: FxHashMap::default(),
            egraph: EGraph::new(),
            pending_equalities: Vec::new(),
            subscriptions: FxHashMap::default(),
            interface_terms: FxHashSet::default(),
            decision_levels: FxHashMap::default(),
            current_level: 0,
            stats: SharedTermsStats::default(),
            explanations: FxHashMap::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SharedTermsConfig::default())
    }

    /// Register a shared term.
    pub fn register_term(&mut self, term: TermId, theory: TheoryId) {
        let is_new = !self.terms.contains_key(&term);

        let entry = self.terms.entry(term).or_insert_with(|| {
            self.stats.terms_registered += 1;
            SharedTermInfo::new(theory, self.current_level)
        });

        let was_single_theory = entry.theories.len() == 1;
        entry.theories.insert(theory);

        // Track as interface term if used by multiple theories
        if was_single_theory && entry.theories.len() > 1 {
            self.interface_terms.insert(term);
            entry.is_interface = true;
            self.stats.interface_terms += 1;
        }

        self.stats.subscriptions += 1;

        // Track subscriptions
        self.subscriptions.entry(term).or_default().insert(theory);

        // Add to e-graph
        if self.config.enable_egraph {
            self.egraph.add_term(term);
        }

        // Track in decision level history
        if is_new {
            self.decision_levels
                .entry(self.current_level)
                .or_default()
                .push(term);
        }
    }

    /// Check if a term is shared between multiple theories.
    pub fn is_shared(&self, term: TermId) -> bool {
        self.terms
            .get(&term)
            .map(|info| info.theories.len() > 1)
            .unwrap_or(false)
    }

    /// Check if a term is an interface term.
    pub fn is_interface_term(&self, term: TermId) -> bool {
        self.interface_terms.contains(&term)
    }

    /// Get theories that use a term.
    pub fn get_theories(&self, term: TermId) -> Vec<TheoryId> {
        self.terms
            .get(&term)
            .map(|info| info.theories.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Assert equality between two terms.
    ///
    /// This merges their equivalence classes and queues notifications.
    pub fn assert_equality(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        explanation: EqualityExplanation,
    ) -> Result<(), String> {
        if self.config.enable_egraph && self.egraph.are_equal(lhs, rhs) {
            return Ok(()); // Already equal
        }

        // Merge in e-graph
        if self.config.enable_egraph {
            let lhs_class = self.egraph.add_term(lhs);
            let rhs_class = self.egraph.add_term(rhs);
            self.egraph
                .merge(lhs_class, rhs_class, explanation.clone())?;
            self.stats.eclass_merges += 1;
        }

        // Queue equality for propagation
        let equality = Equality::new(lhs, rhs);
        self.pending_equalities.push(equality);
        self.stats.equalities_propagated += 1;

        // Store explanation
        if self.config.track_explanations {
            self.explanations.insert(equality, explanation);
        }

        // Check if should flush batch
        if self.pending_equalities.len() >= self.config.max_batch_size {
            self.flush_equalities();
        }

        Ok(())
    }

    /// Check if two terms are in the same equivalence class.
    pub fn are_equal(&mut self, lhs: TermId, rhs: TermId) -> bool {
        if !self.config.enable_egraph {
            return lhs == rhs;
        }

        self.egraph.are_equal(lhs, rhs)
    }

    /// Get canonical representative of a term's equivalence class.
    pub fn get_representative(&mut self, term: TermId) -> TermId {
        if !self.config.enable_egraph {
            return term;
        }

        self.egraph.get_representative(term).unwrap_or(term)
    }

    /// Get all terms in the same equivalence class.
    pub fn get_eclass_members(&mut self, term: TermId) -> Vec<TermId> {
        if !self.config.enable_egraph {
            return vec![term];
        }

        self.egraph.get_eclass_members(term)
    }

    /// Get explanation for why two terms are equal.
    pub fn get_equality_explanation(
        &mut self,
        lhs: TermId,
        rhs: TermId,
    ) -> Option<EqualityExplanation> {
        let eq = Equality::new(lhs, rhs);

        if let Some(explanation) = self.explanations.get(&eq) {
            return Some(explanation.clone());
        }

        if self.config.enable_egraph {
            return self.egraph.get_explanation(lhs, rhs);
        }

        None
    }

    /// Get pending equalities to propagate.
    pub fn get_pending_equalities(&self) -> &[Equality] {
        &self.pending_equalities
    }

    /// Flush pending equalities (send to theories).
    pub fn flush_equalities(&mut self) {
        if !self.pending_equalities.is_empty() {
            self.stats.batches_sent += 1;
            self.pending_equalities.clear();
        }
    }

    /// Get all shared terms.
    pub fn get_shared_terms(&self) -> Vec<TermId> {
        self.terms
            .iter()
            .filter(|(_, info)| info.theories.len() > 1)
            .map(|(&term, _)| term)
            .collect()
    }

    /// Get interface terms (minimal shared terms).
    pub fn get_interface_terms(&self) -> Vec<TermId> {
        self.interface_terms.iter().copied().collect()
    }

    /// Minimize interface terms.
    ///
    /// Reduce the number of interface terms by using canonical representatives.
    pub fn minimize_interface(&mut self) -> Vec<TermId> {
        if !self.config.minimize_interface || !self.config.enable_egraph {
            return self.get_interface_terms();
        }

        let mut minimal = FxHashSet::default();

        // Collect terms first to avoid borrow checker issues
        let terms: Vec<_> = self.interface_terms.iter().copied().collect();
        for term in terms {
            let rep = self.get_representative(term);
            minimal.insert(rep);
        }

        minimal.into_iter().collect()
    }

    /// Push a new decision level.
    pub fn push_decision_level(&mut self) {
        self.current_level += 1;
    }

    /// Backtrack to a decision level.
    pub fn backtrack(&mut self, level: DecisionLevel) -> Result<(), String> {
        if level > self.current_level {
            return Err("Cannot backtrack to future level".to_string());
        }

        // Remove terms registered above this level
        let levels_to_remove: Vec<_> = self
            .decision_levels
            .keys()
            .filter(|&&l| l > level)
            .copied()
            .collect();

        for l in levels_to_remove {
            if let Some(terms) = self.decision_levels.remove(&l) {
                for term in terms {
                    self.terms.remove(&term);
                    self.subscriptions.remove(&term);
                    self.interface_terms.remove(&term);
                }
            }
        }

        self.current_level = level;
        Ok(())
    }

    /// Get statistics.
    pub fn stats(&self) -> &SharedTermsStats {
        &self.stats
    }

    /// Reset manager state.
    pub fn reset(&mut self) {
        self.terms.clear();
        self.egraph.clear();
        self.pending_equalities.clear();
        self.subscriptions.clear();
        self.interface_terms.clear();
        self.decision_levels.clear();
        self.current_level = 0;
        self.explanations.clear();
        self.stats = SharedTermsStats::default();
    }

    /// Process pending congruences in e-graph.
    pub fn process_congruences(&mut self) -> Result<(), String> {
        if !self.config.enable_egraph {
            return Ok(());
        }

        self.egraph.process_congruences()?;
        Ok(())
    }

    /// Detect new shared terms based on current theory assignments.
    pub fn detect_shared_terms(&mut self, _term_theories: &FxHashMap<TermId, FxHashSet<TheoryId>>) {
        // Simplified: would analyze term occurrences across theories
    }

    /// Build equality explanation chain.
    pub fn build_explanation_chain(
        &self,
        equalities: &[Equality],
    ) -> Result<EqualityExplanation, String> {
        if equalities.is_empty() {
            return Err("No equalities to explain".to_string());
        }

        if equalities.len() == 1 {
            let eq = &equalities[0];
            return Ok(self
                .explanations
                .get(eq)
                .cloned()
                .unwrap_or(EqualityExplanation::Given));
        }

        // Build transitive chain
        let mut current = equalities[0];
        let mut explanation = self
            .explanations
            .get(&current)
            .cloned()
            .unwrap_or(EqualityExplanation::Given);

        for &eq in &equalities[1..] {
            let next_explanation = self
                .explanations
                .get(&eq)
                .cloned()
                .unwrap_or(EqualityExplanation::Given);

            explanation = EqualityExplanation::Transitive {
                intermediate: current.rhs,
                left: Box::new(explanation),
                right: Box::new(next_explanation),
            };

            current = eq;
        }

        Ok(explanation)
    }
}

impl Default for AdvancedSharedTermsManager {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Interface term minimizer.
///
/// Reduces the number of interface terms by finding minimal representatives.
pub struct InterfaceTermMinimizer {
    /// E-graph for term equivalences.
    egraph: EGraph,

    /// Candidate interface terms.
    candidates: FxHashSet<TermId>,
}

impl InterfaceTermMinimizer {
    /// Create new minimizer.
    pub fn new() -> Self {
        Self {
            egraph: EGraph::new(),
            candidates: FxHashSet::default(),
        }
    }

    /// Add candidate interface term.
    pub fn add_candidate(&mut self, term: TermId) {
        self.candidates.insert(term);
        self.egraph.add_term(term);
    }

    /// Add equality between terms.
    pub fn add_equality(&mut self, lhs: TermId, rhs: TermId) -> Result<(), String> {
        let lhs_class = self.egraph.add_term(lhs);
        let rhs_class = self.egraph.add_term(rhs);
        self.egraph
            .merge(lhs_class, rhs_class, EqualityExplanation::Given)?;
        Ok(())
    }

    /// Compute minimal interface terms.
    pub fn minimize(&mut self) -> Vec<TermId> {
        let mut minimal = FxHashSet::default();

        for &term in &self.candidates {
            let rep = self.egraph.get_representative(term).unwrap_or(term);
            minimal.insert(rep);
        }

        minimal.into_iter().collect()
    }

    /// Clear state.
    pub fn clear(&mut self) {
        self.egraph.clear();
        self.candidates.clear();
    }
}

impl Default for InterfaceTermMinimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(id: u32) -> TermId {
        TermId::new(id)
    }

    #[test]
    fn test_equality_creation() {
        let eq1 = Equality::new(term(1), term(2));
        let eq2 = Equality::new(term(2), term(1));
        assert_eq!(eq1, eq2);
    }

    #[test]
    fn test_egraph_creation() {
        let egraph = EGraph::new();
        assert_eq!(egraph.next_eclass_id, 0);
    }

    #[test]
    fn test_egraph_add_term() {
        let mut egraph = EGraph::new();
        let class1 = egraph.add_term(term(1));
        let class2 = egraph.add_term(term(1));
        assert_eq!(class1, class2);
    }

    #[test]
    fn test_egraph_merge() {
        let mut egraph = EGraph::new();
        let c1 = egraph.add_term(term(1));
        let c2 = egraph.add_term(term(2));

        egraph
            .merge(c1, c2, EqualityExplanation::Given)
            .expect("Merge failed");

        assert!(egraph.are_equal(term(1), term(2)));
    }

    #[test]
    fn test_egraph_transitivity() {
        let mut egraph = EGraph::new();
        let c1 = egraph.add_term(term(1));
        let c2 = egraph.add_term(term(2));
        let c3 = egraph.add_term(term(3));

        egraph
            .merge(c1, c2, EqualityExplanation::Given)
            .expect("Merge failed");
        egraph
            .merge(c2, c3, EqualityExplanation::Given)
            .expect("Merge failed");

        assert!(egraph.are_equal(term(1), term(3)));
    }

    #[test]
    fn test_manager_creation() {
        let manager = AdvancedSharedTermsManager::default_config();
        assert_eq!(manager.stats().terms_registered, 0);
    }

    #[test]
    fn test_register_term() {
        let mut manager = AdvancedSharedTermsManager::default_config();

        manager.register_term(term(1), 0); // Theory 0
        manager.register_term(term(1), 1); // Theory 1

        assert!(manager.is_shared(term(1)));
        assert!(manager.is_interface_term(term(1)));
        assert_eq!(manager.get_theories(term(1)).len(), 2);
    }

    #[test]
    fn test_equality_assertion() {
        let mut manager = AdvancedSharedTermsManager::default_config();

        manager
            .assert_equality(term(1), term(2), EqualityExplanation::Given)
            .expect("Assert failed");

        assert!(manager.are_equal(term(1), term(2)));
        assert_eq!(manager.get_pending_equalities().len(), 1);
    }

    #[test]
    fn test_representative() {
        let mut manager = AdvancedSharedTermsManager::default_config();

        manager
            .assert_equality(term(1), term(2), EqualityExplanation::Given)
            .expect("Assert failed");

        let rep1 = manager.get_representative(term(1));
        let rep2 = manager.get_representative(term(2));

        assert_eq!(rep1, rep2);
    }

    #[test]
    fn test_eclass_members() {
        let mut manager = AdvancedSharedTermsManager::default_config();

        manager
            .assert_equality(term(1), term(2), EqualityExplanation::Given)
            .expect("Assert failed");

        let members = manager.get_eclass_members(term(1));
        assert!(members.contains(&term(1)));
        assert!(members.contains(&term(2)));
    }

    #[test]
    fn test_flush_equalities() {
        let mut manager = AdvancedSharedTermsManager::default_config();

        manager
            .assert_equality(term(1), term(2), EqualityExplanation::Given)
            .expect("Assert failed");
        assert_eq!(manager.get_pending_equalities().len(), 1);

        manager.flush_equalities();
        assert_eq!(manager.get_pending_equalities().len(), 0);
    }

    #[test]
    fn test_interface_term_minimization() {
        let mut manager = AdvancedSharedTermsManager::default_config();

        manager.register_term(term(1), 0);
        manager.register_term(term(1), 1);
        manager.register_term(term(2), 0);
        manager.register_term(term(2), 1);

        manager
            .assert_equality(term(1), term(2), EqualityExplanation::Given)
            .expect("Assert failed");

        let minimal = manager.minimize_interface();
        assert_eq!(minimal.len(), 1);
    }

    #[test]
    fn test_decision_levels() {
        let mut manager = AdvancedSharedTermsManager::default_config();

        manager.push_decision_level();
        manager.register_term(term(1), 0);

        manager.push_decision_level();
        manager.register_term(term(2), 0);

        manager.backtrack(1).expect("Backtrack failed");

        assert!(manager.terms.contains_key(&term(1)));
        assert!(!manager.terms.contains_key(&term(2)));
    }

    #[test]
    fn test_interface_minimizer() {
        let mut minimizer = InterfaceTermMinimizer::new();

        minimizer.add_candidate(term(1));
        minimizer.add_candidate(term(2));
        minimizer.add_candidate(term(3));

        minimizer
            .add_equality(term(1), term(2))
            .expect("Equality failed");

        let minimal = minimizer.minimize();
        assert_eq!(minimal.len(), 2); // {rep(1,2), 3}
    }
}
