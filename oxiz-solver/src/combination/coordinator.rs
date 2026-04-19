//! Theory Combination Coordinator
//!
//! This module coordinates multiple theory solvers using the Nelson-Oppen method
//! with optimizations:
//! - Lazy vs eager theory combination
//! - Shared term management
//! - Equality sharing between theories
//! - Conflict minimization across theories

#![allow(missing_docs)] // Under development

#[allow(unused_imports)]
use crate::prelude::*;
#[cfg(feature = "profiling")]
use oxiz_core::profiling::{ProfilingCategory, ScopedTimer};
#[cfg(feature = "std")]
use oxiz_core::TermId as ProofTermId;
#[cfg(feature = "std")]
use oxiz_proof::{CombinationStep, CombinationTheoryId, NelsonOppenCertificate, ProofNodeId};

/// Placeholder term identifier
pub type TermId = usize;

/// Theory identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TheoryId {
    Core,
    Arithmetic,
    BitVector,
    Array,
    Datatype,
    String,
    Uninterpreted,
}

/// Theory interface
pub trait TheorySolver {
    /// Get theory ID
    fn theory_id(&self) -> TheoryId;

    /// Assert a formula
    fn assert_formula(&mut self, formula: TermId) -> Result<(), String>;

    /// Check satisfiability
    fn check_sat(&mut self) -> Result<SatResult, String>;

    /// Get model (if SAT)
    fn get_model(&self) -> Option<FxHashMap<TermId, TermId>>;

    /// Get conflict explanation (if UNSAT)
    fn get_conflict(&self) -> Option<Vec<TermId>>;

    /// Backtrack to a level
    fn backtrack(&mut self, level: usize) -> Result<(), String>;

    /// Get implied equalities
    fn get_implied_equalities(&self) -> Vec<(TermId, TermId)>;

    /// Notify of external equality
    fn notify_equality(&mut self, lhs: TermId, rhs: TermId) -> Result<(), String>;
}

/// Satisfiability result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatResult {
    Sat,
    Unsat,
    Unknown,
}

/// Shared term between theories
#[derive(Debug, Clone)]
pub struct SharedTerm {
    /// The term
    pub term: TermId,
    /// Theories that use this term
    pub theories: FxHashSet<TheoryId>,
    /// Current equivalence class representative
    pub representative: TermId,
}

/// Equality propagation item
#[derive(Debug, Clone)]
pub struct EqualityProp {
    /// Left-hand side
    pub lhs: TermId,
    /// Right-hand side
    pub rhs: TermId,
    /// Source theory
    pub source: TheoryId,
    /// Explanation (justification)
    pub explanation: Vec<TermId>,
}

/// Statistics for theory combination
#[derive(Debug, Clone, Default)]
pub struct CoordinatorStats {
    pub check_sat_calls: u64,
    pub theory_conflicts: u64,
    pub equalities_propagated: u64,
    pub shared_terms_count: usize,
    pub theory_combination_rounds: u64,
}

/// Configuration for theory combination
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Use eager theory combination (propagate all equalities immediately)
    pub eager_combination: bool,
    /// Maximum theory combination rounds
    pub max_combination_rounds: usize,
    /// Enable conflict minimization across theories
    pub minimize_conflicts: bool,
    /// Enable theory-combination proof certificates.
    pub proof_mode: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            eager_combination: false,
            max_combination_rounds: 10,
            minimize_conflicts: true,
            proof_mode: false,
        }
    }
}

/// Theory combination coordinator
pub struct TheoryCoordinator {
    config: CoordinatorConfig,
    stats: CoordinatorStats,
    /// Registered theory solvers
    theories: FxHashMap<TheoryId, Box<dyn TheorySolver>>,
    /// Shared terms between theories
    shared_terms: FxHashMap<TermId, SharedTerm>,
    /// Pending equality propagations
    pending_equalities: VecDeque<EqualityProp>,
    /// Memoized implied equalities by theory and decision level.
    theory_propagation_cache: FxHashMap<(TheoryId, u32), Vec<EqualityProp>>,
    /// Equality propagation history for proof certificates.
    propagated_equalities_log: Vec<EqualityProp>,
    /// Last generated theory-combination certificate.
    #[cfg(feature = "std")]
    last_certificate: Option<NelsonOppenCertificate>,
    /// Current decision level
    current_level: usize,
}

impl TheoryCoordinator {
    /// Create a new theory coordinator
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            config,
            stats: CoordinatorStats::default(),
            theories: FxHashMap::default(),
            shared_terms: FxHashMap::default(),
            pending_equalities: VecDeque::new(),
            theory_propagation_cache: FxHashMap::default(),
            propagated_equalities_log: Vec::new(),
            #[cfg(feature = "std")]
            last_certificate: None,
            current_level: 0,
        }
    }

    /// Register a theory solver
    pub fn register_theory(&mut self, theory: Box<dyn TheorySolver>) {
        let theory_id = theory.theory_id();
        self.theories.insert(theory_id, theory);
    }

    /// Assert a formula to the appropriate theory
    pub fn assert_formula(&mut self, formula: TermId, theory: TheoryId) -> Result<(), String> {
        if let Some(solver) = self.theories.get_mut(&theory) {
            solver.assert_formula(formula)?;
            self.clear_from_level(self.current_level as u32);

            // Identify shared terms
            self.identify_shared_terms(formula)?;
        } else {
            return Err(format!("Theory {:?} not registered", theory));
        }

        Ok(())
    }

    /// Check satisfiability with theory combination
    pub fn check_sat(&mut self) -> Result<SatResult, String> {
        #[cfg(feature = "profiling")]
        let _timer = ScopedTimer::new(ProfilingCategory::TheoryCheck);
        self.stats.check_sat_calls += 1;

        // Phase 1: Check individual theories
        for solver in self.theories.values_mut() {
            let result = solver.check_sat()?;

            match result {
                SatResult::Unsat => {
                    self.stats.theory_conflicts += 1;
                    self.maybe_record_certificate_from_log();
                    return Ok(SatResult::Unsat);
                }
                SatResult::Unknown => {
                    return Ok(SatResult::Unknown);
                }
                SatResult::Sat => {
                    // Continue to next theory
                }
            }
        }

        // Phase 2: Theory combination via equality sharing
        if self.config.eager_combination {
            self.eager_theory_combination()
        } else {
            self.lazy_theory_combination()
        }
    }

    /// Eager theory combination: propagate all equalities immediately
    fn eager_theory_combination(&mut self) -> Result<SatResult, String> {
        let mut iteration = 0;

        loop {
            self.stats.theory_combination_rounds += 1;
            iteration += 1;

            if iteration > self.config.max_combination_rounds {
                return Ok(SatResult::Unknown);
            }

            // Collect implied equalities from all theories
            let mut new_equalities = Vec::new();

            for theory_id in self.theories.keys().copied().collect::<Vec<_>>() {
                let equalities = self.cached_theory_propagation(theory_id)?;

                for eq in equalities {
                    // Only propagate equalities between shared terms
                    if self.is_shared_term(eq.lhs) || self.is_shared_term(eq.rhs) {
                        new_equalities.push(eq);
                    }
                }
            }

            // No new equalities: fixed point reached
            if new_equalities.is_empty() {
                return Ok(SatResult::Sat);
            }

            // Propagate equalities to all theories
            for eq in new_equalities {
                self.propagate_equality(eq)?;
            }

            // Re-check theories for conflicts
            for solver in self.theories.values_mut() {
                match solver.check_sat()? {
                    SatResult::Unsat => {
                        self.stats.theory_conflicts += 1;
                        self.maybe_record_certificate_from_log();
                        return Ok(SatResult::Unsat);
                    }
                    SatResult::Unknown => {
                        return Ok(SatResult::Unknown);
                    }
                    SatResult::Sat => {}
                }
            }
        }
    }

    /// Lazy theory combination: propagate equalities on-demand
    fn lazy_theory_combination(&mut self) -> Result<SatResult, String> {
        // Process pending equalities
        while let Some(eq) = self.pending_equalities.pop_front() {
            self.propagate_equality(eq)?;

            // Check for conflicts after each propagation
            for solver in self.theories.values_mut() {
                match solver.check_sat()? {
                    SatResult::Unsat => {
                        self.stats.theory_conflicts += 1;
                        self.maybe_record_certificate_from_log();
                        return Ok(SatResult::Unsat);
                    }
                    SatResult::Unknown => {
                        return Ok(SatResult::Unknown);
                    }
                    SatResult::Sat => {}
                }
            }
        }

        Ok(SatResult::Sat)
    }

    /// Propagate an equality to all relevant theories
    fn propagate_equality(&mut self, eq: EqualityProp) -> Result<(), String> {
        self.stats.equalities_propagated += 1;
        let logged_eq = eq.clone();

        // Update equivalence classes
        self.merge_equivalence_classes(eq.lhs, eq.rhs)?;

        // Notify all theories that use these terms
        let theories_to_notify = self.get_theories_for_terms(eq.lhs, eq.rhs);

        for theory_id in theories_to_notify {
            if theory_id != eq.source
                && let Some(solver) = self.theories.get_mut(&theory_id)
            {
                solver.notify_equality(eq.lhs, eq.rhs)?;
            }
        }

        self.clear_from_level(self.current_level as u32);
        self.propagated_equalities_log.push(logged_eq);

        Ok(())
    }

    /// Identify shared terms in a formula
    fn identify_shared_terms(&mut self, _formula: TermId) -> Result<(), String> {
        // Placeholder: would traverse formula AST and identify terms used by multiple theories
        // For now, just update stats
        self.stats.shared_terms_count = self.shared_terms.len();
        Ok(())
    }

    /// Check if a term is shared between theories
    fn is_shared_term(&self, term: TermId) -> bool {
        self.shared_terms
            .get(&term)
            .is_some_and(|st| st.theories.len() > 1)
    }

    /// Get theories that use given terms
    fn get_theories_for_terms(&self, lhs: TermId, rhs: TermId) -> FxHashSet<TheoryId> {
        let mut theories = FxHashSet::default();

        if let Some(st) = self.shared_terms.get(&lhs) {
            theories.extend(&st.theories);
        }

        if let Some(st) = self.shared_terms.get(&rhs) {
            theories.extend(&st.theories);
        }

        theories
    }

    /// Merge equivalence classes for two terms
    fn merge_equivalence_classes(&mut self, lhs: TermId, rhs: TermId) -> Result<(), String> {
        // Get representatives
        let lhs_rep = self.find_representative(lhs);
        let rhs_rep = self.find_representative(rhs);

        if lhs_rep == rhs_rep {
            return Ok(());
        }

        // Union: make lhs_rep point to rhs_rep
        if let Some(st) = self.shared_terms.get_mut(&lhs_rep) {
            st.representative = rhs_rep;
        }

        Ok(())
    }

    /// Find equivalence class representative
    fn find_representative(&self, term: TermId) -> TermId {
        if let Some(st) = self.shared_terms.get(&term)
            && st.representative != term
        {
            // Path compression would be applied here
            return self.find_representative(st.representative);
        }
        term
    }

    /// Add a shared term
    pub fn add_shared_term(&mut self, term: TermId, theory: TheoryId) {
        self.shared_terms
            .entry(term)
            .or_insert_with(|| SharedTerm {
                term,
                theories: FxHashSet::default(),
                representative: term,
            })
            .theories
            .insert(theory);

        self.stats.shared_terms_count = self.shared_terms.len();
    }

    /// Enqueue an equality for propagation
    pub fn enqueue_equality(&mut self, lhs: TermId, rhs: TermId, source: TheoryId) {
        self.pending_equalities.push_back(EqualityProp {
            lhs,
            rhs,
            source,
            explanation: vec![],
        });
    }

    /// Backtrack all theories to a level
    pub fn backtrack(&mut self, level: usize) -> Result<(), String> {
        self.current_level = level;

        for solver in self.theories.values_mut() {
            solver.backtrack(level)?;
        }

        // Clear pending equalities
        self.pending_equalities.clear();
        self.clear_above_level(level as u32);
        self.propagated_equalities_log.clear();
        #[cfg(feature = "std")]
        {
            self.last_certificate = None;
        }

        Ok(())
    }

    /// Get combined model from all theories
    pub fn get_model(&self) -> Option<FxHashMap<TermId, TermId>> {
        let mut combined_model = FxHashMap::default();

        for solver in self.theories.values() {
            if let Some(model) = solver.get_model() {
                combined_model.extend(model);
            } else {
                return None;
            }
        }

        Some(combined_model)
    }

    /// Get combined conflict explanation
    pub fn get_conflict(&self) -> Option<Vec<TermId>> {
        // Collect conflicts from all theories
        let mut combined_conflict = Vec::new();

        for solver in self.theories.values() {
            if let Some(conflict) = solver.get_conflict() {
                combined_conflict.extend(conflict);
            }
        }

        if combined_conflict.is_empty() {
            None
        } else {
            // Minimize if enabled
            if self.config.minimize_conflicts {
                Some(self.minimize_conflict(combined_conflict))
            } else {
                Some(combined_conflict)
            }
        }
    }

    /// Minimize a conflict explanation
    fn minimize_conflict(&self, mut conflict: Vec<TermId>) -> Vec<TermId> {
        // Placeholder: would use resolution to minimize
        // For now, just remove duplicates
        conflict.sort();
        conflict.dedup();
        conflict
    }

    /// Get statistics
    pub fn stats(&self) -> &CoordinatorStats {
        &self.stats
    }

    /// Get current decision level
    pub fn current_level(&self) -> usize {
        self.current_level
    }

    /// Get the last generated theory-combination proof certificate.
    #[cfg(feature = "std")]
    pub fn proof_certificate(&self) -> Option<&NelsonOppenCertificate> {
        self.last_certificate.as_ref()
    }

    /// Increment decision level
    pub fn increment_level(&mut self) {
        self.current_level += 1;
    }

    fn maybe_record_certificate_from_log(&mut self) {
        #[cfg(feature = "std")]
        {
            if !self.config.proof_mode {
                return;
            }

            self.last_certificate = self.build_certificate_from_log();
        }
    }

    fn cached_theory_propagation(
        &mut self,
        theory_id: TheoryId,
    ) -> Result<Vec<EqualityProp>, String> {
        let level = self.current_level as u32;
        let key = (theory_id, level);

        if let Some(cached) = self.theory_propagation_cache.get(&key) {
            return Ok(cached.clone());
        }

        let solver = self
            .theories
            .get(&theory_id)
            .ok_or_else(|| format!("Theory {:?} not registered", theory_id))?;

        let propagated: Vec<EqualityProp> = solver
            .get_implied_equalities()
            .into_iter()
            .map(|(lhs, rhs)| EqualityProp {
                lhs,
                rhs,
                source: theory_id,
                explanation: vec![],
            })
            .collect();

        self.theory_propagation_cache
            .insert(key, propagated.clone());

        Ok(propagated)
    }

    fn clear_above_level(&mut self, level: u32) {
        self.theory_propagation_cache
            .retain(|(_, cached_level), _| *cached_level <= level);
    }

    fn clear_from_level(&mut self, level: u32) {
        self.theory_propagation_cache
            .retain(|(_, cached_level), _| *cached_level < level);
    }

    #[cfg(feature = "std")]
    fn build_certificate_from_log(&self) -> Option<NelsonOppenCertificate> {
        let last_eq = self.propagated_equalities_log.last()?;
        let mut certificate = NelsonOppenCertificate::new(
            self.to_proof_theory_id(last_eq.source),
            ProofNodeId(0),
        );

        for eq in &self.propagated_equalities_log {
            let lhs = Self::to_proof_term_id(eq.lhs)?;
            let rhs = Self::to_proof_term_id(eq.rhs)?;
            certificate.add_step(CombinationStep {
                theory: self.to_proof_theory_id(eq.source),
                propagated_equalities: vec![(lhs, rhs)],
                justification: Vec::new(),
            });
        }

        Some(certificate)
    }

    #[cfg(feature = "std")]
    fn to_proof_term_id(term: TermId) -> Option<ProofTermId> {
        let raw = u32::try_from(term).ok()?;
        Some(ProofTermId::new(raw))
    }

    #[cfg(feature = "std")]
    const fn to_proof_theory_id(&self, theory: TheoryId) -> CombinationTheoryId {
        let raw = match theory {
            TheoryId::Core => 0,
            TheoryId::Arithmetic => 1,
            TheoryId::BitVector => 2,
            TheoryId::Array => 3,
            TheoryId::Datatype => 4,
            TheoryId::String => 5,
            TheoryId::Uninterpreted => 6,
        };
        CombinationTheoryId(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock theory solver for testing
    struct MockTheory {
        id: TheoryId,
        sat_result: SatResult,
        implied_equalities: Vec<(TermId, TermId)>,
    }

    impl TheorySolver for MockTheory {
        fn theory_id(&self) -> TheoryId {
            self.id
        }

        fn assert_formula(&mut self, _formula: TermId) -> Result<(), String> {
            Ok(())
        }

        fn check_sat(&mut self) -> Result<SatResult, String> {
            Ok(self.sat_result)
        }

        fn get_model(&self) -> Option<FxHashMap<TermId, TermId>> {
            Some(FxHashMap::default())
        }

        fn get_conflict(&self) -> Option<Vec<TermId>> {
            None
        }

        fn backtrack(&mut self, _level: usize) -> Result<(), String> {
            Ok(())
        }

        fn get_implied_equalities(&self) -> Vec<(TermId, TermId)> {
            self.implied_equalities.clone()
        }

        fn notify_equality(&mut self, _lhs: TermId, _rhs: TermId) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn test_coordinator_creation() {
        let config = CoordinatorConfig::default();
        let coordinator = TheoryCoordinator::new(config);
        assert_eq!(coordinator.stats.check_sat_calls, 0);
    }

    #[test]
    fn test_register_theory() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TheoryCoordinator::new(config);

        let mock_theory = MockTheory {
            id: TheoryId::Arithmetic,
            sat_result: SatResult::Sat,
            implied_equalities: Vec::new(),
        };

        coordinator.register_theory(Box::new(mock_theory));
        assert!(coordinator.theories.contains_key(&TheoryId::Arithmetic));
    }

    #[test]
    fn test_check_sat_single_theory() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TheoryCoordinator::new(config);

        let mock_theory = MockTheory {
            id: TheoryId::Arithmetic,
            sat_result: SatResult::Sat,
            implied_equalities: Vec::new(),
        };

        coordinator.register_theory(Box::new(mock_theory));

        let result = coordinator.check_sat();
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test operation should succeed"),
            SatResult::Sat
        );
        assert_eq!(coordinator.stats.check_sat_calls, 1);
    }

    #[test]
    fn test_shared_term_management() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TheoryCoordinator::new(config);

        coordinator.add_shared_term(1, TheoryId::Arithmetic);
        coordinator.add_shared_term(1, TheoryId::BitVector);

        assert!(coordinator.is_shared_term(1));
        assert_eq!(coordinator.stats.shared_terms_count, 1);
    }

    #[test]
    fn test_equivalence_classes() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TheoryCoordinator::new(config);

        coordinator.add_shared_term(1, TheoryId::Arithmetic);
        coordinator.add_shared_term(2, TheoryId::Arithmetic);

        coordinator
            .merge_equivalence_classes(1, 2)
            .expect("test operation should succeed");

        let rep1 = coordinator.find_representative(1);
        let rep2 = coordinator.find_representative(2);
        assert_eq!(rep1, rep2);
    }

    #[test]
    fn test_equality_propagation() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TheoryCoordinator::new(config);

        coordinator.enqueue_equality(1, 2, TheoryId::Arithmetic);
        assert_eq!(coordinator.pending_equalities.len(), 1);
    }

    #[test]
    fn test_backtrack() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TheoryCoordinator::new(config);

        let mock_theory = MockTheory {
            id: TheoryId::Arithmetic,
            sat_result: SatResult::Sat,
            implied_equalities: Vec::new(),
        };

        coordinator.register_theory(Box::new(mock_theory));
        coordinator.increment_level();
        coordinator.increment_level();

        assert_eq!(coordinator.current_level(), 2);

        coordinator
            .backtrack(0)
            .expect("test operation should succeed");
        assert_eq!(coordinator.current_level(), 0);
    }

    #[test]
    fn test_get_model() {
        let config = CoordinatorConfig::default();
        let mut coordinator = TheoryCoordinator::new(config);

        let mock_theory = MockTheory {
            id: TheoryId::Arithmetic,
            sat_result: SatResult::Sat,
            implied_equalities: Vec::new(),
        };

        coordinator.register_theory(Box::new(mock_theory));

        let model = coordinator.get_model();
        assert!(model.is_some());
    }

    #[test]
    fn test_conflict_minimization() {
        let coordinator = TheoryCoordinator::new(CoordinatorConfig {
            minimize_conflicts: true,
            ..Default::default()
        });

        let conflict = vec![1, 2, 2, 3, 1, 4];
        let minimized = coordinator.minimize_conflict(conflict);

        assert_eq!(minimized, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_theory_propagation_cache_clears_on_backtrack() {
        let mut coordinator = TheoryCoordinator::new(CoordinatorConfig::default());
        coordinator.register_theory(Box::new(MockTheory {
            id: TheoryId::Arithmetic,
            sat_result: SatResult::Sat,
            implied_equalities: vec![(1, 2)],
        }));

        assert_eq!(
            coordinator
                .cached_theory_propagation(TheoryId::Arithmetic)
                .expect("initial cache fill should succeed")
                .len(),
            1
        );
        assert_eq!(coordinator.theory_propagation_cache.len(), 1);

        coordinator.increment_level();
        assert_eq!(
            coordinator
                .cached_theory_propagation(TheoryId::Arithmetic)
                .expect("level-one cache fill should succeed")
                .len(),
            1
        );
        assert_eq!(coordinator.theory_propagation_cache.len(), 2);

        coordinator
            .backtrack(0)
            .expect("backtrack should clear higher-level cache entries");
        assert_eq!(coordinator.theory_propagation_cache.len(), 1);
    }
}
