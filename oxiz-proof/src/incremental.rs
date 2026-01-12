//! Incremental proof construction.
//!
//! This module provides infrastructure for building proofs incrementally during
//! solving, with support for backtracking, scoping, and efficient recording.

use crate::proof::{Proof, ProofNodeId};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// A scoped proof builder for incremental construction
///
/// Supports:
/// - Push/pop scoping (for backtracking)
/// - Efficient deduplication
/// - Premise tracking
#[derive(Debug)]
pub struct IncrementalProofBuilder {
    /// The main proof being constructed
    proof: Proof,
    /// Stack of scope markers (number of nodes when scope was pushed)
    scopes: Vec<ScopeFrame>,
    /// Mapping from solver-specific clause IDs to proof nodes
    clause_map: FxHashMap<u32, ProofNodeId>,
    /// Whether proof recording is enabled
    enabled: bool,
}

/// A scope frame in the incremental builder
#[derive(Debug, Clone)]
struct ScopeFrame {
    /// Number of nodes in proof when scope was created
    /// (Reserved for future use in optimization)
    #[allow(dead_code)]
    proof_size: usize,
    /// Clause mappings added in this scope
    clause_mappings: Vec<(u32, ProofNodeId)>,
}

impl IncrementalProofBuilder {
    /// Create a new incremental proof builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            proof: Proof::new(),
            scopes: Vec::new(),
            clause_map: FxHashMap::default(),
            enabled: true,
        }
    }

    /// Enable proof recording
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable proof recording
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if proof recording is enabled
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Push a new scope (for backtracking)
    pub fn push_scope(&mut self) {
        self.scopes.push(ScopeFrame {
            proof_size: self.proof.len(),
            clause_mappings: Vec::new(),
        });
    }

    /// Pop a scope (backtrack)
    pub fn pop_scope(&mut self) {
        if let Some(frame) = self.scopes.pop() {
            // Remove clause mappings added in this scope
            for (clause_id, _) in &frame.clause_mappings {
                self.clause_map.remove(clause_id);
            }

            // Note: We don't remove proof nodes as they might be referenced
            // by nodes in outer scopes. Compression will clean them up later.
        }
    }

    /// Get the current scope level
    #[must_use]
    pub fn scope_level(&self) -> usize {
        self.scopes.len()
    }

    /// Record an axiom (input clause)
    pub fn record_axiom(&mut self, clause_id: u32, conclusion: impl Into<String>) -> ProofNodeId {
        if !self.enabled {
            return ProofNodeId(0); // Dummy ID
        }

        let node_id = self.proof.add_axiom(conclusion);
        self.clause_map.insert(clause_id, node_id);

        // Track in current scope if any
        if let Some(scope) = self.scopes.last_mut() {
            scope.clause_mappings.push((clause_id, node_id));
        }

        node_id
    }

    /// Record an inference step
    pub fn record_inference(
        &mut self,
        clause_id: u32,
        rule: impl Into<String>,
        premise_ids: &[u32],
        conclusion: impl Into<String>,
    ) -> ProofNodeId {
        if !self.enabled {
            return ProofNodeId(0); // Dummy ID
        }

        // Map solver clause IDs to proof node IDs
        let premises: Vec<ProofNodeId> = premise_ids
            .iter()
            .filter_map(|id| self.clause_map.get(id).copied())
            .collect();

        let node_id = self.proof.add_inference(rule, premises, conclusion);
        self.clause_map.insert(clause_id, node_id);

        // Track in current scope if any
        if let Some(scope) = self.scopes.last_mut() {
            scope.clause_mappings.push((clause_id, node_id));
        }

        node_id
    }

    /// Record an inference step with arguments
    pub fn record_inference_with_args(
        &mut self,
        clause_id: u32,
        rule: impl Into<String>,
        premise_ids: &[u32],
        args: Vec<String>,
        conclusion: impl Into<String>,
    ) -> ProofNodeId {
        if !self.enabled {
            return ProofNodeId(0); // Dummy ID
        }

        let premises: Vec<ProofNodeId> = premise_ids
            .iter()
            .filter_map(|id| self.clause_map.get(id).copied())
            .collect();

        let node_id = self
            .proof
            .add_inference_with_args(rule, premises, args, conclusion);
        self.clause_map.insert(clause_id, node_id);

        if let Some(scope) = self.scopes.last_mut() {
            scope.clause_mappings.push((clause_id, node_id));
        }

        node_id
    }

    /// Get the proof node for a solver clause ID
    #[must_use]
    pub fn get_clause_proof(&self, clause_id: u32) -> Option<ProofNodeId> {
        self.clause_map.get(&clause_id).copied()
    }

    /// Get the proof (immutable)
    #[must_use]
    pub const fn proof(&self) -> &Proof {
        &self.proof
    }

    /// Take the proof, leaving an empty one
    pub fn take_proof(&mut self) -> Proof {
        self.clause_map.clear();
        self.scopes.clear();
        std::mem::take(&mut self.proof)
    }

    /// Reset the builder
    pub fn reset(&mut self) {
        self.proof.clear();
        self.scopes.clear();
        self.clause_map.clear();
    }

    /// Get statistics about the incremental construction
    #[must_use]
    pub fn stats(&self) -> IncrementalStats {
        IncrementalStats {
            total_nodes: self.proof.len(),
            scope_level: self.scopes.len(),
            clause_mappings: self.clause_map.len(),
            enabled: self.enabled,
        }
    }
}

impl Default for IncrementalProofBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about incremental proof construction
#[derive(Debug, Clone, Copy)]
pub struct IncrementalStats {
    /// Total number of proof nodes
    pub total_nodes: usize,
    /// Current scope level
    pub scope_level: usize,
    /// Number of clause mappings
    pub clause_mappings: usize,
    /// Whether recording is enabled
    pub enabled: bool,
}

/// A proof recorder that can be used during SAT/SMT solving
///
/// This provides a simple interface for recording proof steps during search
#[derive(Debug)]
pub struct ProofRecorder {
    /// The incremental builder
    builder: IncrementalProofBuilder,
    /// Next clause ID
    next_clause_id: u32,
    /// Queue of recorded steps (for batching)
    queue: VecDeque<RecordedStep>,
    /// Whether to batch steps
    batch_mode: bool,
}

/// A recorded proof step
#[derive(Debug, Clone)]
enum RecordedStep {
    Axiom {
        clause_id: u32,
        conclusion: String,
    },
    Inference {
        clause_id: u32,
        rule: String,
        premises: Vec<u32>,
        conclusion: String,
        args: Vec<String>,
    },
}

impl ProofRecorder {
    /// Create a new proof recorder
    #[must_use]
    pub fn new() -> Self {
        Self {
            builder: IncrementalProofBuilder::new(),
            next_clause_id: 0,
            queue: VecDeque::new(),
            batch_mode: false,
        }
    }

    /// Enable batch mode (queue steps instead of adding immediately)
    pub fn enable_batch_mode(&mut self) {
        self.batch_mode = true;
    }

    /// Disable batch mode
    pub fn disable_batch_mode(&mut self) {
        self.batch_mode = false;
    }

    /// Allocate a new clause ID
    pub fn alloc_clause_id(&mut self) -> u32 {
        let id = self.next_clause_id;
        self.next_clause_id += 1;
        id
    }

    /// Record an input clause
    pub fn record_input(&mut self, conclusion: impl Into<String>) -> u32 {
        let clause_id = self.alloc_clause_id();
        let conclusion = conclusion.into();

        if self.batch_mode {
            self.queue.push_back(RecordedStep::Axiom {
                clause_id,
                conclusion,
            });
        } else {
            self.builder.record_axiom(clause_id, conclusion);
        }

        clause_id
    }

    /// Record a derived clause
    pub fn record_derived(
        &mut self,
        rule: impl Into<String>,
        premises: &[u32],
        conclusion: impl Into<String>,
    ) -> u32 {
        let clause_id = self.alloc_clause_id();
        let rule = rule.into();
        let conclusion = conclusion.into();

        if self.batch_mode {
            self.queue.push_back(RecordedStep::Inference {
                clause_id,
                rule,
                premises: premises.to_vec(),
                conclusion,
                args: Vec::new(),
            });
        } else {
            self.builder
                .record_inference(clause_id, rule, premises, conclusion);
        }

        clause_id
    }

    /// Flush the batched steps
    pub fn flush(&mut self) {
        while let Some(step) = self.queue.pop_front() {
            match step {
                RecordedStep::Axiom {
                    clause_id,
                    conclusion,
                } => {
                    self.builder.record_axiom(clause_id, conclusion);
                }
                RecordedStep::Inference {
                    clause_id,
                    rule,
                    premises,
                    conclusion,
                    args,
                } => {
                    self.builder
                        .record_inference_with_args(clause_id, rule, &premises, args, conclusion);
                }
            }
        }
    }

    /// Get the builder
    #[must_use]
    pub const fn builder(&self) -> &IncrementalProofBuilder {
        &self.builder
    }

    /// Get the mutable builder
    pub fn builder_mut(&mut self) -> &mut IncrementalProofBuilder {
        &mut self.builder
    }

    /// Take the proof
    pub fn take_proof(&mut self) -> Proof {
        self.flush();
        self.builder.take_proof()
    }
}

impl Default for ProofRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_builder_creation() {
        let builder = IncrementalProofBuilder::new();
        assert!(builder.is_enabled());
        assert_eq!(builder.scope_level(), 0);
    }

    #[test]
    fn test_record_axiom() {
        let mut builder = IncrementalProofBuilder::new();
        let clause_id = 1;
        builder.record_axiom(clause_id, "p");

        assert!(builder.get_clause_proof(clause_id).is_some());
        assert_eq!(builder.proof().len(), 1);
    }

    #[test]
    fn test_record_inference() {
        let mut builder = IncrementalProofBuilder::new();
        let c1 = 1;
        let c2 = 2;
        let c3 = 3;

        builder.record_axiom(c1, "p");
        builder.record_axiom(c2, "q");
        builder.record_inference(c3, "and", &[c1, c2], "p /\\ q");

        assert_eq!(builder.proof().len(), 3);
    }

    #[test]
    fn test_push_pop_scope() {
        let mut builder = IncrementalProofBuilder::new();

        builder.push_scope();
        assert_eq!(builder.scope_level(), 1);

        let c1 = 1;
        builder.record_axiom(c1, "p");

        builder.pop_scope();
        assert_eq!(builder.scope_level(), 0);
        assert!(builder.get_clause_proof(c1).is_none()); // Mapping removed
    }

    #[test]
    fn test_disabled_recording() {
        let mut builder = IncrementalProofBuilder::new();
        builder.disable();

        let c1 = 1;
        builder.record_axiom(c1, "p");

        // Proof should still be empty when disabled
        assert_eq!(builder.proof().len(), 0);
    }

    #[test]
    fn test_take_proof() {
        let mut builder = IncrementalProofBuilder::new();
        builder.record_axiom(1, "p");
        builder.record_axiom(2, "q");

        let proof = builder.take_proof();
        assert_eq!(proof.len(), 2);
        assert_eq!(builder.proof().len(), 0);
    }

    #[test]
    fn test_proof_recorder() {
        let mut recorder = ProofRecorder::new();

        let c1 = recorder.record_input("p");
        let c2 = recorder.record_input("q");
        let c3 = recorder.record_derived("and", &[c1, c2], "p /\\ q");

        assert!(c3 > c2);

        let proof = recorder.take_proof();
        assert_eq!(proof.len(), 3);
    }

    #[test]
    fn test_proof_recorder_batch() {
        let mut recorder = ProofRecorder::new();
        recorder.enable_batch_mode();

        let _c1 = recorder.record_input("p");
        let _c2 = recorder.record_input("q");

        // Nothing added yet (batched)
        assert_eq!(recorder.builder().proof().len(), 0);

        recorder.flush();

        // Now added
        assert_eq!(recorder.builder().proof().len(), 2);
    }

    #[test]
    fn test_incremental_stats() {
        let mut builder = IncrementalProofBuilder::new();
        builder.record_axiom(1, "p");
        builder.push_scope();
        builder.record_axiom(2, "q");

        let stats = builder.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.scope_level, 1);
        assert!(stats.enabled);
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn var_name() -> impl Strategy<Value = String> {
            "[a-z][0-9]*".prop_map(|s| s.to_string())
        }

        proptest! {
            /// Recording axioms should increase proof size (unless duplicated)
            #[test]
            fn prop_record_axiom_increases_size(
                conclusions in prop::collection::vec(var_name(), 1..10)
            ) {
                let mut builder = IncrementalProofBuilder::new();
                let mut seen = std::collections::HashSet::new();

                for (i, conclusion) in conclusions.iter().enumerate() {
                    let initial_len = builder.proof().len();
                    builder.record_axiom(i as u32 + 1, conclusion);

                    // Size increases only if this is a new unique conclusion
                    if seen.insert(conclusion.clone()) {
                        prop_assert!(builder.proof().len() > initial_len);
                    } else {
                        // Duplicate conclusions are deduplicated
                        prop_assert_eq!(builder.proof().len(), initial_len);
                    }
                }
            }

            /// Scope level should track push/pop correctly
            #[test]
            fn prop_scope_level_tracking(depth in 0..10_usize) {
                let mut builder = IncrementalProofBuilder::new();
                prop_assert_eq!(builder.scope_level(), 0);

                for i in 0..depth {
                    builder.push_scope();
                    prop_assert_eq!(builder.scope_level(), i + 1);
                }

                for i in (0..depth).rev() {
                    builder.pop_scope();
                    prop_assert_eq!(builder.scope_level(), i);
                }
            }

            /// Pop should remove clause mappings added in that scope
            #[test]
            fn prop_pop_removes_scope_mappings(
                base_count in 1..4_usize,
                scope_count in 1..4_usize
            ) {
                let mut builder = IncrementalProofBuilder::new();

                // Add some base clauses with unique IDs
                for i in 0..base_count {
                    builder.record_axiom(i as u32 + 1, format!("base{}", i));
                }
                let base_mappings = builder.stats().clause_mappings;

                // Push scope and add more with unique IDs
                builder.push_scope();
                for i in 0..scope_count {
                    builder.record_axiom((base_count + i) as u32 + 100, format!("scope{}", i));
                }

                // Pop should revert clause mappings to base count
                builder.pop_scope();
                prop_assert_eq!(builder.stats().clause_mappings, base_mappings);
            }

            /// Disabled builder should not add to proof
            #[test]
            fn prop_disabled_no_recording(
                conclusions in prop::collection::vec(var_name(), 1..10)
            ) {
                let mut builder = IncrementalProofBuilder::new();
                builder.disable();

                for (i, conclusion) in conclusions.iter().enumerate() {
                    builder.record_axiom(i as u32 + 1, conclusion);
                }

                prop_assert_eq!(builder.proof().len(), 0);
            }

            /// Stats should be consistent with builder state
            #[test]
            fn prop_stats_consistency(
                conclusions in prop::collection::vec(var_name(), 1..8),
                scope_depth in 0..5_usize
            ) {
                let mut builder = IncrementalProofBuilder::new();

                for _ in 0..scope_depth {
                    builder.push_scope();
                }

                for (i, conclusion) in conclusions.iter().enumerate() {
                    builder.record_axiom(i as u32 + 1, conclusion);
                }

                let stats = builder.stats();
                prop_assert_eq!(stats.scope_level, scope_depth);
                prop_assert!(stats.enabled);
                prop_assert_eq!(stats.total_nodes, builder.proof().len());
            }

            /// Take proof should empty the builder
            #[test]
            fn prop_take_proof_empties(
                conclusions in prop::collection::vec(var_name(), 1..8)
            ) {
                let mut builder = IncrementalProofBuilder::new();

                for (i, conclusion) in conclusions.iter().enumerate() {
                    builder.record_axiom(i as u32 + 1, conclusion);
                }

                let proof_len = builder.proof().len();
                let taken = builder.take_proof();

                prop_assert_eq!(taken.len(), proof_len);
                prop_assert_eq!(builder.proof().len(), 0);
            }

            /// ProofRecorder should maintain clause ID ordering for unique conclusions
            #[test]
            fn prop_recorder_id_ordering(count in 2..8_usize) {
                let mut recorder = ProofRecorder::new();
                let mut last_id = None;

                // Use unique conclusions to avoid deduplication
                for i in 0..count {
                    let id = recorder.record_input(format!("unique{}", i));
                    if let Some(prev_id) = last_id {
                        prop_assert!(id > prev_id);
                    }
                    last_id = Some(id);
                }
            }

            /// Batch mode should delay additions
            #[test]
            fn prop_batch_mode_delays(
                conclusions in prop::collection::vec(var_name(), 1..8)
            ) {
                let mut recorder = ProofRecorder::new();
                recorder.enable_batch_mode();

                for conclusion in &conclusions {
                    recorder.record_input(conclusion);
                }

                // Nothing added yet in batch mode
                prop_assert_eq!(recorder.builder().proof().len(), 0);

                recorder.flush();

                // Now everything is added (accounting for deduplication)
                let unique_count = conclusions.iter().collect::<std::collections::HashSet<_>>().len();
                prop_assert_eq!(recorder.builder().proof().len(), unique_count);
            }

            /// Recording derived clauses maintains dependencies
            #[test]
            fn prop_derived_dependencies(
                conclusions in prop::collection::vec(var_name(), 2..6)
            ) {
                let mut recorder = ProofRecorder::new();
                let mut clause_ids = Vec::new();

                // Record inputs
                for conclusion in &conclusions {
                    clause_ids.push(recorder.record_input(conclusion));
                }

                // Record derived clause from first two inputs
                if clause_ids.len() >= 2 {
                    let derived_id = recorder.record_derived(
                        "and",
                        &clause_ids[..2],
                        "derived"
                    );

                    // Derived ID should be larger than inputs
                    prop_assert!(derived_id > clause_ids[0]);
                    prop_assert!(derived_id > clause_ids[1]);
                }
            }
        }
    }
}
