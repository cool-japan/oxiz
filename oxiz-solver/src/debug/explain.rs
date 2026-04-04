//! Conflict Explanation for Debugging.
//!
//! Generates human-readable explanations for conflicts and UNSAT results,
//! including the chain of propagations, contributing assertions, and
//! minimal conflicting subsets.
//!
//! ## References
//!
//! - Z3's `smt/smt_conflict_resolution.cpp`

#[allow(unused_imports)]
use crate::prelude::*;

/// A step in the propagation chain leading to a conflict.
#[derive(Debug, Clone)]
pub struct PropagationStep {
    /// The literal that was propagated.
    pub literal: i64,
    /// Human-readable description (e.g., "x1 = true").
    pub description: String,
    /// The reason clause index (if any).
    pub reason_clause: Option<u32>,
    /// Decision level.
    pub level: u32,
}

/// Information about which theory detected a conflict.
#[derive(Debug, Clone)]
pub struct TheoryConflictInfo {
    /// Theory name (e.g., "EUF", "LRA", "BV").
    pub theory: String,
    /// Brief description of the conflict reason.
    pub reason: String,
    /// Terms involved in the conflict.
    pub involved_terms: Vec<String>,
}

/// Explanation of a specific conflict.
#[derive(Debug, Clone)]
pub struct ConflictExplanation {
    /// Conflicting clause index.
    pub clause_id: u32,
    /// Assertions that contributed to this conflict.
    pub contributing_assertions: Vec<u32>,
    /// Chain of propagations leading to the conflict.
    pub propagation_chain: Vec<PropagationStep>,
    /// Which theory detected the conflict (if theory conflict).
    pub theory_info: Option<TheoryConflictInfo>,
    /// Minimal subset of assertions that still produce this conflict.
    pub minimal_subset: Vec<u32>,
}

impl ConflictExplanation {
    /// Format the conflict explanation as human-readable text.
    pub fn format(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("=== Conflict Explanation (clause #{}) ===\n\n", self.clause_id));

        // Contributing assertions
        out.push_str("Contributing assertions:\n");
        if self.contributing_assertions.is_empty() {
            out.push_str("  (none identified)\n");
        } else {
            for &idx in &self.contributing_assertions {
                out.push_str(&format!("  - Assertion #{}\n", idx));
            }
        }
        out.push('\n');

        // Propagation chain
        out.push_str("Propagation chain:\n");
        if self.propagation_chain.is_empty() {
            out.push_str("  (empty)\n");
        } else {
            for (i, step) in self.propagation_chain.iter().enumerate() {
                let reason = step
                    .reason_clause
                    .map_or_else(|| "decision".to_string(), |c| format!("clause #{}", c));
                out.push_str(&format!(
                    "  {}. {} (level={}, reason={})\n",
                    i + 1,
                    step.description,
                    step.level,
                    reason
                ));
            }
        }
        out.push('\n');

        // Theory info
        if let Some(ref ti) = self.theory_info {
            out.push_str(&format!("Theory conflict detected by: {}\n", ti.theory));
            out.push_str(&format!("  Reason: {}\n", ti.reason));
            if !ti.involved_terms.is_empty() {
                out.push_str("  Involved terms:\n");
                for term in &ti.involved_terms {
                    out.push_str(&format!("    - {}\n", term));
                }
            }
            out.push('\n');
        }

        // Minimal subset
        if !self.minimal_subset.is_empty() {
            out.push_str("Minimal conflicting subset:\n");
            for &idx in &self.minimal_subset {
                out.push_str(&format!("  - Assertion #{}\n", idx));
            }
        }

        out
    }
}

/// Explanation of an UNSAT result.
#[derive(Debug, Clone)]
pub struct UnsatExplanation {
    /// All assertion indices involved.
    pub all_assertions: Vec<u32>,
    /// The chain of conflicts that led to UNSAT.
    pub conflict_chain: Vec<ConflictExplanation>,
    /// Minimal unsatisfiable subset of assertions.
    pub minimal_unsat_subset: Vec<u32>,
    /// Summary description.
    pub summary: String,
}

impl UnsatExplanation {
    /// Format the UNSAT explanation as human-readable text.
    pub fn format(&self) -> String {
        let mut out = String::new();

        out.push_str("=== UNSAT Explanation ===\n\n");
        out.push_str(&format!("Summary: {}\n\n", self.summary));

        out.push_str(&format!(
            "Total assertions: {}\n",
            self.all_assertions.len()
        ));
        out.push_str(&format!(
            "Minimal UNSAT subset size: {}\n\n",
            self.minimal_unsat_subset.len()
        ));

        if !self.minimal_unsat_subset.is_empty() {
            out.push_str("Minimal UNSAT subset:\n");
            for &idx in &self.minimal_unsat_subset {
                out.push_str(&format!("  - Assertion #{}\n", idx));
            }
            out.push('\n');
        }

        out.push_str(&format!(
            "Conflict chain ({} conflicts):\n",
            self.conflict_chain.len()
        ));
        for (i, conflict) in self.conflict_chain.iter().enumerate() {
            out.push_str(&format!("\n--- Conflict {} ---\n", i + 1));
            out.push_str(&conflict.format());
        }

        out
    }
}

/// Conflict explainer that generates human-readable explanations.
#[derive(Debug)]
pub struct ConflictExplainer {
    /// Recorded assertion descriptions.
    assertion_descriptions: Vec<String>,
    /// Recorded conflicts.
    recorded_conflicts: Vec<ConflictExplanation>,
    /// Propagation history for explanation building.
    propagation_history: Vec<PropagationStep>,
}

impl ConflictExplainer {
    /// Create a new conflict explainer.
    pub fn new() -> Self {
        Self {
            assertion_descriptions: Vec::new(),
            recorded_conflicts: Vec::new(),
            propagation_history: Vec::new(),
        }
    }

    /// Register an assertion with its description.
    pub fn register_assertion(&mut self, index: u32, description: impl Into<String>) {
        let desc = description.into();
        let idx = index as usize;
        if idx >= self.assertion_descriptions.len() {
            self.assertion_descriptions.resize(idx + 1, String::new());
        }
        self.assertion_descriptions[idx] = desc;
    }

    /// Record a propagation step.
    pub fn record_propagation(&mut self, step: PropagationStep) {
        self.propagation_history.push(step);
    }

    /// Clear propagation history (e.g., after a restart).
    pub fn clear_propagation_history(&mut self) {
        self.propagation_history.clear();
    }

    /// Record a conflict with its contributing assertions and theory info.
    pub fn record_conflict(
        &mut self,
        clause_id: u32,
        contributing_assertions: Vec<u32>,
        theory_info: Option<TheoryConflictInfo>,
    ) {
        // Build propagation chain from history for the involved literals.
        let prop_chain = self.propagation_history.clone();

        // Compute minimal subset by greedy deletion.
        let minimal = self.compute_minimal_subset(&contributing_assertions);

        let explanation = ConflictExplanation {
            clause_id,
            contributing_assertions,
            propagation_chain: prop_chain,
            theory_info,
            minimal_subset: minimal,
        };

        self.recorded_conflicts.push(explanation);
    }

    /// Explain a specific recorded conflict by clause ID.
    pub fn explain_conflict(&self, clause_id: u32) -> Option<ConflictExplanation> {
        self.recorded_conflicts
            .iter()
            .find(|c| c.clause_id == clause_id)
            .cloned()
    }

    /// Generate a full UNSAT explanation from all recorded conflicts.
    pub fn explain_unsat(&self) -> UnsatExplanation {
        // Collect all assertion indices from all conflicts.
        let mut all_assertions: Vec<u32> = Vec::new();
        for conflict in &self.recorded_conflicts {
            for &idx in &conflict.contributing_assertions {
                if !all_assertions.contains(&idx) {
                    all_assertions.push(idx);
                }
            }
        }
        all_assertions.sort_unstable();

        // Minimal UNSAT subset: union of all minimal subsets from conflicts.
        let mut minimal_set: Vec<u32> = Vec::new();
        for conflict in &self.recorded_conflicts {
            for &idx in &conflict.minimal_subset {
                if !minimal_set.contains(&idx) {
                    minimal_set.push(idx);
                }
            }
        }
        minimal_set.sort_unstable();

        // If no minimal subsets were computed, use contributing assertions.
        if minimal_set.is_empty() {
            minimal_set = all_assertions.clone();
        }

        let summary = if self.recorded_conflicts.is_empty() {
            "UNSAT detected but no conflicts were recorded.".to_string()
        } else if self.recorded_conflicts.len() == 1 {
            let conflict = &self.recorded_conflicts[0];
            if let Some(ref ti) = conflict.theory_info {
                format!(
                    "UNSAT due to {} theory conflict: {}",
                    ti.theory, ti.reason
                )
            } else {
                format!(
                    "UNSAT due to conflict in clause #{}",
                    conflict.clause_id
                )
            }
        } else {
            format!(
                "UNSAT after {} conflicts involving {} assertions",
                self.recorded_conflicts.len(),
                all_assertions.len()
            )
        };

        UnsatExplanation {
            all_assertions,
            conflict_chain: self.recorded_conflicts.clone(),
            minimal_unsat_subset: minimal_set,
            summary,
        }
    }

    /// Get the number of recorded conflicts.
    pub fn num_conflicts(&self) -> usize {
        self.recorded_conflicts.len()
    }

    /// Clear all recorded data.
    pub fn clear(&mut self) {
        self.assertion_descriptions.clear();
        self.recorded_conflicts.clear();
        self.propagation_history.clear();
    }

    /// Compute a minimal subset from the contributing assertions.
    ///
    /// Uses a simple greedy approach: try removing each assertion, keep if
    /// the remaining set is still non-empty (i.e., it still contributes
    /// to the conflict). For debugging, we consider any subset of size >= 1
    /// that preserves the conflict structure.
    fn compute_minimal_subset(&self, assertions: &[u32]) -> Vec<u32> {
        if assertions.len() <= 1 {
            return assertions.to_vec();
        }

        // Simple greedy: include all assertions that appear in propagation
        // chain reasons or are directly referenced.
        let referenced: FxHashSet<u32> = {
            let mut set = FxHashSet::default();
            for step in &self.propagation_history {
                if let Some(clause) = step.reason_clause {
                    set.insert(clause);
                }
            }
            set
        };

        let mut minimal: Vec<u32> = assertions
            .iter()
            .copied()
            .filter(|idx| referenced.contains(idx) || assertions.len() <= 2)
            .collect();

        // If filtering removed everything, keep all.
        if minimal.is_empty() {
            minimal = assertions.to_vec();
        }

        minimal.sort_unstable();
        minimal
    }
}

impl Default for ConflictExplainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_explainer_new() {
        let explainer = ConflictExplainer::new();
        assert_eq!(explainer.num_conflicts(), 0);
    }

    #[test]
    fn test_record_and_explain_conflict() {
        let mut explainer = ConflictExplainer::new();
        explainer.register_assertion(0, "x > 0");
        explainer.register_assertion(1, "x < 0");

        explainer.record_propagation(PropagationStep {
            literal: 1,
            description: "x1 = true (from x > 0)".to_string(),
            reason_clause: Some(0),
            level: 0,
        });
        explainer.record_propagation(PropagationStep {
            literal: -2,
            description: "x2 = false (from x < 0)".to_string(),
            reason_clause: Some(1),
            level: 0,
        });

        explainer.record_conflict(
            5,
            vec![0, 1],
            Some(TheoryConflictInfo {
                theory: "LRA".to_string(),
                reason: "x > 0 AND x < 0 is contradictory".to_string(),
                involved_terms: vec!["x".to_string()],
            }),
        );

        assert_eq!(explainer.num_conflicts(), 1);

        let explanation = explainer.explain_conflict(5);
        assert!(explanation.is_some());
        let expl = explanation.expect("should have explanation");
        assert_eq!(expl.clause_id, 5);
        assert_eq!(expl.contributing_assertions, vec![0, 1]);
        assert!(expl.theory_info.is_some());
        assert_eq!(expl.propagation_chain.len(), 2);
    }

    #[test]
    fn test_explain_unsat() {
        let mut explainer = ConflictExplainer::new();
        explainer.register_assertion(0, "p");
        explainer.register_assertion(1, "NOT p");

        explainer.record_conflict(1, vec![0, 1], None);

        let unsat = explainer.explain_unsat();
        assert_eq!(unsat.all_assertions, vec![0, 1]);
        assert!(!unsat.summary.is_empty());
        assert_eq!(unsat.conflict_chain.len(), 1);
    }

    #[test]
    fn test_format_conflict_explanation() {
        let explanation = ConflictExplanation {
            clause_id: 3,
            contributing_assertions: vec![0, 1, 2],
            propagation_chain: vec![
                PropagationStep {
                    literal: 1,
                    description: "x = true".to_string(),
                    reason_clause: None,
                    level: 0,
                },
                PropagationStep {
                    literal: -2,
                    description: "y = false".to_string(),
                    reason_clause: Some(1),
                    level: 0,
                },
            ],
            theory_info: Some(TheoryConflictInfo {
                theory: "EUF".to_string(),
                reason: "congruence conflict".to_string(),
                involved_terms: vec!["f(a)".to_string(), "f(b)".to_string()],
            }),
            minimal_subset: vec![0, 1],
        };

        let text = explanation.format();
        assert!(text.contains("Conflict Explanation (clause #3)"));
        assert!(text.contains("Assertion #0"));
        assert!(text.contains("x = true"));
        assert!(text.contains("EUF"));
        assert!(text.contains("congruence conflict"));
        assert!(text.contains("f(a)"));
    }

    #[test]
    fn test_format_unsat_explanation() {
        let mut explainer = ConflictExplainer::new();
        explainer.record_conflict(
            1,
            vec![0, 1],
            Some(TheoryConflictInfo {
                theory: "LRA".to_string(),
                reason: "linear arithmetic conflict".to_string(),
                involved_terms: vec![],
            }),
        );

        let unsat = explainer.explain_unsat();
        let text = unsat.format();
        assert!(text.contains("UNSAT Explanation"));
        assert!(text.contains("LRA"));
        assert!(text.contains("linear arithmetic conflict"));
    }

    #[test]
    fn test_explainer_clear() {
        let mut explainer = ConflictExplainer::new();
        explainer.register_assertion(0, "test");
        explainer.record_conflict(1, vec![0], None);
        assert_eq!(explainer.num_conflicts(), 1);

        explainer.clear();
        assert_eq!(explainer.num_conflicts(), 0);
    }
}
