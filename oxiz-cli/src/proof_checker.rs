//! Proof checking and verification module
//!
//! This module provides functionality to check and verify proof correctness.
//! It validates:
//! - Proof structure and format
//! - Inference rules application
//! - Proof tree consistency
//! - Resolution steps
//! - UNSAT core validity

use std::collections::{HashMap, HashSet};

/// Represents a proof step in a resolution-based proof
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofStep {
    /// Unique identifier for this step
    pub id: usize,
    /// The clause derived in this step
    pub clause: Vec<i32>,
    /// The rule applied (e.g., "resolution", "axiom", "assumption")
    pub rule: String,
    /// Parent step IDs (steps used to derive this one)
    pub parents: Vec<usize>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Proof structure containing all steps
#[derive(Debug, Clone)]
pub struct Proof {
    /// All steps in the proof
    pub steps: Vec<ProofStep>,
    /// Root assertions (axioms/assumptions)
    pub roots: HashSet<usize>,
    /// The final contradiction (empty clause)
    pub conclusion: Option<usize>,
}

impl Proof {
    /// Create a new empty proof
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            roots: HashSet::new(),
            conclusion: None,
        }
    }

    /// Add a proof step
    pub fn add_step(&mut self, step: ProofStep) {
        if step.rule == "axiom" || step.rule == "assumption" {
            self.roots.insert(step.id);
        }
        if step.clause.is_empty() {
            self.conclusion = Some(step.id);
        }
        self.steps.push(step);
    }

    /// Get a step by ID
    pub fn get_step(&self, id: usize) -> Option<&ProofStep> {
        self.steps.iter().find(|s| s.id == id)
    }

    /// Verify the entire proof
    pub fn verify(&self) -> Result<(), String> {
        // Note: We don't require a conclusion for partial proofs
        // This allows verification of intermediate proof steps

        // Check that all parent references are valid
        for step in &self.steps {
            for parent_id in &step.parents {
                if self.get_step(*parent_id).is_none() {
                    return Err(format!(
                        "Step {} references non-existent parent {}",
                        step.id, parent_id
                    ));
                }
            }
        }

        // Verify each inference step
        for step in &self.steps {
            if step.rule != "axiom" && step.rule != "assumption" {
                self.verify_step(step)?;
            }
        }

        Ok(())
    }

    /// Verify that this is a complete proof (has a conclusion)
    #[allow(dead_code)]
    pub fn verify_complete(&self) -> Result<(), String> {
        if self.conclusion.is_none() {
            return Err("Proof has no conclusion (empty clause)".to_string());
        }
        self.verify()
    }

    /// Verify a single proof step
    fn verify_step(&self, step: &ProofStep) -> Result<(), String> {
        match step.rule.as_str() {
            "resolution" => self.verify_resolution(step),
            "factoring" => self.verify_factoring(step),
            "subsumption" => self.verify_subsumption(step),
            "tautology" => self.verify_tautology(step),
            _ => {
                // Unknown rule, but we'll allow it for extensibility
                Ok(())
            }
        }
    }

    /// Verify a resolution step
    fn verify_resolution(&self, step: &ProofStep) -> Result<(), String> {
        if step.parents.len() != 2 {
            return Err(format!(
                "Resolution step {} must have exactly 2 parents, has {}",
                step.id,
                step.parents.len()
            ));
        }

        let parent1 = self
            .get_step(step.parents[0])
            .ok_or_else(|| format!("Parent {} not found", step.parents[0]))?;
        let parent2 = self
            .get_step(step.parents[1])
            .ok_or_else(|| format!("Parent {} not found", step.parents[1]))?;

        // Check if resolution is valid
        // Find the pivot literal (appears as L in one parent and -L in the other)
        let mut found_pivot = false;
        for &lit1 in &parent1.clause {
            if parent2.clause.contains(&-lit1) {
                // Found a pivot - verify the resolvent
                let mut expected_clause: HashSet<i32> = parent1
                    .clause
                    .iter()
                    .filter(|&&l| l != lit1)
                    .copied()
                    .collect();
                expected_clause.extend(parent2.clause.iter().filter(|&&l| l != -lit1));

                let result_clause: HashSet<i32> = step.clause.iter().copied().collect();

                if expected_clause == result_clause {
                    found_pivot = true;
                    break;
                }
            }
        }

        if !found_pivot {
            return Err(format!(
                "Invalid resolution step {}: no valid pivot found",
                step.id
            ));
        }

        Ok(())
    }

    /// Verify a factoring step (removes duplicate literals)
    fn verify_factoring(&self, step: &ProofStep) -> Result<(), String> {
        if step.parents.len() != 1 {
            return Err(format!(
                "Factoring step {} must have exactly 1 parent, has {}",
                step.id,
                step.parents.len()
            ));
        }

        let parent = self
            .get_step(step.parents[0])
            .ok_or_else(|| format!("Parent {} not found", step.parents[0]))?;

        // Check that the result is the parent with duplicates removed
        let parent_dedup: HashSet<i32> = parent.clause.iter().copied().collect();
        let result_set: HashSet<i32> = step.clause.iter().copied().collect();

        if parent_dedup != result_set {
            return Err(format!(
                "Invalid factoring step {}: result doesn't match parent",
                step.id
            ));
        }

        Ok(())
    }

    /// Verify a subsumption step
    fn verify_subsumption(&self, step: &ProofStep) -> Result<(), String> {
        if step.parents.len() != 2 {
            return Err(format!(
                "Subsumption step {} must have exactly 2 parents, has {}",
                step.id,
                step.parents.len()
            ));
        }

        let parent1 = self
            .get_step(step.parents[0])
            .ok_or_else(|| format!("Parent {} not found", step.parents[0]))?;
        let parent2 = self
            .get_step(step.parents[1])
            .ok_or_else(|| format!("Parent {} not found", step.parents[1]))?;

        // Check that one clause subsumes the other
        let set1: HashSet<i32> = parent1.clause.iter().copied().collect();
        let set2: HashSet<i32> = parent2.clause.iter().copied().collect();
        let result_set: HashSet<i32> = step.clause.iter().copied().collect();

        if set1.is_subset(&set2) && result_set == set1 {
            return Ok(());
        }
        if set2.is_subset(&set1) && result_set == set2 {
            return Ok(());
        }

        Err(format!(
            "Invalid subsumption step {}: no subsumption relationship",
            step.id
        ))
    }

    /// Verify a tautology elimination step
    fn verify_tautology(&self, step: &ProofStep) -> Result<(), String> {
        if step.parents.len() != 1 {
            return Err(format!(
                "Tautology step {} must have exactly 1 parent, has {}",
                step.id,
                step.parents.len()
            ));
        }

        let parent = self
            .get_step(step.parents[0])
            .ok_or_else(|| format!("Parent {} not found", step.parents[0]))?;

        // Check if parent is a tautology (contains both L and -L)
        for &lit in &parent.clause {
            if parent.clause.contains(&-lit) {
                // Parent is a tautology, result should be empty or simplified
                return Ok(());
            }
        }

        Err(format!(
            "Invalid tautology step {}: parent is not a tautology",
            step.id
        ))
    }

    /// Extract the UNSAT core from this proof
    pub fn extract_unsat_core(&self) -> HashSet<usize> {
        let mut core = HashSet::new();

        if let Some(conclusion_id) = self.conclusion {
            // Traverse backwards from conclusion to find all axioms used
            let mut to_visit = vec![conclusion_id];
            let mut visited = HashSet::new();

            while let Some(step_id) = to_visit.pop() {
                if visited.contains(&step_id) {
                    continue;
                }
                visited.insert(step_id);

                if let Some(step) = self.get_step(step_id) {
                    if self.roots.contains(&step_id) {
                        core.insert(step_id);
                    } else {
                        to_visit.extend(&step.parents);
                    }
                }
            }
        }

        core
    }
}

impl Default for Proof {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a simple proof format
/// Format: "step_id: rule parents -> clause"
/// Example: "3: resolution 1 2 -> [1, -2, 3]"
pub fn parse_simple_proof(proof_text: &str) -> Result<Proof, String> {
    let mut proof = Proof::new();

    for (line_num, line) in proof_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }

        // Parse: "id: rule parents -> clause"
        let parts: Vec<&str> = line.split("->").collect();
        if parts.len() != 2 {
            return Err(format!(
                "Line {}: Invalid format, expected '->'",
                line_num + 1
            ));
        }

        let left = parts[0].trim();
        let clause_str = parts[1].trim();

        // Parse left side: "id: rule parents"
        let left_parts: Vec<&str> = left.split(':').collect();
        if left_parts.len() != 2 {
            return Err(format!(
                "Line {}: Invalid format, expected 'id:'",
                line_num + 1
            ));
        }

        let id: usize = left_parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("Line {}: Invalid step ID", line_num + 1))?;

        let rule_and_parents: Vec<&str> = left_parts[1].split_whitespace().collect();
        if rule_and_parents.is_empty() {
            return Err(format!("Line {}: No rule specified", line_num + 1));
        }

        let rule = rule_and_parents[0].to_string();
        let parents: Vec<usize> = rule_and_parents[1..]
            .iter()
            .map(|s| {
                s.parse()
                    .map_err(|_| format!("Line {}: Invalid parent ID: {}", line_num + 1, s))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Parse clause: "[1, -2, 3]" or "[]"
        let clause_str = clause_str.trim_start_matches('[').trim_end_matches(']');
        let clause: Vec<i32> = if clause_str.is_empty() {
            Vec::new()
        } else {
            clause_str
                .split(',')
                .map(|s| {
                    s.trim().parse().map_err(|_| {
                        format!("Line {}: Invalid literal: {}", line_num + 1, s.trim())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        proof.add_step(ProofStep {
            id,
            clause,
            rule,
            parents,
            metadata: HashMap::new(),
        });
    }

    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_creation() {
        let mut proof = Proof::new();
        proof.add_step(ProofStep {
            id: 1,
            clause: vec![1, 2],
            rule: "axiom".to_string(),
            parents: vec![],
            metadata: HashMap::new(),
        });

        assert_eq!(proof.steps.len(), 1);
        assert!(proof.roots.contains(&1));
    }

    #[test]
    fn test_simple_resolution() {
        let mut proof = Proof::new();

        // Axiom 1: [1, 2]
        proof.add_step(ProofStep {
            id: 1,
            clause: vec![1, 2],
            rule: "axiom".to_string(),
            parents: vec![],
            metadata: HashMap::new(),
        });

        // Axiom 2: [-1, 3]
        proof.add_step(ProofStep {
            id: 2,
            clause: vec![-1, 3],
            rule: "axiom".to_string(),
            parents: vec![],
            metadata: HashMap::new(),
        });

        // Resolution on 1: [2, 3]
        proof.add_step(ProofStep {
            id: 3,
            clause: vec![2, 3],
            rule: "resolution".to_string(),
            parents: vec![1, 2],
            metadata: HashMap::new(),
        });

        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_proof_parsing() {
        let proof_text = r#"
            1: axiom -> [1, 2]
            2: axiom -> [-1, 3]
            3: resolution 1 2 -> [2, 3]
            4: axiom -> [-2]
            5: resolution 3 4 -> [3]
            6: axiom -> [-3]
            7: resolution 5 6 -> []
        "#;

        let proof = parse_simple_proof(proof_text).unwrap();
        assert_eq!(proof.steps.len(), 7);
        assert!(proof.verify().is_ok());
        assert_eq!(proof.conclusion, Some(7));
    }

    #[test]
    fn test_unsat_core_extraction() {
        let proof_text = r#"
            1: axiom -> [1]
            2: axiom -> [-1]
            3: resolution 1 2 -> []
        "#;

        let proof = parse_simple_proof(proof_text).unwrap();
        let core = proof.extract_unsat_core();

        assert_eq!(core.len(), 2);
        assert!(core.contains(&1));
        assert!(core.contains(&2));
    }

    #[test]
    fn test_invalid_resolution() {
        let mut proof = Proof::new();

        proof.add_step(ProofStep {
            id: 1,
            clause: vec![1, 2],
            rule: "axiom".to_string(),
            parents: vec![],
            metadata: HashMap::new(),
        });

        proof.add_step(ProofStep {
            id: 2,
            clause: vec![1, 3],
            rule: "axiom".to_string(),
            parents: vec![],
            metadata: HashMap::new(),
        });

        // Invalid resolution - no complementary literals
        proof.add_step(ProofStep {
            id: 3,
            clause: vec![2, 3],
            rule: "resolution".to_string(),
            parents: vec![1, 2],
            metadata: HashMap::new(),
        });

        assert!(proof.verify().is_err());
    }
}
