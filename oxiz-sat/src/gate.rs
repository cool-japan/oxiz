//! Gate detection and optimization
//!
//! This module implements gate detection and extraction from CNF formulas.
//! Many real-world SAT instances contain encoded logical gates (AND, OR, XOR, ITE).
//! Detecting these gates enables specialized reasoning and optimizations.
//!
//! References:
//! - "Gate Extraction for CNF Formulas" (Eén et al.)
//! - "Effective Preprocessing with Hyper-Resolution and Equality Reduction" (Bacchus & Winter)
//! - "Recognition of Nested Gates in CNF Formulas" (Manthey)

use crate::clause::ClauseDatabase;
use crate::literal::Lit;
use std::collections::{HashMap, HashSet};

/// Statistics for gate detection
#[derive(Debug, Clone, Default)]
pub struct GateStats {
    /// Number of AND gates detected
    pub and_gates: usize,
    /// Number of OR gates detected
    pub or_gates: usize,
    /// Number of XOR gates detected
    pub xor_gates: usize,
    /// Number of ITE (if-then-else) gates detected
    pub ite_gates: usize,
    /// Number of MUX gates detected
    pub mux_gates: usize,
    /// Number of equivalent gates merged
    pub gates_merged: usize,
    /// Number of clauses removed via gate substitution
    pub clauses_removed: usize,
}

impl GateStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Gate Detection Statistics:");
        println!("  AND gates: {}", self.and_gates);
        println!("  OR gates: {}", self.or_gates);
        println!("  XOR gates: {}", self.xor_gates);
        println!("  ITE gates: {}", self.ite_gates);
        println!("  MUX gates: {}", self.mux_gates);
        println!("  Gates merged: {}", self.gates_merged);
        println!("  Clauses removed: {}", self.clauses_removed);
    }
}

/// Type of logical gate
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GateType {
    /// AND gate: output = a ∧ b
    And {
        /// Output literal of the AND gate
        output: Lit,
        /// Input literals
        inputs: Vec<Lit>,
    },
    /// OR gate: output = a ∨ b
    Or {
        /// Output literal of the OR gate
        output: Lit,
        /// Input literals
        inputs: Vec<Lit>,
    },
    /// XOR gate: output = a ⊕ b
    Xor {
        /// Output literal of the XOR gate
        output: Lit,
        /// Input literals
        inputs: Vec<Lit>,
    },
    /// ITE gate: output = if c then t else e
    Ite {
        /// Output literal of the ITE gate
        output: Lit,
        /// Condition literal
        condition: Lit,
        /// Then-value literal
        then_val: Lit,
        /// Else-value literal
        else_val: Lit,
    },
    /// MUX gate: generalized ITE with multiple conditions
    Mux {
        /// Output literal of the MUX gate
        output: Lit,
        /// Select literals
        select: Vec<Lit>,
        /// Input literals
        inputs: Vec<Lit>,
    },
}

impl GateType {
    /// Get the output literal of this gate
    #[must_use]
    pub fn output(&self) -> Lit {
        match self {
            GateType::And { output, .. }
            | GateType::Or { output, .. }
            | GateType::Xor { output, .. }
            | GateType::Ite { output, .. }
            | GateType::Mux { output, .. } => *output,
        }
    }

    /// Get all input literals of this gate
    #[must_use]
    pub fn inputs(&self) -> Vec<Lit> {
        match self {
            GateType::And { inputs, .. }
            | GateType::Or { inputs, .. }
            | GateType::Xor { inputs, .. } => inputs.clone(),
            GateType::Ite {
                condition,
                then_val,
                else_val,
                ..
            } => vec![*condition, *then_val, *else_val],
            GateType::Mux { select, inputs, .. } => {
                let mut all = select.clone();
                all.extend(inputs);
                all
            }
        }
    }
}

/// Gate detector and optimizer
#[derive(Debug)]
pub struct GateDetector {
    /// Detected gates indexed by output literal
    gates: HashMap<Lit, GateType>,
    /// Clauses that define each literal
    definitions: HashMap<Lit, Vec<Vec<Lit>>>,
    /// Statistics
    stats: GateStats,
}

impl Default for GateDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl GateDetector {
    /// Create a new gate detector
    #[must_use]
    pub fn new() -> Self {
        Self {
            gates: HashMap::new(),
            definitions: HashMap::new(),
            stats: GateStats::default(),
        }
    }

    /// Build clause definitions for each literal
    pub fn build_definitions(&mut self, clauses: &ClauseDatabase) {
        self.definitions.clear();

        for cid in clauses.iter_ids() {
            if let Some(clause) = clauses.get(cid) {
                let lits: Vec<_> = clause.lits.to_vec();

                // For each literal in the clause, record this clause as a definition
                for &lit in &lits {
                    self.definitions.entry(lit).or_default().push(lits.clone());
                }
            }
        }
    }

    /// Detect AND gates
    ///
    /// Pattern: output = a ∧ b ∧ ...
    /// CNF: (~a ∨ ~b ∨ output) ∧ (a ∨ ~output) ∧ (b ∨ ~output) ...
    pub fn detect_and_gates(&mut self) {
        let outputs: Vec<_> = self.definitions.keys().copied().collect();

        for output in outputs {
            if self.gates.contains_key(&output) {
                continue; // Already detected a gate for this literal
            }

            if let Some(gate_inputs) = self.try_extract_and_gate(output) {
                self.gates.insert(
                    output,
                    GateType::And {
                        output,
                        inputs: gate_inputs,
                    },
                );
                self.stats.and_gates += 1;
            }
        }
    }

    /// Try to extract an AND gate with the given output
    fn try_extract_and_gate(&self, output: Lit) -> Option<Vec<Lit>> {
        // For AND gate: out = a ∧ b
        // CNF: (~a ∨ ~b ∨ out) ∧ (a ∨ ~out) ∧ (b ∨ ~out)

        // Look for binary clauses (input ∨ ~output) in definitions[~output]
        let neg_output_defs = self.definitions.get(&!output)?;
        let mut inputs = Vec::new();

        for clause in neg_output_defs {
            if clause.len() == 2 && clause.contains(&!output) {
                // Binary clause (input ∨ ~output)
                let input = clause.iter().find(|&&lit| lit != !output)?;
                inputs.push(*input);
            }
        }

        if inputs.is_empty() {
            return None;
        }

        // Check for the main AND clause: (~a ∨ ~b ∨ ... ∨ output) in definitions[output]
        let output_defs = self.definitions.get(&output)?;

        for clause in output_defs {
            if clause.contains(&output) && clause.len() == inputs.len() + 1 {
                // Check if all other literals are negations of inputs
                let other_lits: Vec<_> = clause
                    .iter()
                    .filter(|&&lit| lit != output)
                    .copied()
                    .collect();

                let expected_lits: HashSet<_> = inputs.iter().map(|&lit| !lit).collect();
                let actual_lits: HashSet<_> = other_lits.iter().copied().collect();

                if expected_lits == actual_lits {
                    return Some(inputs);
                }
            }
        }

        None
    }

    /// Detect OR gates
    ///
    /// Pattern: output = a ∨ b ∨ ...
    /// CNF: (a ∨ b ∨ ~output) ∧ (~a ∨ output) ∧ (~b ∨ output) ...
    pub fn detect_or_gates(&mut self) {
        let outputs: Vec<_> = self.definitions.keys().copied().collect();

        for output in outputs {
            if self.gates.contains_key(&output) {
                continue;
            }

            if let Some(gate_inputs) = self.try_extract_or_gate(output) {
                self.gates.insert(
                    output,
                    GateType::Or {
                        output,
                        inputs: gate_inputs,
                    },
                );
                self.stats.or_gates += 1;
            }
        }
    }

    /// Try to extract an OR gate with the given output
    fn try_extract_or_gate(&self, output: Lit) -> Option<Vec<Lit>> {
        // For OR gate: out = a ∨ b
        // CNF: (a ∨ b ∨ ~out) ∧ (~a ∨ out) ∧ (~b ∨ out)

        // Look for binary clauses (~input ∨ output) in definitions[output]
        let output_defs = self.definitions.get(&output)?;
        let mut inputs = Vec::new();

        for clause in output_defs {
            if clause.len() == 2 && clause.contains(&output) {
                // Binary clause (~input ∨ output)
                let neg_input = clause.iter().find(|&&lit| lit != output)?;
                inputs.push(!*neg_input);
            }
        }

        if inputs.is_empty() {
            return None;
        }

        // Check for the main OR clause: (a ∨ b ∨ ... ∨ ~output) in definitions[~output]
        let neg_output_defs = self.definitions.get(&!output)?;

        for clause in neg_output_defs {
            if clause.contains(&!output) && clause.len() == inputs.len() + 1 {
                // Check if all other literals match inputs
                let other_lits: Vec<_> = clause
                    .iter()
                    .filter(|&&lit| lit != !output)
                    .copied()
                    .collect();

                let expected_lits: HashSet<_> = inputs.iter().copied().collect();
                let actual_lits: HashSet<_> = other_lits.iter().copied().collect();

                if expected_lits == actual_lits {
                    return Some(inputs);
                }
            }
        }

        None
    }

    /// Detect ITE (if-then-else) gates
    ///
    /// Pattern: output = if c then t else e
    /// CNF: (~c ∨ ~t ∨ output) ∧ (c ∨ ~e ∨ output) ∧ (~c ∨ t ∨ ~output) ∧ (c ∨ e ∨ ~output)
    pub fn detect_ite_gates(&mut self) {
        let outputs: Vec<_> = self.definitions.keys().copied().collect();

        for output in outputs {
            if self.gates.contains_key(&output) {
                continue;
            }

            if let Some((condition, then_val, else_val)) = self.try_extract_ite_gate(output) {
                self.gates.insert(
                    output,
                    GateType::Ite {
                        output,
                        condition,
                        then_val,
                        else_val,
                    },
                );
                self.stats.ite_gates += 1;
            }
        }
    }

    /// Try to extract an ITE gate with the given output
    fn try_extract_ite_gate(&self, output: Lit) -> Option<(Lit, Lit, Lit)> {
        let output_defs = self.definitions.get(&output)?;
        let neg_output_defs = self.definitions.get(&!output)?;

        // Look for the characteristic clauses of an ITE
        // This is a simplified detection - a full implementation would be more thorough
        for clause1 in output_defs {
            if clause1.len() == 3 {
                for clause2 in neg_output_defs {
                    if clause2.len() == 3 {
                        // Try to match ITE pattern
                        // This is a placeholder - full pattern matching would be complex
                        // For now, we'll just detect simple cases
                    }
                }
            }
        }

        None // Simplified - would need full pattern matching
    }

    /// Detect all gate types
    pub fn detect_all(&mut self, clauses: &ClauseDatabase) {
        self.build_definitions(clauses);
        self.detect_and_gates();
        self.detect_or_gates();
        self.detect_ite_gates();
    }

    /// Get all detected gates
    #[must_use]
    pub fn gates(&self) -> &HashMap<Lit, GateType> {
        &self.gates
    }

    /// Get gate for a specific literal
    #[must_use]
    pub fn get_gate(&self, lit: Lit) -> Option<&GateType> {
        self.gates.get(&lit)
    }

    /// Check if a literal is a gate output
    #[must_use]
    pub fn is_gate_output(&self, lit: Lit) -> bool {
        self.gates.contains_key(&lit)
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &GateStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = GateStats::default();
    }

    /// Clear all detected gates
    pub fn clear(&mut self) {
        self.gates.clear();
        self.definitions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_gate_detector_creation() {
        let detector = GateDetector::new();
        assert_eq!(detector.stats().and_gates, 0);
    }

    #[test]
    fn test_build_definitions() {
        let mut detector = GateDetector::new();
        let mut db = ClauseDatabase::new();

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));

        db.add_original(vec![a, b]);
        detector.build_definitions(&db);

        assert!(detector.definitions.contains_key(&a));
        assert!(detector.definitions.contains_key(&b));
    }

    #[test]
    fn test_detect_and_gate() {
        let mut detector = GateDetector::new();
        let mut db = ClauseDatabase::new();

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let out = Lit::pos(Var::new(2));

        // Encode: out = a ∧ b
        // CNF: (~a ∨ ~b ∨ out) ∧ (a ∨ ~out) ∧ (b ∨ ~out)
        db.add_original(vec![!a, !b, out]);
        db.add_original(vec![a, !out]);
        db.add_original(vec![b, !out]);

        detector.detect_all(&db);

        assert!(detector.is_gate_output(out));
        if let Some(GateType::And { inputs, .. }) = detector.get_gate(out) {
            assert_eq!(inputs.len(), 2);
            assert!(inputs.contains(&a));
            assert!(inputs.contains(&b));
        } else {
            panic!("Expected AND gate");
        }
    }

    #[test]
    fn test_detect_or_gate() {
        let mut detector = GateDetector::new();
        let mut db = ClauseDatabase::new();

        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let out = Lit::pos(Var::new(2));

        // Encode: out = a ∨ b
        // CNF: (a ∨ b ∨ ~out) ∧ (~a ∨ out) ∧ (~b ∨ out)
        db.add_original(vec![a, b, !out]);
        db.add_original(vec![!a, out]);
        db.add_original(vec![!b, out]);

        detector.detect_all(&db);

        assert!(detector.is_gate_output(out));
        if let Some(GateType::Or { inputs, .. }) = detector.get_gate(out) {
            assert_eq!(inputs.len(), 2);
            assert!(inputs.contains(&a));
            assert!(inputs.contains(&b));
        } else {
            panic!("Expected OR gate");
        }
    }

    #[test]
    fn test_gate_type_output() {
        let a = Lit::pos(Var::new(0));
        let b = Lit::pos(Var::new(1));
        let out = Lit::pos(Var::new(2));

        let gate = GateType::And {
            output: out,
            inputs: vec![a, b],
        };

        assert_eq!(gate.output(), out);
        let inputs = gate.inputs();
        assert_eq!(inputs.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut detector = GateDetector::new();
        let mut db = ClauseDatabase::new();

        db.add_original(vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        detector.build_definitions(&db);

        assert!(!detector.definitions.is_empty());

        detector.clear();
        assert!(detector.definitions.is_empty());
        assert!(detector.gates.is_empty());
    }
}
