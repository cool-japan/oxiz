//! Interpolant extraction from proofs.
//!
//! An interpolant for a pair of formulas (A, B) where A ∧ B is unsatisfiable
//! is a formula I such that:
//! - A implies I
//! - I ∧ B is unsatisfiable
//! - I only contains symbols common to both A and B
//!
//! Interpolants are useful for model checking, invariant generation,
//! and compositional verification.

use crate::premise::{PremiseId, PremiseTracker};
use crate::proof::{Proof, ProofNodeId, ProofStep};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// A partition of premises into A-side and B-side.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Premises in the A partition.
    a_premises: FxHashSet<PremiseId>,
    /// Premises in the B partition.
    b_premises: FxHashSet<PremiseId>,
}

impl Partition {
    /// Create a new partition.
    #[must_use]
    pub fn new(
        a_premises: impl IntoIterator<Item = PremiseId>,
        b_premises: impl IntoIterator<Item = PremiseId>,
    ) -> Self {
        Self {
            a_premises: a_premises.into_iter().collect(),
            b_premises: b_premises.into_iter().collect(),
        }
    }

    /// Check if a premise is in the A partition.
    #[must_use]
    pub fn is_a_premise(&self, premise: PremiseId) -> bool {
        self.a_premises.contains(&premise)
    }

    /// Check if a premise is in the B partition.
    #[must_use]
    pub fn is_b_premise(&self, premise: PremiseId) -> bool {
        self.b_premises.contains(&premise)
    }

    /// Get all A premises.
    #[must_use]
    pub fn a_premises(&self) -> &FxHashSet<PremiseId> {
        &self.a_premises
    }

    /// Get all B premises.
    #[must_use]
    pub fn b_premises(&self) -> &FxHashSet<PremiseId> {
        &self.b_premises
    }
}

/// Color of a proof node in the interpolation procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    /// Node depends only on A premises.
    A,
    /// Node depends only on B premises.
    B,
    /// Node depends on both A and B premises (mixed).
    AB,
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::AB => write!(f, "AB"),
        }
    }
}

/// An interpolant formula.
#[derive(Debug, Clone)]
pub struct Interpolant {
    /// The interpolant formula (as a string).
    pub formula: String,
    /// Symbols used in the interpolant.
    pub symbols: FxHashSet<String>,
}

impl Interpolant {
    /// Create a new interpolant.
    #[must_use]
    pub fn new(formula: impl Into<String>) -> Self {
        let formula = formula.into();
        let symbols = extract_symbols(&formula);
        Self { formula, symbols }
    }

    /// Check if the interpolant only uses common symbols.
    #[must_use]
    pub fn is_valid(&self, a_symbols: &FxHashSet<String>, b_symbols: &FxHashSet<String>) -> bool {
        let common: FxHashSet<String> = a_symbols.intersection(b_symbols).cloned().collect();
        self.symbols.is_subset(&common)
    }
}

impl fmt::Display for Interpolant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.formula)
    }
}

/// Interpolant extractor.
///
/// Extracts interpolants from resolution proofs using the
/// symmetric interpolation system.
#[derive(Debug)]
pub struct InterpolantExtractor {
    /// Partition of premises.
    #[allow(dead_code)]
    partition: Partition,
    /// Premise tracker for looking up premises.
    #[allow(dead_code)]
    premise_tracker: PremiseTracker,
    /// Node colors (computed during extraction).
    colors: FxHashMap<ProofNodeId, Color>,
    /// Partial interpolants for each node.
    partial_interpolants: FxHashMap<ProofNodeId, String>,
}

impl InterpolantExtractor {
    /// Create a new interpolant extractor.
    #[must_use]
    pub fn new(partition: Partition, premise_tracker: PremiseTracker) -> Self {
        Self {
            partition,
            premise_tracker,
            colors: FxHashMap::default(),
            partial_interpolants: FxHashMap::default(),
        }
    }

    /// Extract an interpolant from a proof.
    ///
    /// The proof should derive false (empty clause) from A ∪ B.
    pub fn extract(&mut self, proof: &Proof) -> Result<Interpolant, String> {
        let root = proof
            .root()
            .ok_or_else(|| "Proof has no root".to_string())?;

        // Compute colors for all nodes
        self.compute_colors(proof, root)?;

        // Build partial interpolants bottom-up
        self.build_interpolants(proof, root)?;

        // The root should be colored AB (depends on both A and B)
        let root_color = self
            .colors
            .get(&root)
            .copied()
            .ok_or_else(|| "Root node has no color".to_string())?;

        if root_color != Color::AB {
            return Err(format!(
                "Root should have color AB, but has color {}",
                root_color
            ));
        }

        // The interpolant is the partial interpolant at the root
        let interpolant_formula = self
            .partial_interpolants
            .get(&root)
            .ok_or_else(|| "No interpolant at root".to_string())?
            .clone();

        Ok(Interpolant::new(interpolant_formula))
    }

    /// Compute the color of each proof node.
    fn compute_colors(&mut self, proof: &Proof, node_id: ProofNodeId) -> Result<Color, String> {
        // Check if already computed
        if let Some(&color) = self.colors.get(&node_id) {
            return Ok(color);
        }

        let node = proof
            .get_node(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        let color = match &node.step {
            ProofStep::Axiom { .. } => {
                // Axioms are colored based on which partition they belong to
                // For now, assume all axioms are in A (would need premise tracking)
                Color::A
            }
            ProofStep::Inference { premises, .. } => {
                // Inference nodes are colored based on their premises
                let mut has_a = false;
                let mut has_b = false;

                for &premise_id in premises {
                    let premise_color = self.compute_colors(proof, premise_id)?;
                    match premise_color {
                        Color::A => has_a = true,
                        Color::B => has_b = true,
                        Color::AB => {
                            has_a = true;
                            has_b = true;
                        }
                    }
                }

                if has_a && has_b {
                    Color::AB
                } else if has_a {
                    Color::A
                } else if has_b {
                    Color::B
                } else {
                    // No premises - shouldn't happen in a well-formed proof
                    Color::A
                }
            }
        };

        self.colors.insert(node_id, color);
        Ok(color)
    }

    /// Build partial interpolants for each node.
    fn build_interpolants(&mut self, proof: &Proof, node_id: ProofNodeId) -> Result<(), String> {
        // Check if already computed
        if self.partial_interpolants.contains_key(&node_id) {
            return Ok(());
        }

        let node = proof
            .get_node(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        let color = *self
            .colors
            .get(&node_id)
            .ok_or_else(|| format!("Node {} has no color", node_id))?;

        let interpolant = match &node.step {
            ProofStep::Axiom { conclusion } => {
                // Axiom interpolants depend on color
                match color {
                    Color::A => "true".to_string(),
                    Color::B => conclusion.clone(),
                    Color::AB => {
                        // Shouldn't happen for axioms
                        "true".to_string()
                    }
                }
            }
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                ..
            } => {
                // First, compute interpolants for all premises
                for &premise_id in premises {
                    self.build_interpolants(proof, premise_id)?;
                }

                // Combine premise interpolants based on the rule and colors
                self.combine_interpolants(rule, premises, conclusion, color)?
            }
        };

        self.partial_interpolants.insert(node_id, interpolant);
        Ok(())
    }

    /// Combine interpolants from premises.
    fn combine_interpolants(
        &self,
        rule: &str,
        premises: &[ProofNodeId],
        _conclusion: &str,
        color: Color,
    ) -> Result<String, String> {
        match color {
            Color::A => {
                // A-colored nodes: interpolant is true
                Ok("true".to_string())
            }
            Color::B => {
                // B-colored nodes: interpolant is the conclusion
                // For now, use a placeholder
                Ok("(b-node)".to_string())
            }
            Color::AB => {
                // AB-colored nodes: combine premise interpolants
                if rule == "resolution" && premises.len() == 2 {
                    let i1 = self
                        .partial_interpolants
                        .get(&premises[0])
                        .ok_or_else(|| format!("No interpolant for premise {}", premises[0]))?;
                    let i2 = self
                        .partial_interpolants
                        .get(&premises[1])
                        .ok_or_else(|| format!("No interpolant for premise {}", premises[1]))?;

                    let c1 = self.colors.get(&premises[0]).copied().unwrap_or(Color::A);
                    let c2 = self.colors.get(&premises[1]).copied().unwrap_or(Color::A);

                    // Symmetric interpolation system
                    match (c1, c2) {
                        (Color::A, Color::B) | (Color::B, Color::A) => {
                            // One A premise, one B premise: disjoin interpolants
                            Ok(format!("(or {} {})", i1, i2))
                        }
                        (Color::A, Color::AB) | (Color::AB, Color::A) => {
                            // One A, one AB: use AB's interpolant
                            if c1 == Color::AB {
                                Ok(i1.clone())
                            } else {
                                Ok(i2.clone())
                            }
                        }
                        (Color::B, Color::AB) | (Color::AB, Color::B) => {
                            // One B, one AB: use AB's interpolant
                            if c1 == Color::AB {
                                Ok(i1.clone())
                            } else {
                                Ok(i2.clone())
                            }
                        }
                        (Color::AB, Color::AB) => {
                            // Both AB: disjoin
                            Ok(format!("(or {} {})", i1, i2))
                        }
                        _ => {
                            // Shouldn't happen for AB-colored inference
                            Ok("(error)".to_string())
                        }
                    }
                } else {
                    // Other rules: placeholder
                    Ok("(combined)".to_string())
                }
            }
        }
    }
}

/// Extract symbols from a formula string.
///
/// This is a simplified version - real implementation would parse the formula.
fn extract_symbols(formula: &str) -> FxHashSet<String> {
    let mut symbols = FxHashSet::default();

    // Logical operators and keywords to exclude
    let keywords: FxHashSet<&str> = [
        "and", "or", "not", "implies", "iff", "xor", "forall", "exists", "true", "false", "let",
        "ite", "distinct",
    ]
    .iter()
    .copied()
    .collect();

    // Simple heuristic: extract words (alphanumeric sequences)
    let mut current = String::new();
    for ch in formula.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            if !current.is_empty()
                && !current.chars().all(|c| c.is_numeric())
                && !keywords.contains(current.as_str())
            {
                symbols.insert(current.clone());
            }
            current.clear();
        }
    }

    if !current.is_empty()
        && !current.chars().all(|c| c.is_numeric())
        && !keywords.contains(current.as_str())
    {
        symbols.insert(current);
    }

    symbols
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_creation() {
        let partition = Partition::new(vec![PremiseId(0), PremiseId(1)], vec![PremiseId(2)]);

        assert!(partition.is_a_premise(PremiseId(0)));
        assert!(partition.is_a_premise(PremiseId(1)));
        assert!(partition.is_b_premise(PremiseId(2)));
        assert!(!partition.is_b_premise(PremiseId(0)));
    }

    #[test]
    fn test_color_display() {
        assert_eq!(format!("{}", Color::A), "A");
        assert_eq!(format!("{}", Color::B), "B");
        assert_eq!(format!("{}", Color::AB), "AB");
    }

    #[test]
    fn test_interpolant_creation() {
        let interp = Interpolant::new("(and p q)");
        assert_eq!(interp.formula, "(and p q)");
        // "and" is a keyword and should not be included
        assert!(!interp.symbols.contains("and"));
        assert!(interp.symbols.contains("p"));
        assert!(interp.symbols.contains("q"));
    }

    #[test]
    fn test_interpolant_validity() {
        let interp = Interpolant::new("(and x y)");

        let mut a_symbols = FxHashSet::default();
        a_symbols.insert("x".to_string());
        a_symbols.insert("y".to_string());
        a_symbols.insert("z".to_string());

        let mut b_symbols = FxHashSet::default();
        b_symbols.insert("x".to_string());
        b_symbols.insert("y".to_string());
        b_symbols.insert("w".to_string());

        // x and y are common, so the interpolant is valid
        assert!(interp.is_valid(&a_symbols, &b_symbols));
    }

    #[test]
    fn test_interpolant_invalid() {
        let interp = Interpolant::new("(and x z)");

        let mut a_symbols = FxHashSet::default();
        a_symbols.insert("x".to_string());
        a_symbols.insert("z".to_string());

        let mut b_symbols = FxHashSet::default();
        b_symbols.insert("x".to_string());
        b_symbols.insert("w".to_string());

        // z is not common, so the interpolant is invalid
        assert!(!interp.is_valid(&a_symbols, &b_symbols));
    }

    #[test]
    fn test_extract_symbols() {
        let symbols = extract_symbols("(and x (or y z))");
        // Keywords like "and" and "or" should be excluded
        assert!(!symbols.contains("and"));
        assert!(symbols.contains("x"));
        assert!(!symbols.contains("or"));
        assert!(symbols.contains("y"));
        assert!(symbols.contains("z"));
    }

    #[test]
    fn test_extract_symbols_numbers() {
        let symbols = extract_symbols("(= x 42)");
        assert!(symbols.contains("x"));
        // Numbers should not be included as symbols
        assert!(!symbols.contains("42"));
    }

    #[test]
    fn test_extractor_creation() {
        let partition = Partition::new(vec![PremiseId(0)], vec![PremiseId(1)]);
        let tracker = PremiseTracker::new();
        let extractor = InterpolantExtractor::new(partition, tracker);

        assert_eq!(extractor.colors.len(), 0);
        assert_eq!(extractor.partial_interpolants.len(), 0);
    }

    #[test]
    fn test_simple_interpolant_extraction() {
        let partition = Partition::new(vec![PremiseId(0)], vec![PremiseId(1)]);
        let tracker = PremiseTracker::new();
        let mut extractor = InterpolantExtractor::new(partition, tracker);

        // Simple proof: axiom
        let mut proof = Proof::new();
        proof.add_axiom("p");

        // This will fail because we haven't properly set up premise tracking,
        // but it tests the basic structure
        let result = extractor.extract(&proof);
        assert!(result.is_err());
    }
}
