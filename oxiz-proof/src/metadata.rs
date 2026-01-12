//! Proof node metadata and annotations.
//!
//! This module provides a flexible metadata system for annotating proof nodes
//! with additional information useful for proof search, analysis, and debugging.

use rustc_hash::FxHashMap;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Priority level for proof nodes (for proof search guidance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Priority {
    /// Lowest priority
    VeryLow,
    /// Low priority
    Low,
    /// Normal priority (default)
    #[default]
    Normal,
    /// High priority
    High,
    /// Highest priority
    VeryHigh,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Priority::VeryLow => write!(f, "very-low"),
            Priority::Low => write!(f, "low"),
            Priority::Normal => write!(f, "normal"),
            Priority::High => write!(f, "high"),
            Priority::VeryHigh => write!(f, "very-high"),
        }
    }
}

/// Difficulty estimate for proof steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Difficulty {
    /// Trivial step (e.g., axiom, reflexivity)
    Trivial,
    /// Easy step (e.g., simple resolution)
    Easy,
    /// Medium difficulty
    #[default]
    Medium,
    /// Hard step (e.g., complex theory reasoning)
    Hard,
    /// Very hard step (e.g., quantifier instantiation)
    VeryHard,
}

impl fmt::Display for Difficulty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Difficulty::Trivial => write!(f, "trivial"),
            Difficulty::Easy => write!(f, "easy"),
            Difficulty::Medium => write!(f, "medium"),
            Difficulty::Hard => write!(f, "hard"),
            Difficulty::VeryHard => write!(f, "very-hard"),
        }
    }
}

/// Strategy hint for proof search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Strategy {
    /// Case splitting
    CaseSplit,
    /// Induction
    Induction,
    /// Contradiction
    Contradiction,
    /// Resolution-based
    Resolution,
    /// Theory reasoning
    Theory,
    /// Quantifier instantiation
    Quantifier,
    /// Rewriting
    Rewrite,
    /// Simplification
    Simplify,
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Strategy::CaseSplit => write!(f, "case-split"),
            Strategy::Induction => write!(f, "induction"),
            Strategy::Contradiction => write!(f, "contradiction"),
            Strategy::Resolution => write!(f, "resolution"),
            Strategy::Theory => write!(f, "theory"),
            Strategy::Quantifier => write!(f, "quantifier"),
            Strategy::Rewrite => write!(f, "rewrite"),
            Strategy::Simplify => write!(f, "simplify"),
        }
    }
}

/// Metadata attached to a proof node.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProofMetadata {
    /// Priority level
    priority: Priority,
    /// Difficulty estimate
    difficulty: Difficulty,
    /// Strategy hints
    strategies: Vec<Strategy>,
    /// User-defined tags
    tags: Vec<String>,
    /// Custom key-value attributes
    attributes: FxHashMap<String, String>,
    /// Optional description
    description: Option<String>,
}

impl ProofMetadata {
    /// Create new empty metadata.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set priority level.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set difficulty estimate.
    pub fn with_difficulty(mut self, difficulty: Difficulty) -> Self {
        self.difficulty = difficulty;
        self
    }

    /// Add a strategy hint.
    pub fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a custom attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Get priority level.
    #[must_use]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Get difficulty estimate.
    #[must_use]
    pub fn difficulty(&self) -> Difficulty {
        self.difficulty
    }

    /// Get strategy hints.
    #[must_use]
    pub fn strategies(&self) -> &[Strategy] {
        &self.strategies
    }

    /// Get tags.
    #[must_use]
    pub fn tags(&self) -> &[String] {
        &self.tags
    }

    /// Get description.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get custom attribute.
    #[must_use]
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(|s| s.as_str())
    }

    /// Get all custom attributes.
    #[must_use]
    pub fn attributes(&self) -> &FxHashMap<String, String> {
        &self.attributes
    }

    /// Check if has tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Check if has strategy.
    #[must_use]
    pub fn has_strategy(&self, strategy: Strategy) -> bool {
        self.strategies.contains(&strategy)
    }

    /// Set priority (mutable).
    pub fn set_priority(&mut self, priority: Priority) {
        self.priority = priority;
    }

    /// Set difficulty (mutable).
    pub fn set_difficulty(&mut self, difficulty: Difficulty) {
        self.difficulty = difficulty;
    }

    /// Add strategy (mutable).
    pub fn add_strategy(&mut self, strategy: Strategy) {
        if !self.strategies.contains(&strategy) {
            self.strategies.push(strategy);
        }
    }

    /// Add tag (mutable).
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Set attribute (mutable).
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Remove tag.
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.remove(pos);
            true
        } else {
            false
        }
    }

    /// Clear all metadata.
    pub fn clear(&mut self) {
        self.priority = Priority::default();
        self.difficulty = Difficulty::default();
        self.strategies.clear();
        self.tags.clear();
        self.attributes.clear();
        self.description = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let meta = ProofMetadata::new()
            .with_priority(Priority::High)
            .with_difficulty(Difficulty::Hard)
            .with_strategy(Strategy::Resolution)
            .with_tag("important")
            .with_description("Critical proof step");

        assert_eq!(meta.priority(), Priority::High);
        assert_eq!(meta.difficulty(), Difficulty::Hard);
        assert!(meta.has_strategy(Strategy::Resolution));
        assert!(meta.has_tag("important"));
        assert_eq!(meta.description(), Some("Critical proof step"));
    }

    #[test]
    fn test_metadata_attributes() {
        let meta = ProofMetadata::new()
            .with_attribute("author", "Alice")
            .with_attribute("timestamp", "2024-01-01");

        assert_eq!(meta.get_attribute("author"), Some("Alice"));
        assert_eq!(meta.get_attribute("timestamp"), Some("2024-01-01"));
        assert_eq!(meta.get_attribute("nonexistent"), None);
    }

    #[test]
    fn test_metadata_mutation() {
        let mut meta = ProofMetadata::new();
        meta.set_priority(Priority::VeryHigh);
        meta.set_difficulty(Difficulty::Trivial);
        meta.add_strategy(Strategy::Simplify);
        meta.add_tag("automated");

        assert_eq!(meta.priority(), Priority::VeryHigh);
        assert_eq!(meta.difficulty(), Difficulty::Trivial);
        assert!(meta.has_strategy(Strategy::Simplify));
        assert!(meta.has_tag("automated"));
    }

    #[test]
    fn test_metadata_removal() {
        let mut meta = ProofMetadata::new().with_tag("temp").with_tag("keep");

        assert!(meta.remove_tag("temp"));
        assert!(!meta.has_tag("temp"));
        assert!(meta.has_tag("keep"));
        assert!(!meta.remove_tag("nonexistent"));
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::VeryHigh > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        assert!(Priority::Low > Priority::VeryLow);
    }

    #[test]
    fn test_difficulty_ordering() {
        assert!(Difficulty::VeryHard > Difficulty::Hard);
        assert!(Difficulty::Hard > Difficulty::Medium);
        assert!(Difficulty::Medium > Difficulty::Easy);
        assert!(Difficulty::Easy > Difficulty::Trivial);
    }

    #[test]
    fn test_metadata_clear() {
        let mut meta = ProofMetadata::new()
            .with_priority(Priority::High)
            .with_tag("test");

        meta.clear();
        assert_eq!(meta.priority(), Priority::Normal);
        assert!(!meta.has_tag("test"));
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(Strategy::CaseSplit.to_string(), "case-split");
        assert_eq!(Strategy::Resolution.to_string(), "resolution");
        assert_eq!(Strategy::Quantifier.to_string(), "quantifier");
    }

    #[test]
    fn test_multiple_strategies() {
        let meta = ProofMetadata::new()
            .with_strategy(Strategy::Resolution)
            .with_strategy(Strategy::Theory);

        assert_eq!(meta.strategies().len(), 2);
        assert!(meta.has_strategy(Strategy::Resolution));
        assert!(meta.has_strategy(Strategy::Theory));
        assert!(!meta.has_strategy(Strategy::Induction));
    }

    #[test]
    fn test_duplicate_tag_prevention() {
        let mut meta = ProofMetadata::new();
        meta.add_tag("unique");
        meta.add_tag("unique");

        assert_eq!(meta.tags().len(), 1);
    }
}
