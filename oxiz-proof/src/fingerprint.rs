//! Proof fingerprinting for fast similarity detection.
//!
//! This module generates compact fingerprints of proofs that enable
//! fast similarity detection and proof reuse.

use crate::proof::{Proof, ProofStep};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A compact fingerprint of a proof.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProofFingerprint {
    /// Primary hash (structural)
    pub structural_hash: u64,
    /// Rule sequence hash
    pub rule_hash: u64,
    /// Size features
    pub size_features: SizeFeatures,
    /// Bloom filter for fast membership testing
    pub bloom_filter: Vec<u64>,
}

/// Size-based features of a proof.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SizeFeatures {
    /// Number of nodes
    pub num_nodes: usize,
    /// Maximum depth
    pub max_depth: usize,
    /// Number of axioms
    pub num_axioms: usize,
    /// Number of unique rules
    pub num_unique_rules: usize,
}

impl fmt::Display for ProofFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fingerprint[struct:{:016x}, rules:{:016x}, nodes:{}, depth:{}]",
            self.structural_hash,
            self.rule_hash,
            self.size_features.num_nodes,
            self.size_features.max_depth
        )
    }
}

/// Fingerprint generator for proofs.
pub struct FingerprintGenerator {
    /// Bloom filter size (number of bits)
    bloom_size: usize,
    /// Number of hash functions for bloom filter
    num_hash_functions: usize,
}

impl Default for FingerprintGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprintGenerator {
    /// Create a new fingerprint generator with default settings.
    pub fn new() -> Self {
        Self {
            bloom_size: 256,
            num_hash_functions: 3,
        }
    }

    /// Set the bloom filter size.
    pub fn with_bloom_size(mut self, size: usize) -> Self {
        self.bloom_size = size;
        self
    }

    /// Set the number of hash functions.
    pub fn with_hash_functions(mut self, num: usize) -> Self {
        self.num_hash_functions = num;
        self
    }

    /// Generate a fingerprint for a proof.
    pub fn generate(&self, proof: &Proof) -> ProofFingerprint {
        let structural_hash = self.compute_structural_hash(proof);
        let rule_hash = self.compute_rule_hash(proof);
        let size_features = self.extract_size_features(proof);
        let bloom_filter = self.create_bloom_filter(proof);

        ProofFingerprint {
            structural_hash,
            rule_hash,
            size_features,
            bloom_filter,
        }
    }

    /// Check if two fingerprints are likely similar.
    pub fn are_similar(&self, fp1: &ProofFingerprint, fp2: &ProofFingerprint) -> bool {
        // Check exact structural match
        if fp1.structural_hash == fp2.structural_hash {
            return true;
        }

        // Check size similarity
        if !self.size_features_similar(&fp1.size_features, &fp2.size_features) {
            return false;
        }

        // Check bloom filter overlap
        self.bloom_overlap(&fp1.bloom_filter, &fp2.bloom_filter) > 0.7
    }

    /// Compute similarity score between two fingerprints (0.0 - 1.0).
    pub fn similarity_score(&self, fp1: &ProofFingerprint, fp2: &ProofFingerprint) -> f64 {
        if fp1 == fp2 {
            return 1.0;
        }

        let mut score = 0.0;

        // Structural similarity (40% weight)
        if fp1.structural_hash == fp2.structural_hash {
            score += 0.4;
        }

        // Rule similarity (30% weight)
        if fp1.rule_hash == fp2.rule_hash {
            score += 0.3;
        }

        // Size similarity (10% weight)
        let size_sim = self.compute_size_similarity(&fp1.size_features, &fp2.size_features);
        score += 0.1 * size_sim;

        // Bloom filter overlap (20% weight)
        let bloom_sim = self.bloom_overlap(&fp1.bloom_filter, &fp2.bloom_filter);
        score += 0.2 * bloom_sim;

        score
    }

    // Helper: Compute structural hash of a proof
    fn compute_structural_hash(&self, proof: &Proof) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash the proof structure
        for node in proof.nodes() {
            match &node.step {
                ProofStep::Axiom { conclusion } => {
                    "axiom".hash(&mut hasher);
                    conclusion.hash(&mut hasher);
                }
                ProofStep::Inference { rule, premises, .. } => {
                    "inference".hash(&mut hasher);
                    rule.hash(&mut hasher);
                    premises.len().hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }

    // Helper: Compute hash of rule sequence
    fn compute_rule_hash(&self, proof: &Proof) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut rules = Vec::new();

        for node in proof.nodes() {
            if let ProofStep::Inference { rule, .. } = &node.step {
                rules.push(rule.clone());
            }
        }

        rules.sort();
        rules.dedup();
        rules.hash(&mut hasher);

        hasher.finish()
    }

    // Helper: Extract size features
    fn extract_size_features(&self, proof: &Proof) -> SizeFeatures {
        let num_nodes = proof.len();
        let max_depth = proof.nodes().iter().map(|n| n.depth).max().unwrap_or(0);
        let num_axioms = proof
            .nodes()
            .iter()
            .filter(|n| matches!(n.step, ProofStep::Axiom { .. }))
            .count();

        let mut unique_rules = FxHashSet::default();
        for node in proof.nodes() {
            if let ProofStep::Inference { rule, .. } = &node.step {
                unique_rules.insert(rule.clone());
            }
        }

        SizeFeatures {
            num_nodes,
            max_depth: max_depth as usize,
            num_axioms,
            num_unique_rules: unique_rules.len(),
        }
    }

    // Helper: Create bloom filter for proof elements
    fn create_bloom_filter(&self, proof: &Proof) -> Vec<u64> {
        let num_words = self.bloom_size.div_ceil(64);
        let mut bloom = vec![0u64; num_words];

        for node in proof.nodes() {
            // Add node conclusion to bloom filter
            self.bloom_add(&mut bloom, node.conclusion());

            // Add rule to bloom filter
            if let ProofStep::Inference { rule, .. } = &node.step {
                self.bloom_add(&mut bloom, rule);
            }
        }

        bloom
    }

    // Helper: Add element to bloom filter
    fn bloom_add(&self, bloom: &mut [u64], element: &str) {
        for i in 0..self.num_hash_functions {
            let hash = self.hash_with_seed(element, i as u64);
            let bit_index = (hash as usize) % (bloom.len() * 64);
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            bloom[word_index] |= 1u64 << bit_offset;
        }
    }

    // Helper: Hash with seed
    fn hash_with_seed(&self, element: &str, seed: u64) -> u64 {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        element.hash(&mut hasher);
        hasher.finish()
    }

    // Helper: Check if size features are similar
    fn size_features_similar(&self, sf1: &SizeFeatures, sf2: &SizeFeatures) -> bool {
        // Allow 20% difference in size
        let size_ratio = (sf1.num_nodes as f64) / (sf2.num_nodes.max(1) as f64);
        if !(0.8..=1.2).contains(&size_ratio) {
            return false;
        }

        // Allow 30% difference in depth
        let depth_ratio = (sf1.max_depth as f64) / (sf2.max_depth.max(1) as f64);
        if !(0.7..=1.3).contains(&depth_ratio) {
            return false;
        }

        true
    }

    // Helper: Compute size similarity score
    fn compute_size_similarity(&self, sf1: &SizeFeatures, sf2: &SizeFeatures) -> f64 {
        let size_sim = 1.0
            - ((sf1.num_nodes as f64 - sf2.num_nodes as f64).abs()
                / sf1.num_nodes.max(sf2.num_nodes).max(1) as f64);
        let depth_sim = 1.0
            - ((sf1.max_depth as f64 - sf2.max_depth as f64).abs()
                / sf1.max_depth.max(sf2.max_depth).max(1) as f64);
        let axiom_sim = 1.0
            - ((sf1.num_axioms as f64 - sf2.num_axioms as f64).abs()
                / sf1.num_axioms.max(sf2.num_axioms).max(1) as f64);

        (size_sim + depth_sim + axiom_sim) / 3.0
    }

    // Helper: Compute bloom filter overlap
    fn bloom_overlap(&self, bloom1: &[u64], bloom2: &[u64]) -> f64 {
        if bloom1.len() != bloom2.len() {
            return 0.0;
        }

        let mut intersection_bits = 0;
        let mut union_bits = 0;

        for (&b1, &b2) in bloom1.iter().zip(bloom2.iter()) {
            intersection_bits += (b1 & b2).count_ones();
            union_bits += (b1 | b2).count_ones();
        }

        if union_bits == 0 {
            return 1.0;
        }

        (intersection_bits as f64) / (union_bits as f64)
    }
}

/// Database for storing and querying proof fingerprints.
pub struct FingerprintDatabase {
    /// Generator for creating fingerprints
    generator: FingerprintGenerator,
    /// Stored fingerprints with associated IDs
    fingerprints: FxHashMap<String, ProofFingerprint>,
}

impl Default for FingerprintDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprintDatabase {
    /// Create a new fingerprint database.
    pub fn new() -> Self {
        Self {
            generator: FingerprintGenerator::new(),
            fingerprints: FxHashMap::default(),
        }
    }

    /// Create with a custom generator.
    pub fn with_generator(generator: FingerprintGenerator) -> Self {
        Self {
            generator,
            fingerprints: FxHashMap::default(),
        }
    }

    /// Add a proof to the database.
    pub fn add_proof(&mut self, id: String, proof: &Proof) {
        let fingerprint = self.generator.generate(proof);
        self.fingerprints.insert(id, fingerprint);
    }

    /// Find similar proofs in the database.
    pub fn find_similar(&self, proof: &Proof, threshold: f64) -> Vec<(String, f64)> {
        let query_fp = self.generator.generate(proof);
        let mut results = Vec::new();

        for (id, fp) in &self.fingerprints {
            let score = self.generator.similarity_score(&query_fp, fp);
            if score >= threshold {
                results.push((id.clone(), score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get the number of stored fingerprints.
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Check if the database is empty.
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Clear the database.
    pub fn clear(&mut self) {
        self.fingerprints.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_generator_new() {
        let generator = FingerprintGenerator::new();
        assert_eq!(generator.bloom_size, 256);
        assert_eq!(generator.num_hash_functions, 3);
    }

    #[test]
    fn test_fingerprint_generator_with_settings() {
        let generator = FingerprintGenerator::new()
            .with_bloom_size(512)
            .with_hash_functions(5);
        assert_eq!(generator.bloom_size, 512);
        assert_eq!(generator.num_hash_functions, 5);
    }

    #[test]
    fn test_generate_fingerprint_empty_proof() {
        let generator = FingerprintGenerator::new();
        let proof = Proof::new();
        let fp = generator.generate(&proof);
        assert_eq!(fp.size_features.num_nodes, 0);
        assert_eq!(fp.size_features.max_depth, 0);
        assert_eq!(fp.size_features.num_axioms, 0);
    }

    #[test]
    fn test_fingerprint_display() {
        let fp = ProofFingerprint {
            structural_hash: 0x1234567890abcdef,
            rule_hash: 0xfedcba0987654321,
            size_features: SizeFeatures {
                num_nodes: 10,
                max_depth: 5,
                num_axioms: 2,
                num_unique_rules: 3,
            },
            bloom_filter: vec![0, 0, 0, 0],
        };
        let display = fp.to_string();
        assert!(display.contains("1234567890abcdef"));
        assert!(display.contains("nodes:10"));
        assert!(display.contains("depth:5"));
    }

    #[test]
    fn test_identical_proofs_same_fingerprint() {
        let generator = FingerprintGenerator::new();
        let mut proof = Proof::new();
        proof.add_axiom("x = x");

        let fp1 = generator.generate(&proof);
        let fp2 = generator.generate(&proof);

        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_similarity_score_identical() {
        let generator = FingerprintGenerator::new();
        let proof = Proof::new();
        let fp = generator.generate(&proof);

        let score = generator.similarity_score(&fp, &fp);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_size_features_similar() {
        let generator = FingerprintGenerator::new();
        let sf1 = SizeFeatures {
            num_nodes: 100,
            max_depth: 10,
            num_axioms: 5,
            num_unique_rules: 3,
        };
        let sf2 = SizeFeatures {
            num_nodes: 105,
            max_depth: 11,
            num_axioms: 5,
            num_unique_rules: 3,
        };
        assert!(generator.size_features_similar(&sf1, &sf2));
    }

    #[test]
    fn test_size_features_dissimilar() {
        let generator = FingerprintGenerator::new();
        let sf1 = SizeFeatures {
            num_nodes: 100,
            max_depth: 10,
            num_axioms: 5,
            num_unique_rules: 3,
        };
        let sf2 = SizeFeatures {
            num_nodes: 200,
            max_depth: 10,
            num_axioms: 5,
            num_unique_rules: 3,
        };
        assert!(!generator.size_features_similar(&sf1, &sf2));
    }

    #[test]
    fn test_fingerprint_database_new() {
        let db = FingerprintDatabase::new();
        assert_eq!(db.len(), 0);
        assert!(db.is_empty());
    }

    #[test]
    fn test_fingerprint_database_add() {
        let mut db = FingerprintDatabase::new();
        let proof = Proof::new();
        db.add_proof("proof1".to_string(), &proof);
        assert_eq!(db.len(), 1);
        assert!(!db.is_empty());
    }

    #[test]
    fn test_fingerprint_database_find_similar() {
        let mut db = FingerprintDatabase::new();
        let proof1 = Proof::new();
        let proof2 = Proof::new();

        db.add_proof("proof1".to_string(), &proof1);

        let similar = db.find_similar(&proof2, 0.9);
        assert_eq!(similar.len(), 1);
        assert_eq!(similar[0].0, "proof1");
        assert_eq!(similar[0].1, 1.0);
    }

    #[test]
    fn test_fingerprint_database_clear() {
        let mut db = FingerprintDatabase::new();
        let proof = Proof::new();
        db.add_proof("proof1".to_string(), &proof);
        db.clear();
        assert!(db.is_empty());
    }
}
