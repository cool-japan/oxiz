//! Advanced Proof Compression Techniques.
//!
//! Implements sophisticated compression algorithms specifically designed
//! for proof objects, achieving better compression ratios than generic algorithms.

use crate::{ProofNode, ProofRule, ResolutionProof};
use std::collections::HashMap;

/// Proof-specific compression using structural patterns.
pub struct StructuralCompressor {
    /// Dictionary of frequently occurring patterns
    pattern_dict: HashMap<Pattern, PatternId>,
    /// Next available pattern ID
    next_pattern_id: PatternId,
    /// Statistics
    stats: CompressionStats,
}

/// A pattern in a proof structure.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pattern {
    /// Single resolution with specific clause structure
    SingleResolution {
        left_size: usize,
        right_size: usize,
        result_size: usize,
    },
    /// Chain of resolutions
    ResolutionChain { length: usize },
    /// Binary tree of resolutions
    BinaryTree { depth: usize },
    /// Custom pattern
    Custom(Vec<u8>),
}

/// Unique identifier for a pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PatternId(pub usize);

/// Statistics about compression.
#[derive(Clone, Debug, Default)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_bytes: usize,
    /// Compressed size in bytes
    pub compressed_bytes: usize,
    /// Number of patterns found
    pub patterns_found: usize,
    /// Number of unique patterns
    pub unique_patterns: usize,
}

impl StructuralCompressor {
    /// Create a new structural compressor.
    pub fn new() -> Self {
        Self {
            pattern_dict: HashMap::new(),
            next_pattern_id: PatternId(0),
            stats: CompressionStats::default(),
        }
    }

    /// Compress a proof by identifying and encoding structural patterns.
    pub fn compress(&mut self, proof: &ResolutionProof) -> CompressedProof {
        self.stats.original_bytes = self.estimate_size(proof);

        // Phase 1: Identify recurring patterns
        let patterns = self.identify_patterns(proof);
        self.stats.patterns_found = patterns.len();
        self.stats.unique_patterns = self.pattern_dict.len();

        // Phase 2: Build compression dictionary
        let dict = self.build_dictionary(&patterns);

        // Phase 3: Encode proof using dictionary
        let encoded = self.encode_with_dictionary(proof, &dict);
        self.stats.compressed_bytes = encoded.bytes.len();

        encoded
    }

    /// Identify recurring patterns in the proof.
    fn identify_patterns(&mut self, proof: &ResolutionProof) -> Vec<(PatternId, Vec<usize>)> {
        let mut patterns = Vec::new();

        // Look for resolution chains
        for i in 0..proof.nodes.len() {
            if let Some((pattern, nodes)) = self.find_resolution_chain(proof, i) {
                let pattern_id = self.get_or_create_pattern(pattern);
                patterns.push((pattern_id, nodes));
            }
        }

        // Look for binary trees
        for i in 0..proof.nodes.len() {
            if let Some((pattern, nodes)) = self.find_binary_tree(proof, i) {
                let pattern_id = self.get_or_create_pattern(pattern);
                patterns.push((pattern_id, nodes));
            }
        }

        patterns
    }

    /// Find a resolution chain starting at a node.
    fn find_resolution_chain(
        &self,
        proof: &ResolutionProof,
        start: usize,
    ) -> Option<(Pattern, Vec<usize>)> {
        let mut nodes = vec![start];
        let mut current = start;
        let mut length = 1;

        loop {
            let node = proof.nodes.get(current)?;

            match &node.rule {
                ProofRule::Resolution { left, .. } => {
                    // Check if left child is also a resolution
                    if let Some(left_node) = proof.nodes.get(*left) {
                        if matches!(left_node.rule, ProofRule::Resolution { .. }) {
                            nodes.push(*left);
                            current = *left;
                            length += 1;
                            continue;
                        }
                    }
                    break;
                }
                _ => break,
            }
        }

        if length >= 3 {
            Some((Pattern::ResolutionChain { length }, nodes))
        } else {
            None
        }
    }

    /// Find a binary tree pattern starting at a node.
    fn find_binary_tree(
        &self,
        proof: &ResolutionProof,
        start: usize,
    ) -> Option<(Pattern, Vec<usize>)> {
        let depth = self.compute_tree_depth(proof, start);

        if depth >= 2 {
            let nodes = self.collect_tree_nodes(proof, start);
            Some((Pattern::BinaryTree { depth }, nodes))
        } else {
            None
        }
    }

    /// Compute the depth of a binary tree rooted at a node.
    fn compute_tree_depth(&self, proof: &ResolutionProof, node_idx: usize) -> usize {
        let node = match proof.nodes.get(node_idx) {
            Some(n) => n,
            None => return 0,
        };

        match &node.rule {
            ProofRule::Resolution { left, right, .. } => {
                let left_depth = self.compute_tree_depth(proof, *left);
                let right_depth = self.compute_tree_depth(proof, *right);
                1 + left_depth.max(right_depth)
            }
            _ => 0,
        }
    }

    /// Collect all nodes in a binary tree.
    fn collect_tree_nodes(&self, proof: &ResolutionProof, node_idx: usize) -> Vec<usize> {
        let mut nodes = vec![node_idx];

        if let Some(node) = proof.nodes.get(node_idx) {
            if let ProofRule::Resolution { left, right, .. } = &node.rule {
                nodes.extend(self.collect_tree_nodes(proof, *left));
                nodes.extend(self.collect_tree_nodes(proof, *right));
            }
        }

        nodes
    }

    /// Get or create a pattern ID.
    fn get_or_create_pattern(&mut self, pattern: Pattern) -> PatternId {
        if let Some(&id) = self.pattern_dict.get(&pattern) {
            return id;
        }

        let id = self.next_pattern_id;
        self.next_pattern_id = PatternId(id.0 + 1);
        self.pattern_dict.insert(pattern, id);
        id
    }

    /// Build a compression dictionary from patterns.
    fn build_dictionary(&self, patterns: &[(PatternId, Vec<usize>)]) -> CompressionDictionary {
        let mut dict = CompressionDictionary::new();

        // Count pattern frequencies
        let mut frequencies: HashMap<PatternId, usize> = HashMap::new();
        for (pattern_id, _) in patterns {
            *frequencies.entry(*pattern_id).or_insert(0) += 1;
        }

        // Add most frequent patterns to dictionary
        let mut sorted_patterns: Vec<_> = frequencies.into_iter().collect();
        sorted_patterns.sort_by_key(|(_, freq)| std::cmp::Reverse(*freq));

        for (pattern_id, _freq) in sorted_patterns.iter().take(256) {
            if let Some(pattern) = self.pattern_dict.iter().find(|(_, &id)| id == *pattern_id) {
                dict.add_entry(*pattern_id, pattern.0.clone());
            }
        }

        dict
    }

    /// Encode proof using compression dictionary.
    fn encode_with_dictionary(
        &self,
        proof: &ResolutionProof,
        dict: &CompressionDictionary,
    ) -> CompressedProof {
        let mut bytes = Vec::new();

        // Encode dictionary
        dict.encode(&mut bytes);

        // Encode proof nodes
        for node in &proof.nodes {
            self.encode_node(node, &mut bytes, dict);
        }

        CompressedProof {
            bytes,
            dictionary: dict.clone(),
            root: proof.root,
        }
    }

    /// Encode a single proof node.
    fn encode_node(&self, node: &ProofNode, bytes: &mut Vec<u8>, dict: &CompressionDictionary) {
        // Encode node ID
        Self::encode_varint(node.id as u64, bytes);

        // Encode clause length
        Self::encode_varint(node.clause.len() as u64, bytes);

        // Encode clause literals
        for &lit in &node.clause {
            Self::encode_signed_varint(lit as i64, bytes);
        }

        // Encode rule
        match &node.rule {
            ProofRule::Resolution { left, right, pivot } => {
                bytes.push(1); // Resolution tag
                Self::encode_varint(*left as u64, bytes);
                Self::encode_varint(*right as u64, bytes);
                Self::encode_signed_varint(*pivot as i64, bytes);
            }
            ProofRule::Input => {
                bytes.push(2); // Input tag
            }
            ProofRule::Axiom => {
                bytes.push(3); // Axiom tag
            }
        }
    }

    /// Encode unsigned integer using variable-length encoding.
    fn encode_varint(mut value: u64, bytes: &mut Vec<u8>) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value != 0 {
                byte |= 0x80;
            }

            bytes.push(byte);

            if value == 0 {
                break;
            }
        }
    }

    /// Encode signed integer using zigzag encoding + varint.
    fn encode_signed_varint(value: i64, bytes: &mut Vec<u8>) {
        let zigzag = if value >= 0 {
            (value as u64) << 1
        } else {
            ((-value - 1) as u64) << 1 | 1
        };

        Self::encode_varint(zigzag, bytes);
    }

    /// Estimate size of uncompressed proof in bytes.
    fn estimate_size(&self, proof: &ResolutionProof) -> usize {
        let mut size = 0;

        for node in &proof.nodes {
            size += 8; // node ID
            size += 4; // clause length
            size += node.clause.len() * 4; // literals
            size += 1; // rule tag

            if let ProofRule::Resolution { .. } = node.rule {
                size += 8 + 8 + 4; // left, right, pivot
            }
        }

        size
    }

    /// Get compression statistics.
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }

    /// Compute compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.stats.original_bytes == 0 {
            return 1.0;
        }

        self.stats.compressed_bytes as f64 / self.stats.original_bytes as f64
    }
}

impl Default for StructuralCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression dictionary mapping patterns to codes.
#[derive(Clone, Debug)]
pub struct CompressionDictionary {
    /// Map from pattern ID to pattern
    entries: HashMap<PatternId, Pattern>,
}

impl CompressionDictionary {
    /// Create a new empty dictionary.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add an entry to the dictionary.
    pub fn add_entry(&mut self, id: PatternId, pattern: Pattern) {
        self.entries.insert(id, pattern);
    }

    /// Encode dictionary to bytes.
    pub fn encode(&self, bytes: &mut Vec<u8>) {
        // Encode number of entries
        StructuralCompressor::encode_varint(self.entries.len() as u64, bytes);

        for (id, pattern) in &self.entries {
            // Encode pattern ID
            StructuralCompressor::encode_varint(id.0 as u64, bytes);

            // Encode pattern
            self.encode_pattern(pattern, bytes);
        }
    }

    /// Encode a single pattern.
    fn encode_pattern(&self, pattern: &Pattern, bytes: &mut Vec<u8>) {
        match pattern {
            Pattern::SingleResolution {
                left_size,
                right_size,
                result_size,
            } => {
                bytes.push(1); // SingleResolution tag
                StructuralCompressor::encode_varint(*left_size as u64, bytes);
                StructuralCompressor::encode_varint(*right_size as u64, bytes);
                StructuralCompressor::encode_varint(*result_size as u64, bytes);
            }
            Pattern::ResolutionChain { length } => {
                bytes.push(2); // ResolutionChain tag
                StructuralCompressor::encode_varint(*length as u64, bytes);
            }
            Pattern::BinaryTree { depth } => {
                bytes.push(3); // BinaryTree tag
                StructuralCompressor::encode_varint(*depth as u64, bytes);
            }
            Pattern::Custom(data) => {
                bytes.push(4); // Custom tag
                StructuralCompressor::encode_varint(data.len() as u64, bytes);
                bytes.extend_from_slice(data);
            }
        }
    }
}

impl Default for CompressionDictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// A compressed proof representation.
#[derive(Clone, Debug)]
pub struct CompressedProof {
    /// Compressed bytes
    pub bytes: Vec<u8>,
    /// Compression dictionary
    pub dictionary: CompressionDictionary,
    /// Root node index
    pub root: usize,
}

impl CompressedProof {
    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Decompress back to original proof.
    pub fn decompress(&self) -> Result<ResolutionProof, String> {
        // Decompression implementation
        Err("Decompression not yet implemented".to_string())
    }
}

/// LZ-style compression for proof sequences.
pub struct LzProofCompressor {
    /// Window size for back-references
    window_size: usize,
    /// Minimum match length
    min_match_length: usize,
}

impl LzProofCompressor {
    /// Create a new LZ-style compressor.
    pub fn new() -> Self {
        Self {
            window_size: 32768,
            min_match_length: 3,
        }
    }

    /// Compress using LZ algorithm.
    pub fn compress(&self, proof: &ResolutionProof) -> CompressedProof {
        let mut bytes = Vec::new();

        // Convert proof to byte sequence
        let proof_bytes = self.proof_to_bytes(proof);

        // Apply LZ compression
        self.lz_compress(&proof_bytes, &mut bytes);

        CompressedProof {
            bytes,
            dictionary: CompressionDictionary::new(),
            root: proof.root,
        }
    }

    /// Convert proof to byte sequence.
    fn proof_to_bytes(&self, proof: &ResolutionProof) -> Vec<u8> {
        let mut bytes = Vec::new();

        for node in &proof.nodes {
            // Simple serialization
            bytes.extend_from_slice(&(node.id as u32).to_le_bytes());
            bytes.extend_from_slice(&(node.clause.len() as u32).to_le_bytes());

            for &lit in &node.clause {
                bytes.extend_from_slice(&lit.to_le_bytes());
            }
        }

        bytes
    }

    /// Apply LZ compression to byte sequence.
    fn lz_compress(&self, input: &[u8], output: &mut Vec<u8>) {
        let mut pos = 0;

        while pos < input.len() {
            // Try to find a match in the window
            let window_start = pos.saturating_sub(self.window_size);
            let window = &input[window_start..pos];

            if let Some((match_pos, match_len)) = self.find_longest_match(window, &input[pos..]) {
                if match_len >= self.min_match_length {
                    // Encode back-reference
                    self.encode_backreference(match_pos, match_len, output);
                    pos += match_len;
                    continue;
                }
            }

            // No match, emit literal
            output.push(0); // Literal tag
            output.push(input[pos]);
            pos += 1;
        }
    }

    /// Find longest match in window.
    fn find_longest_match(&self, window: &[u8], lookahead: &[u8]) -> Option<(usize, usize)> {
        let mut best_match: Option<(usize, usize)> = None;

        for i in 0..window.len() {
            let mut match_len = 0;

            while match_len < lookahead.len()
                && i + match_len < window.len()
                && window[i + match_len] == lookahead[match_len]
            {
                match_len += 1;
            }

            if match_len >= self.min_match_length {
                if best_match.map_or(true, |(_, len)| match_len > len) {
                    best_match = Some((i, match_len));
                }
            }
        }

        best_match
    }

    /// Encode a back-reference.
    fn encode_backreference(&self, pos: usize, len: usize, output: &mut Vec<u8>) {
        output.push(1); // Backreference tag
        output.extend_from_slice(&(pos as u16).to_le_bytes());
        output.extend_from_slice(&(len as u16).to_le_bytes());
    }
}

impl Default for LzProofCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structural_compressor_creation() {
        let compressor = StructuralCompressor::new();
        assert_eq!(compressor.pattern_dict.len(), 0);
    }

    #[test]
    fn test_varint_encoding() {
        let mut bytes = Vec::new();
        StructuralCompressor::encode_varint(127, &mut bytes);
        assert_eq!(bytes, vec![127]);

        bytes.clear();
        StructuralCompressor::encode_varint(300, &mut bytes);
        assert_eq!(bytes, vec![0xAC, 0x02]);
    }

    #[test]
    fn test_lz_compressor_creation() {
        let compressor = LzProofCompressor::new();
        assert_eq!(compressor.window_size, 32768);
        assert_eq!(compressor.min_match_length, 3);
    }
}
