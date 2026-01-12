//! Proof streaming for processing large proofs efficiently.
//!
//! This module provides streaming APIs to process large proofs without
//! loading the entire proof into memory at once.

use crate::proof::{Proof, ProofNode, ProofNodeId, ProofStep};
use rustc_hash::FxHashMap;
use std::io::{self, BufWriter, Write};

/// Configuration for proof streaming.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Chunk size for streaming (number of nodes)
    pub chunk_size: usize,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
    /// Enable compression during streaming
    pub compress: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            buffer_size: 8192,
            compress: false,
        }
    }
}

impl StreamConfig {
    /// Create a new stream configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the buffer size.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Enable compression.
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compress = enabled;
        self
    }
}

/// Chunk of proof nodes for streaming.
#[derive(Debug, Clone)]
pub struct ProofChunk {
    /// Starting node ID in this chunk
    pub start_id: ProofNodeId,
    /// Nodes in this chunk
    pub nodes: Vec<ProofNode>,
    /// Chunk index
    pub index: usize,
    /// Total number of chunks
    pub total_chunks: usize,
}

impl ProofChunk {
    /// Create a new proof chunk.
    pub fn new(
        start_id: ProofNodeId,
        nodes: Vec<ProofNode>,
        index: usize,
        total_chunks: usize,
    ) -> Self {
        Self {
            start_id,
            nodes,
            index,
            total_chunks,
        }
    }

    /// Get the number of nodes in this chunk.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Check if this is the last chunk.
    pub fn is_last(&self) -> bool {
        self.index + 1 == self.total_chunks
    }
}

/// Streaming proof reader.
pub struct ProofStreamer {
    config: StreamConfig,
}

impl Default for ProofStreamer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofStreamer {
    /// Create a new proof streamer.
    pub fn new() -> Self {
        Self {
            config: StreamConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: StreamConfig) -> Self {
        Self { config }
    }

    /// Stream proof in chunks.
    pub fn stream_chunks<'a>(&self, proof: &'a Proof) -> ProofChunkIterator<'a> {
        ProofChunkIterator::new(proof, self.config.chunk_size)
    }

    /// Process proof in streaming fashion with a callback.
    pub fn process_streaming<F>(&self, proof: &Proof, mut callback: F) -> Result<(), String>
    where
        F: FnMut(&ProofChunk) -> Result<(), String>,
    {
        for chunk in self.stream_chunks(proof) {
            callback(&chunk)?;
        }
        Ok(())
    }

    /// Write proof to a writer in streaming fashion.
    pub fn write_streaming<W: Write>(&self, proof: &Proof, writer: W) -> io::Result<()> {
        let mut buf_writer = BufWriter::with_capacity(self.config.buffer_size, writer);

        for chunk in self.stream_chunks(proof) {
            for node in &chunk.nodes {
                writeln!(buf_writer, "{}", self.format_node(node))?;
            }
        }

        buf_writer.flush()
    }

    // Helper: Format a node for output
    fn format_node(&self, node: &ProofNode) -> String {
        match &node.step {
            ProofStep::Axiom { conclusion } => {
                format!("axiom {} : {}", node.id, conclusion)
            }
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                ..
            } => {
                let premise_str = premises
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "infer {} : {} from [{}] => {}",
                    node.id, rule, premise_str, conclusion
                )
            }
        }
    }
}

/// Iterator over proof chunks.
pub struct ProofChunkIterator<'a> {
    proof: &'a Proof,
    chunk_size: usize,
    current_index: usize,
    total_nodes: usize,
}

impl<'a> ProofChunkIterator<'a> {
    /// Create a new chunk iterator.
    pub fn new(proof: &'a Proof, chunk_size: usize) -> Self {
        Self {
            proof,
            chunk_size,
            current_index: 0,
            total_nodes: proof.len(),
        }
    }
}

impl<'a> Iterator for ProofChunkIterator<'a> {
    type Item = ProofChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_nodes {
            return None;
        }

        let start_idx = self.current_index;
        let end_idx = (start_idx + self.chunk_size).min(self.total_nodes);

        let nodes: Vec<ProofNode> = self.proof.nodes()[start_idx..end_idx].to_vec();

        let total_chunks = self.total_nodes.div_ceil(self.chunk_size);
        let chunk_index = start_idx / self.chunk_size;

        let start_id = if !nodes.is_empty() {
            nodes[0].id
        } else {
            ProofNodeId(0)
        };

        self.current_index = end_idx;

        Some(ProofChunk::new(start_id, nodes, chunk_index, total_chunks))
    }
}

/// Streaming proof builder for incremental construction.
pub struct StreamingProofBuilder {
    /// Accumulated proof nodes
    nodes: Vec<ProofNode>,
    /// Node ID mapping
    node_map: FxHashMap<ProofNodeId, usize>,
    /// Current proof
    proof: Proof,
    /// Stream configuration (reserved for future use)
    #[allow(dead_code)]
    config: StreamConfig,
}

impl Default for StreamingProofBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingProofBuilder {
    /// Create a new streaming proof builder.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_map: FxHashMap::default(),
            proof: Proof::new(),
            config: StreamConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: StreamConfig) -> Self {
        Self {
            nodes: Vec::new(),
            node_map: FxHashMap::default(),
            proof: Proof::new(),
            config,
        }
    }

    /// Add an axiom to the stream.
    pub fn add_axiom(&mut self, conclusion: &str) -> ProofNodeId {
        self.proof.add_axiom(conclusion)
    }

    /// Add an inference to the stream.
    pub fn add_inference(
        &mut self,
        rule: &str,
        premises: Vec<ProofNodeId>,
        conclusion: &str,
    ) -> ProofNodeId {
        self.proof.add_inference(rule, premises, conclusion)
    }

    /// Flush accumulated nodes and get the current proof.
    pub fn flush(&mut self) -> Proof {
        let proof = std::mem::take(&mut self.proof);
        self.nodes.clear();
        self.node_map.clear();
        proof
    }

    /// Get the number of nodes in the stream.
    pub fn len(&self) -> usize {
        self.proof.len()
    }

    /// Check if the stream is empty.
    pub fn is_empty(&self) -> bool {
        self.proof.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_new() {
        let config = StreamConfig::new();
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.buffer_size, 8192);
        assert!(!config.compress);
    }

    #[test]
    fn test_stream_config_with_settings() {
        let config = StreamConfig::new()
            .with_chunk_size(500)
            .with_buffer_size(4096)
            .with_compression(true);
        assert_eq!(config.chunk_size, 500);
        assert_eq!(config.buffer_size, 4096);
        assert!(config.compress);
    }

    #[test]
    fn test_proof_chunk_new() {
        let chunk = ProofChunk::new(ProofNodeId(0), Vec::new(), 0, 1);
        assert_eq!(chunk.index, 0);
        assert_eq!(chunk.total_chunks, 1);
        assert!(chunk.is_empty());
        assert!(chunk.is_last());
    }

    #[test]
    fn test_proof_streamer_new() {
        let streamer = ProofStreamer::new();
        assert_eq!(streamer.config.chunk_size, 1000);
    }

    #[test]
    fn test_stream_chunks() {
        let mut proof = Proof::new();
        for i in 0..5 {
            proof.add_axiom(format!("axiom_{}", i));
        }

        let streamer = ProofStreamer::with_config(StreamConfig::new().with_chunk_size(2));
        let chunks: Vec<ProofChunk> = streamer.stream_chunks(&proof).collect();

        assert_eq!(chunks.len(), 3); // 5 nodes / 2 per chunk = 3 chunks
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 2);
        assert_eq!(chunks[2].len(), 1);
    }

    #[test]
    fn test_chunk_iterator() {
        let mut proof = Proof::new();
        proof.add_axiom("x = x");
        proof.add_axiom("y = y");

        let mut iter = ProofChunkIterator::new(&proof, 1);
        let chunk1 = iter.next();
        assert!(chunk1.is_some());
        assert_eq!(chunk1.unwrap().len(), 1);

        let chunk2 = iter.next();
        assert!(chunk2.is_some());
        assert_eq!(chunk2.unwrap().len(), 1);

        let chunk3 = iter.next();
        assert!(chunk3.is_none());
    }

    #[test]
    fn test_process_streaming() {
        let mut proof = Proof::new();
        proof.add_axiom("x = x");
        proof.add_axiom("y = y");

        let streamer = ProofStreamer::new();
        let mut count = 0;
        let result = streamer.process_streaming(&proof, |chunk| {
            count += chunk.len();
            Ok(())
        });

        assert!(result.is_ok());
        assert_eq!(count, 2);
    }

    #[test]
    fn test_streaming_builder() {
        let mut builder = StreamingProofBuilder::new();
        builder.add_axiom("x = x");
        builder.add_axiom("y = y");

        assert_eq!(builder.len(), 2);
        assert!(!builder.is_empty());

        let proof = builder.flush();
        assert_eq!(proof.len(), 2);
        assert_eq!(builder.len(), 0);
    }

    #[test]
    fn test_write_streaming() {
        let mut proof = Proof::new();
        proof.add_axiom("x = x");
        proof.add_axiom("y = y");

        let streamer = ProofStreamer::new();
        let mut output = Vec::new();
        let result = streamer.write_streaming(&proof, &mut output);

        assert!(result.is_ok());
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("axiom"));
        assert!(output_str.contains("x = x"));
    }
}
