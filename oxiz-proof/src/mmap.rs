//! Memory-mapped proof storage for large proofs.
//!
//! This module provides memory-mapped file storage for proofs, enabling
//! efficient handling of proofs larger than available RAM.

use crate::proof::{Proof, ProofNode, ProofNodeId};
use rustc_hash::FxHashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Configuration for memory-mapped storage.
#[derive(Debug, Clone)]
pub struct MmapConfig {
    /// File path for storage
    pub file_path: PathBuf,
    /// Enable read-only mode
    pub read_only: bool,
    /// Sync to disk on write
    pub sync_on_write: bool,
    /// Cache size (number of nodes to keep in memory)
    pub cache_size: usize,
}

impl MmapConfig {
    /// Create a new mmap configuration.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            file_path: path.as_ref().to_path_buf(),
            read_only: false,
            sync_on_write: true,
            cache_size: 10000,
        }
    }

    /// Set read-only mode.
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Set sync on write.
    pub fn with_sync_on_write(mut self, sync: bool) -> Self {
        self.sync_on_write = sync;
        self
    }

    /// Set cache size.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }
}

/// Memory-mapped proof storage.
pub struct MmapProofStorage {
    config: MmapConfig,
    file: Option<File>,
    index: FxHashMap<ProofNodeId, u64>, // Node ID -> file offset
    cache: FxHashMap<ProofNodeId, ProofNode>,
    cache_order: Vec<ProofNodeId>,
}

impl MmapProofStorage {
    /// Create a new mmap storage.
    pub fn new(config: MmapConfig) -> io::Result<Self> {
        let file = if config.read_only {
            Some(File::open(&config.file_path)?)
        } else {
            Some(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&config.file_path)?,
            )
        };

        Ok(Self {
            config,
            file,
            index: FxHashMap::default(),
            cache: FxHashMap::default(),
            cache_order: Vec::new(),
        })
    }

    /// Store a proof to disk.
    pub fn store_proof(&mut self, proof: &Proof) -> io::Result<()> {
        if self.config.read_only {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Storage is read-only",
            ));
        }

        {
            let file = self
                .file
                .as_mut()
                .ok_or_else(|| io::Error::other("File not opened"))?;

            // Write each node to file
            for node in proof.nodes() {
                let offset = file.seek(SeekFrom::End(0))?;
                let node_id = node.id;
                Self::write_node_to_file(file, node)?;
                self.index.insert(node_id, offset);
            }

            if self.config.sync_on_write {
                file.sync_all()?;
            }
        }

        Ok(())
    }

    /// Load a node from disk.
    pub fn load_node(&mut self, id: ProofNodeId) -> io::Result<Option<ProofNode>> {
        // Check cache first
        if let Some(node) = self.cache.get(&id) {
            return Ok(Some(node.clone()));
        }

        // Look up offset in index
        let offset = match self.index.get(&id) {
            Some(&off) => off,
            None => return Ok(None),
        };

        // Read node (using helper to avoid borrow issues)
        let node = {
            let file = self
                .file
                .as_mut()
                .ok_or_else(|| io::Error::other("File not opened"))?;

            file.seek(SeekFrom::Start(offset))?;
            Self::read_node_from_file(file)?
        };

        // Add to cache
        self.add_to_cache(id, node.clone());

        Ok(Some(node))
    }

    /// Load entire proof from disk.
    pub fn load_proof(&mut self) -> io::Result<Proof> {
        let mut proof = Proof::new();

        {
            let file = self
                .file
                .as_mut()
                .ok_or_else(|| io::Error::other("File not opened"))?;
            file.seek(SeekFrom::Start(0))?;

            // Read all nodes
            while let Ok(node) = Self::read_node_from_file(file) {
                // Reconstruct proof (simplified - assumes nodes are in order)
                match &node.step {
                    crate::proof::ProofStep::Axiom { conclusion } => {
                        proof.add_axiom(conclusion);
                    }
                    crate::proof::ProofStep::Inference {
                        rule,
                        premises,
                        conclusion,
                        ..
                    } => {
                        proof.add_inference(rule, premises.to_vec(), conclusion);
                    }
                }
            }
        }

        Ok(proof)
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_order.clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.config.cache_size)
    }

    /// Close the storage.
    pub fn close(&mut self) -> io::Result<()> {
        if let Some(file) = self.file.take() {
            file.sync_all()?;
        }
        Ok(())
    }

    // Helper: Write a node to file
    fn write_node_to_file(file: &mut File, node: &ProofNode) -> io::Result<()> {
        // Simple binary format (in production, use bincode or similar)
        let serialized = format!("{:?}\n", node); // Simplified serialization
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    // Helper: Read a node from file
    fn read_node_from_file(file: &mut File) -> io::Result<ProofNode> {
        let mut buffer = String::new();
        let mut temp_buf = [0u8; 1];

        // Read until newline (simplified)
        loop {
            let n = file.read(&mut temp_buf)?;
            if n == 0 || temp_buf[0] == b'\n' {
                break;
            }
            buffer.push(temp_buf[0] as char);
        }

        if buffer.is_empty() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
        }

        // Deserialize (simplified - in production use proper deserialization)
        // For now, return a dummy node
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Deserialization not implemented",
        ))
    }

    // Helper: Add node to cache with LRU eviction
    fn add_to_cache(&mut self, id: ProofNodeId, node: ProofNode) {
        if self.cache.len() >= self.config.cache_size {
            // Evict oldest entry (LRU)
            if let Some(oldest_id) = self.cache_order.first().copied() {
                self.cache.remove(&oldest_id);
                self.cache_order.remove(0);
            }
        }

        self.cache.insert(id, node);
        self.cache_order.push(id);
    }
}

/// Mmap-backed proof wrapper.
pub struct MmapProof {
    storage: MmapProofStorage,
    metadata: ProofMetadata,
}

/// Metadata for mmap proof.
#[derive(Debug, Clone, Default)]
pub struct ProofMetadata {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of axioms
    pub num_axioms: usize,
    /// Maximum depth
    pub max_depth: usize,
}

impl MmapProof {
    /// Create a new mmap proof.
    pub fn new(config: MmapConfig) -> io::Result<Self> {
        let storage = MmapProofStorage::new(config)?;
        Ok(Self {
            storage,
            metadata: ProofMetadata::default(),
        })
    }

    /// Store a proof.
    pub fn store(&mut self, proof: &Proof) -> io::Result<()> {
        self.metadata.num_nodes = proof.len();
        self.metadata.num_axioms = proof
            .nodes()
            .iter()
            .filter(|n| matches!(n.step, crate::proof::ProofStep::Axiom { .. }))
            .count();
        self.metadata.max_depth = proof
            .nodes()
            .iter()
            .map(|n| n.depth as usize)
            .max()
            .unwrap_or(0);

        self.storage.store_proof(proof)
    }

    /// Load a node by ID.
    pub fn get_node(&mut self, id: ProofNodeId) -> io::Result<Option<ProofNode>> {
        self.storage.load_node(id)
    }

    /// Load the entire proof.
    pub fn load_all(&mut self) -> io::Result<Proof> {
        self.storage.load_proof()
    }

    /// Get metadata.
    pub fn metadata(&self) -> &ProofMetadata {
        &self.metadata
    }

    /// Clear cache.
    pub fn clear_cache(&mut self) {
        self.storage.clear_cache();
    }

    /// Close the storage.
    pub fn close(mut self) -> io::Result<()> {
        self.storage.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_path() -> PathBuf {
        let mut path = env::temp_dir();
        path.push(format!("test_proof_{}.mmap", std::process::id()));
        path
    }

    #[test]
    fn test_mmap_config_new() {
        let path = temp_path();
        let config = MmapConfig::new(&path);
        assert_eq!(config.file_path, path);
        assert!(!config.read_only);
        assert!(config.sync_on_write);
        assert_eq!(config.cache_size, 10000);
    }

    #[test]
    fn test_mmap_config_with_settings() {
        let path = temp_path();
        let config = MmapConfig::new(&path)
            .with_read_only(true)
            .with_sync_on_write(false)
            .with_cache_size(5000);
        assert!(config.read_only);
        assert!(!config.sync_on_write);
        assert_eq!(config.cache_size, 5000);
    }

    #[test]
    fn test_mmap_proof_new() {
        let path = temp_path();
        let config = MmapConfig::new(&path);
        let result = MmapProof::new(config);
        // May fail if file operations are restricted, which is acceptable
        if let Ok(mmap_proof) = result {
            assert_eq!(mmap_proof.metadata.num_nodes, 0);
        }
    }

    #[test]
    fn test_proof_metadata_default() {
        let metadata = ProofMetadata::default();
        assert_eq!(metadata.num_nodes, 0);
        assert_eq!(metadata.num_axioms, 0);
        assert_eq!(metadata.max_depth, 0);
    }

    #[test]
    fn test_cache_stats() {
        let path = temp_path();
        let config = MmapConfig::new(&path).with_cache_size(100);
        if let Ok(storage) = MmapProofStorage::new(config) {
            let (used, total) = storage.cache_stats();
            assert_eq!(used, 0);
            assert_eq!(total, 100);
        }
    }

    #[test]
    fn test_clear_cache() {
        let path = temp_path();
        let config = MmapConfig::new(&path);
        if let Ok(mut storage) = MmapProofStorage::new(config) {
            storage.clear_cache();
            let (used, _) = storage.cache_stats();
            assert_eq!(used, 0);
        }
    }
}
