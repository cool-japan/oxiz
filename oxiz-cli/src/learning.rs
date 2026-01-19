//! Constraint Learning and Sharing Module
//!
//! This module provides functionality for tracking learned constraints (clauses/lemmas)
//! across solving sessions. It enables:
//! - Caching learned clauses by problem fingerprint
//! - Sharing learned constraints between similar problems
//! - LRU-based eviction for cache management
//! - Persistence to disk in JSON format
//!
//! # Usage
//!
//! ```rust,ignore
//! use oxiz_cli::learning::{LearnedConstraintCache, LearnedClause};
//!
//! // Create a new cache
//! let mut cache = LearnedConstraintCache::new(Some("/path/to/cache".into()));
//!
//! // Store learned clauses for a problem
//! let clause = LearnedClause::new(vec![1, -2, 3], 0.5);
//! cache.put("problem_fingerprint", vec![clause]);
//!
//! // Retrieve learned clauses for a similar problem
//! if let Some(entry) = cache.get("problem_fingerprint") {
//!     for clause in &entry.clauses {
//!         println!("Learned clause: {:?}", clause.literals);
//!     }
//! }
//! ```
//!
//! # CLI Integration
//!
//! The module integrates with the CLI via:
//! - `--learn-cache <path>`: Specify the cache file path
//! - `--share-learned`: Enable sharing learned constraints between similar problems

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

/// A learned clause/lemma from the solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedClause {
    /// Literals in the clause (using DIMACS convention: positive = true, negative = false)
    pub literals: Vec<i32>,
    /// Activity score (for prioritization)
    pub activity: f64,
    /// Generation (solving iteration when learned)
    #[serde(default)]
    pub generation: usize,
    /// LBD (Literal Block Distance) score - lower is better
    #[serde(default)]
    pub lbd: usize,
}

impl LearnedClause {
    /// Create a new learned clause
    #[allow(dead_code)]
    pub fn new(literals: Vec<i32>, activity: f64) -> Self {
        Self {
            literals,
            activity,
            generation: 0,
            lbd: 0,
        }
    }

    /// Create a learned clause with LBD score
    #[allow(dead_code)]
    pub fn with_lbd(literals: Vec<i32>, activity: f64, lbd: usize) -> Self {
        Self {
            literals,
            activity,
            generation: 0,
            lbd,
        }
    }
}

/// Cache entry for learned constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedConstraintEntry {
    /// Problem fingerprint (hash of the problem structure)
    pub fingerprint: String,
    /// Learned clauses
    pub clauses: Vec<LearnedClause>,
    /// Timestamp when cached
    pub timestamp: u64,
    /// Last access timestamp (for LRU eviction)
    #[serde(default)]
    pub last_access: u64,
    /// Number of times this entry was used
    #[serde(default)]
    pub use_count: u64,
    /// Metadata about the problem
    #[serde(default)]
    pub metadata: ProblemMetadata,
}

/// Metadata about the problem for smarter sharing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProblemMetadata {
    /// Number of variables in the problem
    #[serde(default)]
    pub num_variables: usize,
    /// Number of clauses in the original problem
    #[serde(default)]
    pub num_clauses: usize,
    /// Logic used (e.g., QF_LIA, QF_BV)
    #[serde(default)]
    pub logic: Option<String>,
    /// Theories involved
    #[serde(default)]
    pub theories: Vec<String>,
}

/// Cache for learned constraints across solving sessions
pub struct LearnedConstraintCache {
    /// Cache file path
    cache_path: PathBuf,
    /// In-memory cache indexed by fingerprint
    entries: HashMap<String, LearnedConstraintEntry>,
    /// Maximum number of entries to keep
    max_entries: usize,
    /// Maximum clauses per entry
    max_clauses_per_entry: usize,
    /// Enable sharing between similar problems
    #[allow(dead_code)]
    sharing_enabled: bool,
}

impl LearnedConstraintCache {
    /// Create a new learned constraint cache
    #[allow(dead_code)]
    pub fn new(cache_path: Option<PathBuf>) -> Self {
        let cache_path = cache_path.unwrap_or_else(|| {
            let mut path = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            path.push(".oxiz");
            path.push("learned_constraints.json");
            path
        });

        let mut cache = Self {
            cache_path,
            entries: HashMap::new(),
            max_entries: 1000,
            max_clauses_per_entry: 10000,
            sharing_enabled: false,
        };

        // Load existing cache from disk
        cache.load_from_disk();

        cache
    }

    /// Create a cache with sharing enabled
    #[allow(dead_code)]
    pub fn with_sharing(cache_path: Option<PathBuf>) -> Self {
        let mut cache = Self::new(cache_path);
        cache.sharing_enabled = true;
        cache
    }

    /// Set maximum number of entries
    #[allow(dead_code)]
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set maximum clauses per entry
    #[allow(dead_code)]
    pub fn with_max_clauses(mut self, max: usize) -> Self {
        self.max_clauses_per_entry = max;
        self
    }

    /// Compute a fingerprint for a problem
    ///
    /// The fingerprint is based on the structural characteristics of the problem,
    /// allowing similar problems to share learned constraints.
    #[allow(dead_code)]
    pub fn compute_fingerprint(script: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Extract structural features for fingerprinting
        let mut features = Vec::new();

        // Count declarations
        let declare_count = script.matches("declare-").count();
        features.push(format!("decl:{}", declare_count));

        // Count assertions
        let assert_count = script.matches("(assert").count();
        features.push(format!("assert:{}", assert_count));

        // Extract logic if present
        if let Some(logic_start) = script.find("(set-logic ") {
            let rest = &script[logic_start + 11..];
            if let Some(logic_end) = rest.find(')') {
                let logic = &rest[..logic_end];
                features.push(format!("logic:{}", logic));
            }
        }

        // Count common operators
        let and_count = script.matches("(and").count();
        let or_count = script.matches("(or").count();
        let not_count = script.matches("(not").count();
        features.push(format!("ops:{}:{}:{}", and_count, or_count, not_count));

        // Hash the features
        let mut hasher = DefaultHasher::new();
        features.hash(&mut hasher);

        // Also include content hash for exact matching
        let mut content_hasher = DefaultHasher::new();
        script.hash(&mut content_hasher);

        format!("{:x}_{:x}", hasher.finish(), content_hasher.finish())
    }

    /// Get learned constraints for a problem
    #[allow(dead_code)]
    pub fn get(&mut self, fingerprint: &str) -> Option<LearnedConstraintEntry> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let Some(entry) = self.entries.get_mut(fingerprint) {
            // Update access time for LRU
            entry.last_access = now;
            entry.use_count += 1;
            return Some(entry.clone());
        }

        None
    }

    /// Store learned constraints for a problem
    #[allow(dead_code)]
    pub fn put(&mut self, fingerprint: &str, clauses: Vec<LearnedClause>) {
        self.put_with_metadata(fingerprint, clauses, ProblemMetadata::default());
    }

    /// Store learned constraints with problem metadata
    #[allow(dead_code)]
    pub fn put_with_metadata(
        &mut self,
        fingerprint: &str,
        mut clauses: Vec<LearnedClause>,
        metadata: ProblemMetadata,
    ) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Limit clauses to max_clauses_per_entry, keeping best ones (by LBD/activity)
        if clauses.len() > self.max_clauses_per_entry {
            // Sort by LBD (lower is better), then by activity (higher is better)
            clauses.sort_by(|a, b| {
                a.lbd
                    .cmp(&b.lbd)
                    .then_with(|| b.activity.partial_cmp(&a.activity).unwrap())
            });
            clauses.truncate(self.max_clauses_per_entry);
        }

        // Check if we need to evict entries
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(fingerprint) {
            self.evict_lru();
        }

        let entry = LearnedConstraintEntry {
            fingerprint: fingerprint.to_string(),
            clauses,
            timestamp: now,
            last_access: now,
            use_count: 0,
            metadata,
        };

        self.entries.insert(fingerprint.to_string(), entry);
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| key.clone())
        {
            self.entries.remove(&lru_key);
        }
    }

    /// Save cache to disk
    #[allow(dead_code)]
    pub fn save_to_disk(&self) -> Result<(), String> {
        // Ensure parent directory exists
        if let Some(parent) = self.cache_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
        }

        let entries_vec: Vec<&LearnedConstraintEntry> = self.entries.values().collect();
        let json = serde_json::to_string_pretty(&entries_vec)
            .map_err(|e| format!("Failed to serialize cache: {}", e))?;

        fs::write(&self.cache_path, json)
            .map_err(|e| format!("Failed to write cache file: {}", e))?;

        Ok(())
    }

    /// Load cache from disk
    fn load_from_disk(&mut self) {
        if !self.cache_path.exists() {
            return;
        }

        if let Ok(contents) = fs::read_to_string(&self.cache_path)
            && let Ok(entries_vec) = serde_json::from_str::<Vec<LearnedConstraintEntry>>(&contents)
        {
            for entry in entries_vec {
                self.entries.insert(entry.fingerprint.clone(), entry);
            }
        }
    }

    /// Clear all cache entries
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear();
        let _ = fs::remove_file(&self.cache_path);
    }

    /// Get cache statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> CacheStats {
        let total_clauses: usize = self.entries.values().map(|e| e.clauses.len()).sum();
        let total_uses: u64 = self.entries.values().map(|e| e.use_count).sum();

        CacheStats {
            num_entries: self.entries.len(),
            total_clauses,
            total_uses,
            cache_path: self.cache_path.clone(),
        }
    }

    /// Merge learned constraints from multiple entries (for sharing)
    #[allow(dead_code)]
    pub fn merge_similar(&self, fingerprint: &str) -> Vec<LearnedClause> {
        let mut merged = Vec::new();
        let target_prefix = fingerprint.split('_').next().unwrap_or("");

        for (key, entry) in &self.entries {
            // Match on structural prefix (ignoring content hash)
            if key.starts_with(target_prefix) {
                merged.extend(entry.clauses.clone());
            }
        }

        // Deduplicate and sort by quality
        merged.sort_by(|a, b| {
            a.lbd
                .cmp(&b.lbd)
                .then_with(|| b.activity.partial_cmp(&a.activity).unwrap())
        });
        merged.dedup_by(|a, b| a.literals == b.literals);

        merged
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CacheStats {
    /// Number of cached entries
    pub num_entries: usize,
    /// Total number of learned clauses
    pub total_clauses: usize,
    /// Total number of cache uses
    pub total_uses: u64,
    /// Cache file path
    pub cache_path: PathBuf,
}

/// Format cache statistics for display
#[allow(dead_code)]
pub fn format_cache_stats(stats: &CacheStats) -> String {
    let mut output = String::new();
    output.push_str("Learned Constraint Cache Statistics\n");
    output.push_str("===================================\n");
    output.push_str(&format!("Cache file: {}\n", stats.cache_path.display()));
    output.push_str(&format!("Entries: {}\n", stats.num_entries));
    output.push_str(&format!("Total clauses: {}\n", stats.total_clauses));
    output.push_str(&format!("Total uses: {}\n", stats.total_uses));
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_clause_creation() {
        let clause = LearnedClause::new(vec![1, -2, 3], 0.75);
        assert_eq!(clause.literals, vec![1, -2, 3]);
        assert_eq!(clause.activity, 0.75);
        assert_eq!(clause.lbd, 0);
    }

    #[test]
    fn test_learned_clause_with_lbd() {
        let clause = LearnedClause::with_lbd(vec![1, 2], 0.5, 2);
        assert_eq!(clause.lbd, 2);
    }

    #[test]
    fn test_fingerprint_computation() {
        let script1 = "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))";
        let script2 = "(set-logic QF_LIA)\n(declare-const y Int)\n(assert (> y 0))";
        let script3 =
            "(set-logic QF_BV)\n(declare-const x (_ BitVec 32))\n(assert (bvsgt x #x00000000))";

        let fp1 = LearnedConstraintCache::compute_fingerprint(script1);
        let fp2 = LearnedConstraintCache::compute_fingerprint(script2);
        let fp3 = LearnedConstraintCache::compute_fingerprint(script3);

        // Similar structure should have same prefix
        let prefix1 = fp1.split('_').next().unwrap();
        let prefix2 = fp2.split('_').next().unwrap();

        assert_eq!(
            prefix1, prefix2,
            "Similar problems should have same structural fingerprint"
        );
        assert_ne!(
            fp1, fp2,
            "Different content should have different full fingerprint"
        );
        assert_ne!(
            fp1, fp3,
            "Different logic should have different fingerprint"
        );
    }

    #[test]
    fn test_cache_put_get() {
        let cache_path = PathBuf::from("/tmp/oxiz_test_learning_cache.json");
        let _ = fs::remove_file(&cache_path);

        let mut cache = LearnedConstraintCache::new(Some(cache_path.clone()));

        let fingerprint = "test_fingerprint_12345";
        let clauses = vec![
            LearnedClause::new(vec![1, -2], 0.8),
            LearnedClause::new(vec![2, 3], 0.6),
        ];

        // Initially not cached
        assert!(cache.get(fingerprint).is_none());

        // Put in cache
        cache.put(fingerprint, clauses.clone());

        // Should be cached now
        let entry = cache.get(fingerprint).unwrap();
        assert_eq!(entry.clauses.len(), 2);
        assert_eq!(entry.clauses[0].literals, vec![1, -2]);

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    fn test_lru_eviction() {
        let cache_path = PathBuf::from("/tmp/oxiz_test_learning_lru.json");
        let _ = fs::remove_file(&cache_path);

        let mut cache = LearnedConstraintCache::new(Some(cache_path.clone())).with_max_entries(2);

        // Add 2 entries
        cache.put("fp1", vec![LearnedClause::new(vec![1], 0.5)]);
        cache.put("fp2", vec![LearnedClause::new(vec![2], 0.5)]);

        assert_eq!(cache.entries.len(), 2);

        // Manually set different access times to ensure deterministic LRU behavior
        // fp1 has more recent access (larger timestamp), fp2 has older access
        if let Some(entry) = cache.entries.get_mut("fp1") {
            entry.last_access = 2000; // More recent
        }
        if let Some(entry) = cache.entries.get_mut("fp2") {
            entry.last_access = 1000; // Older, should be evicted
        }

        // Add 3rd entry - should evict fp2 (least recently used)
        cache.put("fp3", vec![LearnedClause::new(vec![3], 0.5)]);

        assert_eq!(cache.entries.len(), 2);
        assert!(cache.entries.contains_key("fp1"));
        assert!(!cache.entries.contains_key("fp2"));
        assert!(cache.entries.contains_key("fp3"));

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    fn test_cache_persistence() {
        let cache_path = PathBuf::from("/tmp/oxiz_test_learning_persist.json");
        let _ = fs::remove_file(&cache_path);

        // Create and populate cache
        {
            let mut cache = LearnedConstraintCache::new(Some(cache_path.clone()));
            cache.put(
                "persistent_fp",
                vec![LearnedClause::new(vec![1, 2, 3], 0.9)],
            );
            cache.save_to_disk().unwrap();
        }

        // Load cache in a new instance
        {
            let mut cache = LearnedConstraintCache::new(Some(cache_path.clone()));
            let entry = cache.get("persistent_fp").unwrap();
            assert_eq!(entry.clauses[0].literals, vec![1, 2, 3]);
        }

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    fn test_cache_stats() {
        let cache_path = PathBuf::from("/tmp/oxiz_test_learning_stats.json");
        let _ = fs::remove_file(&cache_path);

        let mut cache = LearnedConstraintCache::new(Some(cache_path.clone()));

        cache.put(
            "fp1",
            vec![
                LearnedClause::new(vec![1], 0.5),
                LearnedClause::new(vec![2], 0.5),
            ],
        );
        cache.put("fp2", vec![LearnedClause::new(vec![3], 0.5)]);

        let stats = cache.stats();
        assert_eq!(stats.num_entries, 2);
        assert_eq!(stats.total_clauses, 3);

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    fn test_merge_similar() {
        let cache_path = PathBuf::from("/tmp/oxiz_test_learning_merge.json");
        let _ = fs::remove_file(&cache_path);

        let mut cache = LearnedConstraintCache::new(Some(cache_path.clone()));

        // Add entries with similar structural prefix
        cache.put("abc_123", vec![LearnedClause::with_lbd(vec![1, 2], 0.5, 2)]);
        cache.put("abc_456", vec![LearnedClause::with_lbd(vec![3, 4], 0.7, 1)]);
        cache.put("def_789", vec![LearnedClause::with_lbd(vec![5, 6], 0.6, 3)]);

        let merged = cache.merge_similar("abc_000");
        assert_eq!(merged.len(), 2);
        // Should be sorted by LBD (lower first)
        assert_eq!(merged[0].literals, vec![3, 4]); // LBD 1
        assert_eq!(merged[1].literals, vec![1, 2]); // LBD 2

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }
}
