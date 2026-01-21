//! Result caching for SMT solver
//!
//! Caches solver results to speed up repeated queries

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

/// Cache entry for a solver result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Input hash (SHA256 of the problem)
    pub input_hash: String,
    /// Solver result
    pub result: String,
    /// Timestamp when cached
    pub timestamp: u64,
    /// Execution time in milliseconds
    pub time_ms: u128,
    /// Solver version
    pub solver_version: String,
    /// Last access timestamp (for LRU eviction)
    #[serde(default)]
    pub last_access: u64,
}

/// Result cache manager
pub struct ResultCache {
    /// Cache directory
    cache_dir: PathBuf,
    /// In-memory cache
    memory_cache: HashMap<String, CacheEntry>,
    /// Maximum cache size (number of entries)
    #[allow(dead_code)]
    max_entries: usize,
}

impl ResultCache {
    /// Create a new result cache
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            let mut dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            dir.push(".oxiz");
            dir.push("cache");
            dir
        });

        // Create cache directory if it doesn't exist
        let _ = fs::create_dir_all(&cache_dir);

        Self {
            cache_dir,
            memory_cache: HashMap::new(),
            max_entries: 1000,
        }
    }

    /// Compute hash of input
    #[allow(dead_code)]
    fn hash_input(input: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get cache file path for a hash
    #[allow(dead_code)]
    fn cache_file_path(&self, hash: &str) -> PathBuf {
        let mut path = self.cache_dir.clone();
        path.push(format!("{}.json", hash));
        path
    }

    /// Check if result is cached (updates LRU access time)
    #[allow(dead_code)]
    pub fn get(&mut self, input: &str) -> Option<CacheEntry> {
        let hash = Self::hash_input(input);
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("serialization should succeed")
            .as_secs();

        // Check memory cache first
        if let Some(entry) = self.memory_cache.get_mut(&hash) {
            // Update last access time for LRU
            entry.last_access = now;
            return Some(entry.clone());
        }

        // Check disk cache
        let cache_file = self.cache_file_path(&hash);
        if cache_file.exists()
            && let Ok(contents) = fs::read_to_string(&cache_file)
            && let Ok(mut entry) = serde_json::from_str::<CacheEntry>(&contents)
        {
            // Update last access time for LRU
            entry.last_access = now;

            // Add to memory cache with LRU eviction if needed
            self.insert_with_lru_eviction(hash, entry.clone());
            return Some(entry);
        }

        None
    }

    /// Insert entry with LRU eviction if cache is full
    fn insert_with_lru_eviction(&mut self, hash: String, entry: CacheEntry) {
        if self.memory_cache.len() >= self.max_entries {
            // Find and remove least recently used entry
            if let Some(lru_key) = self.find_lru_entry() {
                self.memory_cache.remove(&lru_key);
            }
        }
        self.memory_cache.insert(hash, entry);
    }

    /// Find the least recently used entry key
    fn find_lru_entry(&self) -> Option<String> {
        self.memory_cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| key.clone())
    }

    /// Store result in cache
    #[allow(dead_code)]
    pub fn put(&mut self, input: &str, result: &str, time_ms: u128) {
        let hash = Self::hash_input(input);
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("serialization should succeed")
            .as_secs();

        let entry = CacheEntry {
            input_hash: hash.clone(),
            result: result.to_string(),
            timestamp: now,
            time_ms,
            solver_version: env!("CARGO_PKG_VERSION").to_string(),
            last_access: now,
        };

        // Store in memory cache with LRU eviction
        self.insert_with_lru_eviction(hash.clone(), entry.clone());

        // Store in disk cache
        let cache_file = self.cache_file_path(&hash);
        if let Ok(json) = serde_json::to_string_pretty(&entry) {
            let _ = fs::write(cache_file, json);
        }
    }

    /// Clear all cache entries
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.memory_cache.clear();
        if self.cache_dir.exists() {
            let _ = fs::remove_dir_all(&self.cache_dir);
            let _ = fs::create_dir_all(&self.cache_dir);
        }
    }

    /// Get cache statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> CacheStats {
        let disk_entries = if self.cache_dir.exists() {
            fs::read_dir(&self.cache_dir)
                .map(|entries| entries.count())
                .unwrap_or(0)
        } else {
            0
        };

        CacheStats {
            memory_entries: self.memory_cache.len(),
            disk_entries,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CacheStats {
    /// Number of entries in memory cache
    pub memory_entries: usize,
    /// Number of entries in disk cache
    pub disk_entries: usize,
}

/// Benchmark tracking for performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    /// Problem identifier
    pub problem: String,
    /// Result (sat/unsat/unknown)
    pub result: String,
    /// Execution time in milliseconds
    pub time_ms: u128,
    /// Memory used in bytes
    pub memory_bytes: u64,
    /// Number of decisions
    pub decisions: u64,
    /// Number of conflicts
    pub conflicts: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Solver version
    pub solver_version: String,
}

/// Benchmark tracker
pub struct BenchmarkTracker {
    /// Benchmark file path
    file_path: PathBuf,
    /// Benchmark entries
    entries: Vec<BenchmarkEntry>,
}

impl BenchmarkTracker {
    /// Create a new benchmark tracker
    pub fn new(file_path: PathBuf) -> Self {
        let entries = if file_path.exists() {
            if let Ok(contents) = fs::read_to_string(&file_path) {
                serde_json::from_str(&contents).unwrap_or_else(|_| Vec::new())
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        Self { file_path, entries }
    }

    /// Add a benchmark entry
    pub fn add_entry(&mut self, entry: BenchmarkEntry) {
        self.entries.push(entry);
    }

    /// Save benchmarks to file
    pub fn save(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| format!("Failed to serialize benchmarks: {}", e))?;
        fs::write(&self.file_path, json)
            .map_err(|e| format!("Failed to write benchmarks: {}", e))?;
        Ok(())
    }

    /// Get statistics for a specific problem
    #[allow(dead_code)]
    pub fn get_stats(&self, problem: &str) -> BenchmarkStats {
        let problem_entries: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.problem == problem)
            .collect();

        if problem_entries.is_empty() {
            return BenchmarkStats::default();
        }

        let times: Vec<u128> = problem_entries.iter().map(|e| e.time_ms).collect();
        let min_time = *times.iter().min().unwrap_or(&0);
        let max_time = *times.iter().max().unwrap_or(&0);
        let avg_time = if !times.is_empty() {
            times.iter().sum::<u128>() / times.len() as u128
        } else {
            0
        };

        BenchmarkStats {
            num_runs: problem_entries.len(),
            min_time_ms: min_time,
            max_time_ms: max_time,
            avg_time_ms: avg_time,
        }
    }
}

/// Benchmark statistics
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct BenchmarkStats {
    /// Number of benchmark runs
    pub num_runs: usize,
    /// Minimum time
    pub min_time_ms: u128,
    /// Maximum time
    pub max_time_ms: u128,
    /// Average time
    pub avg_time_ms: u128,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hash() {
        let input1 = "(assert (= x 42))";
        let input2 = "(assert (= x 42))";
        let input3 = "(assert (= y 42))";

        let hash1 = ResultCache::hash_input(input1);
        let hash2 = ResultCache::hash_input(input2);
        let hash3 = ResultCache::hash_input(input3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_cache_put_get() {
        let cache_dir = PathBuf::from("/tmp/oxiz_test_cache");
        let mut cache = ResultCache::new(Some(cache_dir.clone()));

        let input = "(assert (= x 42))";
        let result = "sat";

        // Initially not cached
        assert!(cache.get(input).is_none());

        // Put in cache
        cache.put(input, result, 100);

        // Should be cached now
        let cached = cache.get(input).unwrap();
        assert_eq!(cached.result, result);
        assert_eq!(cached.time_ms, 100);

        // Cleanup
        let _ = fs::remove_dir_all(cache_dir);
    }

    #[test]
    fn test_lru_eviction() {
        let cache_dir = PathBuf::from("/tmp/oxiz_test_lru_cache_lru");
        let _ = fs::remove_dir_all(&cache_dir);

        // Create cache with max 2 entries for simpler testing
        let mut cache = ResultCache {
            cache_dir,
            memory_cache: HashMap::new(),
            max_entries: 2,
        };

        // Add 2 entries
        cache.put("(assert (= x 1))", "sat", 100);
        cache.put("(assert (= x 2))", "sat", 200);

        assert_eq!(cache.memory_cache.len(), 2);

        // Manually set different last_access times to ensure deterministic LRU
        let hash1 = ResultCache::hash_input("(assert (= x 1))");
        let hash2 = ResultCache::hash_input("(assert (= x 2))");

        if let Some(entry) = cache.memory_cache.get_mut(&hash1) {
            entry.last_access = 1000; // Older access time
        }
        if let Some(entry) = cache.memory_cache.get_mut(&hash2) {
            entry.last_access = 2000; // More recent access time
        }

        // Add 3rd entry - should evict hash1 (has oldest last_access)
        cache.put("(assert (= x 3))", "sat", 300);
        let hash3 = ResultCache::hash_input("(assert (= x 3))");

        // Verify: cache should still have 2 entries
        assert_eq!(cache.memory_cache.len(), 2);

        // hash1 should be evicted (oldest last_access)
        assert!(!cache.memory_cache.contains_key(&hash1));

        // hash2 and hash3 should remain
        assert!(cache.memory_cache.contains_key(&hash2));
        assert!(cache.memory_cache.contains_key(&hash3));
    }
}
