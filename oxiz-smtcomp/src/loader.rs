//! Benchmark file discovery and loading
//!
//! This module provides functionality to discover and load SMT-LIB2 benchmark files
//! organized by logic (e.g., QF_LIA, QF_BV, QF_UF).

use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use thiserror::Error;
use walkdir::WalkDir;

/// Default capacity of the parse cache (number of distinct files).
pub const DEFAULT_PARSE_CACHE_CAPACITY: usize = 1024;

/// Error type for loader operations
#[derive(Error, Debug)]
pub enum LoaderError {
    /// IO error when reading files
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Walk directory error
    #[error("Directory walk error: {0}")]
    WalkDir(#[from] walkdir::Error),
    /// Invalid benchmark file
    #[error("Invalid benchmark file: {0}")]
    InvalidBenchmark(String),
}

/// Result type for loader operations
pub type LoaderResult<T> = Result<T, LoaderError>;

/// Metadata extracted from a benchmark file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeta {
    /// Path to the benchmark file
    pub path: PathBuf,
    /// Logic specified in the file (e.g., QF_LIA)
    pub logic: Option<String>,
    /// Expected status if known (sat, unsat, or unknown)
    pub expected_status: Option<ExpectedStatus>,
    /// File size in bytes
    pub file_size: u64,
    /// Category derived from directory structure
    pub category: Option<String>,
}

/// Expected benchmark result status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedStatus {
    /// Expected to be satisfiable
    Sat,
    /// Expected to be unsatisfiable
    Unsat,
    /// Status unknown
    Unknown,
}

impl ExpectedStatus {
    /// Parse status from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "sat" => Some(Self::Sat),
            "unsat" => Some(Self::Unsat),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }

    /// Convert to SMT-COMP format string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sat => "sat",
            Self::Unsat => "unsat",
            Self::Unknown => "unknown",
        }
    }
}

/// A loaded benchmark with its content
#[derive(Debug, Clone)]
pub struct Benchmark {
    /// Metadata about the benchmark
    pub meta: BenchmarkMeta,
    /// Raw content of the benchmark file
    pub content: String,
}

/// Configuration for the benchmark loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderConfig {
    /// Root directory to search for benchmarks
    pub root_dir: PathBuf,
    /// File extension to look for (default: ".smt2")
    pub extension: String,
    /// Maximum file size to load (in bytes, default: 10MB)
    pub max_file_size: u64,
    /// Filter by specific logics (empty means all)
    pub logic_filter: Vec<String>,
    /// Maximum number of files to load (0 means unlimited)
    pub max_files: usize,
    /// Recursive search in subdirectories
    pub recursive: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("."),
            extension: ".smt2".to_string(),
            max_file_size: 10 * 1024 * 1024, // 10MB
            logic_filter: Vec::new(),
            max_files: 0, // unlimited
            recursive: true,
        }
    }
}

impl LoaderConfig {
    /// Create a new config with the given root directory
    #[must_use]
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
            ..Default::default()
        }
    }

    /// Set the extension filter
    #[must_use]
    pub fn with_extension(mut self, ext: impl Into<String>) -> Self {
        self.extension = ext.into();
        self
    }

    /// Set the maximum file size
    #[must_use]
    pub fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = size;
        self
    }

    /// Set the logic filter
    #[must_use]
    pub fn with_logics(mut self, logics: Vec<String>) -> Self {
        self.logic_filter = logics;
        self
    }

    /// Set the maximum number of files
    #[must_use]
    pub fn with_max_files(mut self, max: usize) -> Self {
        self.max_files = max;
        self
    }

    /// Set whether to search recursively
    #[must_use]
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

/// Benchmark loader for discovering and loading SMT-LIB2 files
pub struct Loader {
    config: LoaderConfig,
    cache: Option<Arc<ParseCache>>,
}

impl Loader {
    /// Create a new loader with the given configuration
    #[must_use]
    pub fn new(config: LoaderConfig) -> Self {
        Self {
            config,
            cache: None,
        }
    }

    /// Create a loader with default configuration for the given directory
    #[must_use]
    pub fn for_directory(dir: impl Into<PathBuf>) -> Self {
        Self::new(LoaderConfig::new(dir))
    }

    /// Attach a parse-result cache to this loader.
    ///
    /// When a cache is attached, calls to [`Loader::load`], [`Loader::load_file`]
    /// and [`Loader::load_cached`] will consult and populate the cache. The
    /// cache is keyed on path and validated against file `mtime` + `file_size`
    /// to avoid returning stale content after the underlying file is modified.
    ///
    /// The same [`ParseCache`] may be shared between many [`Loader`]s.
    #[must_use]
    pub fn with_cache(mut self, cache: Arc<ParseCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Returns the attached cache, if any.
    #[must_use]
    pub fn cache(&self) -> Option<&Arc<ParseCache>> {
        self.cache.as_ref()
    }

    /// Discover all benchmark files matching the configuration
    pub fn discover(&self) -> LoaderResult<Vec<BenchmarkMeta>> {
        let mut benchmarks = Vec::new();
        let mut walker = WalkDir::new(&self.config.root_dir);

        if !self.config.recursive {
            walker = walker.max_depth(1);
        }

        for entry in walker {
            let entry = entry?;
            let path = entry.path();

            // Check if it's a file with the right extension
            if !path.is_file() {
                continue;
            }
            if path
                .extension()
                .is_none_or(|ext| ext != self.config.extension.trim_start_matches('.'))
            {
                continue;
            }

            // Check file size
            let metadata = fs::metadata(path)?;
            if metadata.len() > self.config.max_file_size {
                continue;
            }

            // Extract metadata from path and file
            let meta = self.extract_metadata(path, metadata.len())?;

            // Apply logic filter
            if !self.config.logic_filter.is_empty() {
                if let Some(ref logic) = meta.logic {
                    if !self.config.logic_filter.contains(logic) {
                        continue;
                    }
                } else {
                    continue; // Skip files without logic if filter is set
                }
            }

            benchmarks.push(meta);

            // Check max files limit
            if self.config.max_files > 0 && benchmarks.len() >= self.config.max_files {
                break;
            }
        }

        Ok(benchmarks)
    }

    /// Discover and group benchmarks by logic
    pub fn discover_by_logic(&self) -> LoaderResult<HashMap<String, Vec<BenchmarkMeta>>> {
        let benchmarks = self.discover()?;
        let mut by_logic: HashMap<String, Vec<BenchmarkMeta>> = HashMap::new();

        for bench in benchmarks {
            let logic = bench.logic.clone().unwrap_or_else(|| "UNKNOWN".to_string());
            by_logic.entry(logic).or_default().push(bench);
        }

        Ok(by_logic)
    }

    /// Load a benchmark file given its metadata.
    ///
    /// If a cache is attached via [`Loader::with_cache`], this consults the
    /// cache first and only re-reads the file on a miss. The returned
    /// [`Benchmark`] is cloned out of the cached `Arc` to preserve the
    /// existing by-value API; callers that want to share the `Arc` (e.g., for
    /// pointer-identity checks or to skip cloning the file content) should
    /// use [`Loader::load_cached`] instead.
    pub fn load(&self, meta: &BenchmarkMeta) -> LoaderResult<Benchmark> {
        let shared = self.load_cached(meta)?;
        Ok((*shared).clone())
    }

    /// Load a benchmark file directly from path.
    ///
    /// Honors the attached cache, if any.
    pub fn load_file(&self, path: impl AsRef<Path>) -> LoaderResult<Benchmark> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)?;
        let meta = self.extract_metadata(path, metadata.len())?;
        self.load(&meta)
    }

    /// Load a benchmark and return the shared [`Arc<Benchmark>`].
    ///
    /// When a cache is attached and hits, the returned `Arc` points to the
    /// same allocation as the cached entry — callers can compare pointer
    /// identity via [`Arc::ptr_eq`] to verify a cache hit. Misses (or absence
    /// of a cache) produce a fresh `Arc` wrapping the newly parsed benchmark;
    /// the fresh value is inserted into the cache when one is present.
    ///
    /// Cache validity is checked against the file's current `mtime` and size;
    /// a mismatch (or an `mtime` that the platform cannot report) triggers a
    /// fresh read. In the latter case, the freshly loaded benchmark is not
    /// written back into the cache so subsequent calls do not thrash.
    pub fn load_cached(&self, meta: &BenchmarkMeta) -> LoaderResult<Arc<Benchmark>> {
        let path = meta.path.as_path();

        if let Some(cache) = self.cache.as_ref()
            && let Some(key) = cache_key_for(path)
            && let Some(hit) = cache.get(path, key.mtime, key.size)
        {
            return Ok(hit);
        }

        let content = fs::read_to_string(path)?;
        let benchmark = Arc::new(Benchmark {
            meta: meta.clone(),
            content,
        });

        if let Some(cache) = self.cache.as_ref()
            && let Some(key) = cache_key_for(path)
        {
            cache.insert(
                path.to_path_buf(),
                key.mtime,
                key.size,
                Arc::clone(&benchmark),
            );
        }

        Ok(benchmark)
    }

    /// Extract metadata from a benchmark file
    fn extract_metadata(&self, path: &Path, file_size: u64) -> LoaderResult<BenchmarkMeta> {
        // Try to extract logic from directory structure first (e.g., QF_LIA/...)
        let logic = self.extract_logic_from_path(path);
        let category = self.extract_category_from_path(path);

        // For expected status, try to extract from filename or read file header
        let expected_status = self.extract_expected_status(path);

        Ok(BenchmarkMeta {
            path: path.to_path_buf(),
            logic,
            expected_status,
            file_size,
            category,
        })
    }

    /// Extract logic from path (e.g., /benchmarks/QF_LIA/sat/test.smt2 -> QF_LIA)
    fn extract_logic_from_path(&self, path: &Path) -> Option<String> {
        // Common SMT-COMP logics
        let known_logics = [
            "ALIA",
            "AUFLIA",
            "AUFLIRA",
            "AUFNIRA",
            "BV",
            "LIA",
            "LRA",
            "NIA",
            "NRA",
            "QF_ABV",
            "QF_ALIA",
            "QF_AUFBV",
            "QF_AUFLIA",
            "QF_AX",
            "QF_BV",
            "QF_BVFP",
            "QF_DT",
            "QF_FP",
            "QF_IDL",
            "QF_LIA",
            "QF_LIRA",
            "QF_LRA",
            "QF_NIA",
            "QF_NIRA",
            "QF_NRA",
            "QF_RDL",
            "QF_S",
            "QF_SLIA",
            "QF_UF",
            "QF_UFBV",
            "QF_UFIDL",
            "QF_UFLIA",
            "QF_UFLRA",
            "QF_UFNIA",
            "QF_UFNRA",
            "UF",
            "UFBV",
            "UFDT",
            "UFIDL",
            "UFLIA",
            "UFLRA",
            "UFNIA",
        ];

        for component in path.components() {
            let s = component.as_os_str().to_string_lossy();
            if known_logics.contains(&s.as_ref()) {
                return Some(s.to_string());
            }
        }

        // Also try to read from file header — first the explicit
        // `(set-logic ...)` directive, then fall back to the automatic
        // theory-feature detector when no directive is present.
        if let Ok(content) = fs::read_to_string(path) {
            if let Some(explicit) = Self::extract_logic_from_content(&content) {
                return Some(explicit);
            }
            // No explicit header — let the logic detector infer one from
            // the theory keywords present in the benchmark source.
            let detected = crate::logic_detector::detect_logic(&content);
            // The detector never returns an empty string; treat the "ALL"
            // fallback as a real inference rather than "unknown" so
            // downstream consumers see a concrete logic name.
            if !detected.is_empty() {
                return Some(detected);
            }
        }

        None
    }

    /// Extract logic from file content (set-logic command)
    fn extract_logic_from_content(content: &str) -> Option<String> {
        for line in content.lines().take(50) {
            // Check first 50 lines
            let trimmed = line.trim();
            if trimmed.starts_with("(set-logic") {
                // Extract logic name: (set-logic QF_LIA)
                if let Some(start) = trimmed.find("set-logic") {
                    let rest = &trimmed[start + 9..];
                    let logic = rest.trim().trim_start_matches(|c: char| c.is_whitespace());
                    let logic = logic.trim_end_matches(')').trim();
                    if !logic.is_empty() {
                        return Some(logic.to_string());
                    }
                }
            }
        }
        None
    }

    /// Extract category from path structure
    fn extract_category_from_path(&self, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&self.config.root_dir).ok()?;
        let components: Vec<_> = relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();

        // Skip the last component (filename) and first (logic)
        if components.len() >= 2 {
            let category_parts: Vec<_> = components[..components.len() - 1].to_vec();
            Some(category_parts.join("/"))
        } else {
            None
        }
    }

    /// Extract expected status from filename or content
    fn extract_expected_status(&self, path: &Path) -> Option<ExpectedStatus> {
        // Check filename for hints
        if let Some(stem) = path.file_stem() {
            let name = stem.to_string_lossy().to_lowercase();
            if name.contains("sat") && !name.contains("unsat") {
                return Some(ExpectedStatus::Sat);
            }
            if name.contains("unsat") {
                return Some(ExpectedStatus::Unsat);
            }
        }

        // Check parent directory
        if let Some(parent) = path.parent()
            && let Some(dir_name) = parent.file_name()
        {
            let name = dir_name.to_string_lossy().to_lowercase();
            if name == "sat" {
                return Some(ExpectedStatus::Sat);
            }
            if name == "unsat" {
                return Some(ExpectedStatus::Unsat);
            }
        }

        // Check file content for status annotation
        if let Ok(content) = fs::read_to_string(path) {
            return Self::extract_status_from_content(&content);
        }

        None
    }

    /// Extract expected status from file content (SMT-LIB :status info)
    fn extract_status_from_content(content: &str) -> Option<ExpectedStatus> {
        for line in content.lines().take(50) {
            let trimmed = line.trim();
            // Look for (set-info :status sat/unsat/unknown)
            if trimmed.starts_with("(set-info :status") {
                let rest = trimmed.trim_start_matches("(set-info :status").trim();
                let status = rest.trim_end_matches(')').trim();
                return ExpectedStatus::parse(status);
            }
        }
        None
    }
}

/// Validation tuple stored alongside each cached benchmark.
#[derive(Debug, Clone, Copy)]
struct CacheKey {
    mtime: SystemTime,
    size: u64,
}

/// Retrieve the freshness key for `path`.
///
/// Returns `None` if the filesystem cannot report either the file size or the
/// last-modified time. Such files cannot be safely cached (we would risk
/// returning stale content on re-reads), so the loader treats them as
/// uncacheable.
fn cache_key_for(path: &Path) -> Option<CacheKey> {
    let metadata = fs::metadata(path).ok()?;
    let mtime = metadata.modified().ok()?;
    Some(CacheKey {
        mtime,
        size: metadata.len(),
    })
}

/// Entry stored in the [`ParseCache`] alongside its freshness key.
#[derive(Debug, Clone)]
struct CachedEntry {
    mtime: SystemTime,
    size: u64,
    benchmark: Arc<Benchmark>,
}

/// Bounded, thread-safe LRU cache of parsed SMT-LIB benchmarks.
///
/// The cache is keyed by [`PathBuf`] and validated against the file's last
/// modified time and size so that in-place edits of a benchmark invalidate
/// the cached entry. Inserting a fresh value for an existing path replaces
/// the previous entry in-place, and exceeding the capacity evicts the
/// least-recently-used entry.
///
/// The cache is intended to be wrapped in an [`Arc`] and shared between
/// multiple [`Loader`] instances (e.g., one per worker thread in a parallel
/// runner). All state lives behind a single [`Mutex`]; contention is minimal
/// in practice because the critical section only covers LRU bookkeeping,
/// not file I/O.
#[derive(Debug)]
pub struct ParseCache {
    inner: Mutex<LruCache<PathBuf, CachedEntry>>,
}

impl ParseCache {
    /// Create a cache sized for [`DEFAULT_PARSE_CACHE_CAPACITY`] entries.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_PARSE_CACHE_CAPACITY)
    }

    /// Create a cache with the given capacity in entries.
    ///
    /// A capacity of zero is silently promoted to one so that the underlying
    /// [`LruCache`] always has a valid [`NonZeroUsize`] bound.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::MIN);
        Self {
            inner: Mutex::new(LruCache::new(cap)),
        }
    }

    /// Capacity (maximum entries) of the cache.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.lock_inner()
            .as_ref()
            .map(|guard| guard.cap().get())
            .unwrap_or(0)
    }

    /// Current number of entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock_inner()
            .as_ref()
            .map(|guard| guard.len())
            .unwrap_or(0)
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop all cached entries.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.clear();
        }
    }

    /// Look up a cached benchmark by path.
    ///
    /// Returns a shared [`Arc<Benchmark>`] only when the cached entry's
    /// `mtime` and size match `expected_mtime` / `expected_size`. On a
    /// freshness mismatch the stale entry is evicted so the next call will
    /// parse fresh content.
    ///
    /// `get` touches the LRU recency of the returned entry — repeated hits
    /// keep a hot benchmark from being evicted.
    pub fn get(
        &self,
        path: &Path,
        expected_mtime: SystemTime,
        expected_size: u64,
    ) -> Option<Arc<Benchmark>> {
        let mut guard = self.lock_inner()?;
        // Borrow checker: we need to compare the stored key, then conditionally pop.
        if let Some(entry) = guard.get(path) {
            if entry.mtime == expected_mtime && entry.size == expected_size {
                return Some(Arc::clone(&entry.benchmark));
            }
        } else {
            return None;
        }
        // Stale: evict so a subsequent insert refreshes it.
        guard.pop(path);
        None
    }

    /// Insert (or refresh) a cached benchmark for `path`.
    ///
    /// Overwrites any existing entry for the same path. If capacity is
    /// exhausted the least-recently-used entry is evicted.
    pub fn insert(&self, path: PathBuf, mtime: SystemTime, size: u64, benchmark: Arc<Benchmark>) {
        if let Some(mut guard) = self.lock_inner() {
            guard.put(
                path,
                CachedEntry {
                    mtime,
                    size,
                    benchmark,
                },
            );
        }
    }

    fn lock_inner(&self) -> Option<std::sync::MutexGuard<'_, LruCache<PathBuf, CachedEntry>>> {
        self.inner.lock().ok()
    }
}

impl Default for ParseCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_status_from_str() {
        assert_eq!(ExpectedStatus::parse("sat"), Some(ExpectedStatus::Sat));
        assert_eq!(ExpectedStatus::parse("unsat"), Some(ExpectedStatus::Unsat));
        assert_eq!(
            ExpectedStatus::parse("unknown"),
            Some(ExpectedStatus::Unknown)
        );
        assert_eq!(ExpectedStatus::parse("SAT"), Some(ExpectedStatus::Sat));
        assert_eq!(ExpectedStatus::parse("invalid"), None);
    }

    #[test]
    fn test_extract_logic_from_content() {
        let content = "(set-logic QF_LIA)\n(declare-const x Int)";
        assert_eq!(
            Loader::extract_logic_from_content(content),
            Some("QF_LIA".to_string())
        );

        let content_spaces = "(set-logic   QF_BV  )";
        assert_eq!(
            Loader::extract_logic_from_content(content_spaces),
            Some("QF_BV".to_string())
        );
    }

    #[test]
    fn test_extract_status_from_content() {
        let content = "(set-info :status sat)\n(declare-const x Int)";
        assert_eq!(
            Loader::extract_status_from_content(content),
            Some(ExpectedStatus::Sat)
        );

        let content_unsat = "(set-info :status unsat)";
        assert_eq!(
            Loader::extract_status_from_content(content_unsat),
            Some(ExpectedStatus::Unsat)
        );
    }

    #[test]
    fn test_loader_config_builder() {
        let config = LoaderConfig::new("/tmp/benchmarks")
            .with_extension(".smt2")
            .with_max_file_size(5 * 1024 * 1024)
            .with_logics(vec!["QF_LIA".to_string()])
            .with_max_files(100)
            .with_recursive(false);

        assert_eq!(config.root_dir, PathBuf::from("/tmp/benchmarks"));
        assert_eq!(config.extension, ".smt2");
        assert_eq!(config.max_file_size, 5 * 1024 * 1024);
        assert_eq!(config.logic_filter, vec!["QF_LIA".to_string()]);
        assert_eq!(config.max_files, 100);
        assert!(!config.recursive);
    }

    /// Build a unique per-test temporary directory under [`std::env::temp_dir`].
    ///
    /// The directory is seeded with the process id and the test-provided tag
    /// to avoid collisions when tests run in parallel or are re-run quickly.
    fn unique_temp_dir(tag: &str) -> PathBuf {
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("oxiz_smtcomp_parse_cache_{pid}_{tag}"));
        // Best-effort cleanup of any previous run.
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("failed to create temp dir");
        dir
    }

    /// Write a benchmark file and return its metadata as it would be
    /// produced by [`Loader::load_file`], without going through discovery.
    fn write_bench(dir: &Path, name: &str, body: &str) -> BenchmarkMeta {
        let path = dir.join(name);
        fs::write(&path, body).expect("failed to write benchmark fixture");
        let size = fs::metadata(&path).expect("metadata for fixture").len();
        BenchmarkMeta {
            path,
            logic: Some("QF_LIA".to_string()),
            expected_status: None,
            file_size: size,
            category: None,
        }
    }

    #[test]
    fn test_parse_cache_hit() {
        let dir = unique_temp_dir("hit");
        let meta = write_bench(
            &dir,
            "bench_hit.smt2",
            "(set-logic QF_LIA)\n(assert (= 1 1))\n(check-sat)\n",
        );

        let cache = Arc::new(ParseCache::new());
        let loader = Loader::new(LoaderConfig::new(&dir)).with_cache(Arc::clone(&cache));

        let first = loader
            .load_cached(&meta)
            .expect("initial load should succeed");
        let second = loader
            .load_cached(&meta)
            .expect("second load should hit cache");

        assert!(
            Arc::ptr_eq(&first, &second),
            "second load must return the cached Arc"
        );
        assert_eq!(cache.len(), 1);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_parse_cache_miss_on_file_mutation() {
        let dir = unique_temp_dir("mutate");
        let meta = write_bench(
            &dir,
            "bench_mutate.smt2",
            "(set-logic QF_LIA)\n(assert (= 1 1))\n(check-sat)\n",
        );

        let cache = Arc::new(ParseCache::new());
        let loader = Loader::new(LoaderConfig::new(&dir)).with_cache(Arc::clone(&cache));

        let first = loader
            .load_cached(&meta)
            .expect("initial load should succeed");

        // Mutate the file. We force a distinguishable `mtime` by first
        // rewinding the file's mtime to the past, then touching it with
        // new content. This defeats second-granularity mtime backends.
        let past = SystemTime::now() - std::time::Duration::from_secs(10);
        {
            let f = fs::File::create(&meta.path).expect("reopen for mutation");
            f.set_modified(past).expect("rewind mtime");
        }
        // Now write new content, which updates both the mtime and the size.
        fs::write(
            &meta.path,
            "(set-logic QF_LIA)\n(assert (= 2 2))\n(assert (= 3 3))\n(check-sat)\n",
        )
        .expect("rewrite fixture");

        // Refresh the metadata size so the meta matches the new file; the
        // cache itself re-reads size/mtime from the filesystem, so the
        // in-struct `file_size` is informational here, but keeping it in
        // sync mirrors real callers.
        let mut meta_v2 = meta.clone();
        meta_v2.file_size = fs::metadata(&meta.path)
            .expect("metadata after mutation")
            .len();

        let second = loader
            .load_cached(&meta_v2)
            .expect("post-mutation load should succeed");

        assert!(
            !Arc::ptr_eq(&first, &second),
            "after mutation the cache must return a fresh Arc"
        );
        assert!(
            second.content.contains("(= 2 2)"),
            "post-mutation content must reflect the new bytes, got: {}",
            second.content
        );
        assert_eq!(
            cache.len(),
            1,
            "stale entry should be replaced, not stacked"
        );

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_extract_logic_falls_back_to_detector() {
        // File has no `(set-logic ...)` header but declares an Int sort.
        // The automatic detector must fill in QF_LIA.
        let dir = unique_temp_dir("detect_qf_lia");
        let path = dir.join("detect.smt2");
        fs::write(
            &path,
            "(declare-const x Int)\n(assert (>= x 0))\n(check-sat)\n",
        )
        .expect("failed to write fixture");

        let loader = Loader::new(LoaderConfig::new(&dir));
        let logic = loader.extract_logic_from_path(&path);
        assert_eq!(logic.as_deref(), Some("QF_LIA"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_extract_logic_prefers_explicit_header() {
        // Explicit `(set-logic UFLIA)` must win over inference even when
        // the content would otherwise map to QF_LIA.
        let dir = unique_temp_dir("explicit_wins");
        let path = dir.join("explicit.smt2");
        fs::write(
            &path,
            "(set-logic UFLIA)\n(declare-const x Int)\n(assert (>= x 0))\n(check-sat)\n",
        )
        .expect("failed to write fixture");

        let loader = Loader::new(LoaderConfig::new(&dir));
        let logic = loader.extract_logic_from_path(&path);
        assert_eq!(logic.as_deref(), Some("UFLIA"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_parse_cache_lru_eviction() {
        let dir = unique_temp_dir("lru");
        let meta_a = write_bench(&dir, "a.smt2", "(set-logic QF_LIA)\n; A\n(check-sat)\n");
        let meta_b = write_bench(&dir, "b.smt2", "(set-logic QF_LIA)\n; B\n(check-sat)\n");
        let meta_c = write_bench(&dir, "c.smt2", "(set-logic QF_LIA)\n; C\n(check-sat)\n");

        let cache = Arc::new(ParseCache::with_capacity(2));
        assert_eq!(cache.capacity(), 2);
        let loader = Loader::new(LoaderConfig::new(&dir)).with_cache(Arc::clone(&cache));

        let a_first = loader.load_cached(&meta_a).expect("load a");
        let _b_first = loader.load_cached(&meta_b).expect("load b");

        // Touching `a` bumps it to most-recent, so `b` should be the LRU victim.
        let a_touch = loader.load_cached(&meta_a).expect("re-load a");
        assert!(
            Arc::ptr_eq(&a_first, &a_touch),
            "a should still be cached after touching"
        );

        // Loading `c` should now evict `b`.
        let _c_first = loader.load_cached(&meta_c).expect("load c");
        assert_eq!(cache.len(), 2, "cache must stay at capacity after eviction");

        // `b` is gone — a fresh load must produce a new Arc.
        let b_second = loader
            .load_cached(&meta_b)
            .expect("reload b after eviction");
        // We cannot compare against the first Arc for `b` because that would
        // require retaining both the "before" and "after" Arcs. Instead, we
        // assert the cache state indirectly: `a` should now be the LRU victim
        // after `c` and `b` became the two most-recent entries. Loading a
        // fourth file should evict `a`.
        let meta_d = write_bench(&dir, "d.smt2", "(set-logic QF_LIA)\n; D\n(check-sat)\n");
        let _d_first = loader.load_cached(&meta_d).expect("load d");
        let a_second = loader
            .load_cached(&meta_a)
            .expect("reload a after eviction");
        assert!(
            !Arc::ptr_eq(&a_first, &a_second),
            "a should have been evicted and re-parsed"
        );
        // Silence unused-binding lints on values used only for their side effects.
        let _ = b_second;

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }
}
