//! Benchmark subset selection (representative sampling)
//!
//! This module provides functionality to select representative subsets
//! of benchmarks for efficient testing while maintaining coverage.

use crate::benchmark::{BenchmarkStatus, SingleResult};
use crate::loader::{BenchmarkMeta, ExpectedStatus};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Sampling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Random sampling
    Random,
    /// Stratified by logic
    StratifiedByLogic,
    /// Stratified by expected status
    StratifiedByStatus,
    /// Stratified by file size buckets
    StratifiedBySize,
    /// Diversity sampling (maximize coverage)
    Diversity,
    /// Difficulty-based (select hard benchmarks)
    DifficultyBased,
}

/// Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Sampling strategy
    pub strategy: SamplingStrategy,
    /// Target sample size (absolute number)
    pub target_size: Option<usize>,
    /// Target sample fraction (0.0-1.0)
    pub target_fraction: Option<f64>,
    /// Minimum samples per category (for stratified)
    pub min_per_category: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Include all from specified logics
    pub required_logics: HashSet<String>,
    /// Size bucket boundaries for size-based stratification
    pub size_buckets: Vec<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::StratifiedByLogic,
            target_size: None,
            target_fraction: Some(0.1), // 10%
            min_per_category: 1,
            seed: None,
            required_logics: HashSet::new(),
            size_buckets: vec![1024, 10240, 102400, 1024000], // 1KB, 10KB, 100KB, 1MB
        }
    }
}

impl SamplingConfig {
    /// Create config with target size
    #[must_use]
    pub fn with_size(size: usize) -> Self {
        Self {
            target_size: Some(size),
            target_fraction: None,
            ..Default::default()
        }
    }

    /// Create config with target fraction
    #[must_use]
    pub fn with_fraction(fraction: f64) -> Self {
        Self {
            target_size: None,
            target_fraction: Some(fraction.clamp(0.0, 1.0)),
            ..Default::default()
        }
    }

    /// Set sampling strategy
    #[must_use]
    pub fn with_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add required logic
    #[must_use]
    pub fn with_required_logic(mut self, logic: impl Into<String>) -> Self {
        self.required_logics.insert(logic.into());
        self
    }

    /// Calculate target size from config
    fn calculate_target(&self, total: usize) -> usize {
        if let Some(size) = self.target_size {
            size.min(total)
        } else if let Some(frac) = self.target_fraction {
            ((total as f64) * frac).ceil() as usize
        } else {
            total
        }
    }
}

/// Benchmark sampler
pub struct Sampler {
    config: SamplingConfig,
    rng: Box<dyn RngCore>,
}

impl Sampler {
    /// Create a new sampler
    #[must_use]
    pub fn new(config: SamplingConfig) -> Self {
        let rng: Box<dyn RngCore> = if let Some(seed) = config.seed {
            Box::new(StdRng::seed_from_u64(seed))
        } else {
            Box::new(rand::rng())
        };

        Self { config, rng }
    }

    /// Sample benchmarks according to config
    pub fn sample(&mut self, benchmarks: &[BenchmarkMeta]) -> Vec<BenchmarkMeta> {
        match self.config.strategy {
            SamplingStrategy::Random => self.random_sample(benchmarks),
            SamplingStrategy::StratifiedByLogic => self.stratified_by_logic(benchmarks),
            SamplingStrategy::StratifiedByStatus => self.stratified_by_status(benchmarks),
            SamplingStrategy::StratifiedBySize => self.stratified_by_size(benchmarks),
            SamplingStrategy::Diversity => self.diversity_sample(benchmarks),
            SamplingStrategy::DifficultyBased => self.random_sample(benchmarks), // Needs historical data
        }
    }

    /// Random sampling
    fn random_sample(&mut self, benchmarks: &[BenchmarkMeta]) -> Vec<BenchmarkMeta> {
        let target = self.config.calculate_target(benchmarks.len());

        let mut indices: Vec<usize> = (0..benchmarks.len()).collect();
        indices.shuffle(&mut self.rng);

        indices
            .into_iter()
            .take(target)
            .map(|i| benchmarks[i].clone())
            .collect()
    }

    /// Stratified sampling by logic
    fn stratified_by_logic(&mut self, benchmarks: &[BenchmarkMeta]) -> Vec<BenchmarkMeta> {
        let target = self.config.calculate_target(benchmarks.len());

        // Group by logic
        let mut by_logic: HashMap<String, Vec<&BenchmarkMeta>> = HashMap::new();
        for bench in benchmarks {
            let logic = bench.logic.clone().unwrap_or_else(|| "UNKNOWN".to_string());
            by_logic.entry(logic).or_default().push(bench);
        }

        self.stratified_sample(&by_logic, target)
    }

    /// Stratified sampling by expected status
    fn stratified_by_status(&mut self, benchmarks: &[BenchmarkMeta]) -> Vec<BenchmarkMeta> {
        let target = self.config.calculate_target(benchmarks.len());

        // Group by status
        let mut by_status: HashMap<String, Vec<&BenchmarkMeta>> = HashMap::new();
        for bench in benchmarks {
            let status = match bench.expected_status {
                Some(ExpectedStatus::Sat) => "sat",
                Some(ExpectedStatus::Unsat) => "unsat",
                Some(ExpectedStatus::Unknown) => "unknown",
                None => "none",
            };
            by_status.entry(status.to_string()).or_default().push(bench);
        }

        self.stratified_sample(&by_status, target)
    }

    /// Stratified sampling by file size
    fn stratified_by_size(&mut self, benchmarks: &[BenchmarkMeta]) -> Vec<BenchmarkMeta> {
        let target = self.config.calculate_target(benchmarks.len());

        // Group by size bucket
        let mut by_size: HashMap<String, Vec<&BenchmarkMeta>> = HashMap::new();
        for bench in benchmarks {
            let bucket = self.size_bucket(bench.file_size);
            by_size.entry(bucket).or_default().push(bench);
        }

        self.stratified_sample(&by_size, target)
    }

    /// Get size bucket name
    fn size_bucket(&self, size: u64) -> String {
        for (i, &boundary) in self.config.size_buckets.iter().enumerate() {
            if size < boundary {
                return format!("bucket_{}", i);
            }
        }
        format!("bucket_{}", self.config.size_buckets.len())
    }

    /// Generic stratified sampling
    fn stratified_sample(
        &mut self,
        groups: &HashMap<String, Vec<&BenchmarkMeta>>,
        target: usize,
    ) -> Vec<BenchmarkMeta> {
        let mut result = Vec::new();
        let total: usize = groups.values().map(|v| v.len()).sum();

        if total == 0 {
            return result;
        }

        // First pass: ensure minimum per category
        for members in groups.values() {
            let n = self.config.min_per_category.min(members.len());
            let mut indices: Vec<usize> = (0..members.len()).collect();
            indices.shuffle(&mut self.rng);

            for i in indices.into_iter().take(n) {
                result.push(members[i].clone());
            }
        }

        // Second pass: proportional allocation for remaining
        let remaining = target.saturating_sub(result.len());
        if remaining > 0 {
            let mut additional = Vec::new();

            for members in groups.values() {
                let proportion = members.len() as f64 / total as f64;
                let allocation = ((remaining as f64 * proportion).ceil() as usize)
                    .min(members.len().saturating_sub(self.config.min_per_category));

                let mut indices: Vec<usize> =
                    (self.config.min_per_category..members.len()).collect();
                indices.shuffle(&mut self.rng);

                for i in indices.into_iter().take(allocation) {
                    additional.push(members[i].clone());
                }
            }

            // Shuffle and take only what we need
            additional.shuffle(&mut self.rng);
            result.extend(additional.into_iter().take(remaining));
        }

        result
    }

    /// Diversity sampling (maximize coverage)
    fn diversity_sample(&mut self, benchmarks: &[BenchmarkMeta]) -> Vec<BenchmarkMeta> {
        let target = self.config.calculate_target(benchmarks.len());

        // For diversity, we want to maximize coverage of different characteristics
        // Combine logic, status, and size bucket as features
        let mut by_features: HashMap<String, Vec<&BenchmarkMeta>> = HashMap::new();

        for bench in benchmarks {
            let logic = bench.logic.as_deref().unwrap_or("UNKNOWN");
            let status = match bench.expected_status {
                Some(ExpectedStatus::Sat) => "sat",
                Some(ExpectedStatus::Unsat) => "unsat",
                _ => "unk",
            };
            let size_bucket = self.size_bucket(bench.file_size);
            let feature = format!("{}_{}_{}", logic, status, size_bucket);

            by_features.entry(feature).or_default().push(bench);
        }

        // Take at least one from each feature combination if possible
        let mut result = Vec::new();
        let mut feature_keys: Vec<_> = by_features.keys().cloned().collect();
        feature_keys.shuffle(&mut self.rng);

        for key in &feature_keys {
            if result.len() >= target {
                break;
            }
            if let Some(members) = by_features.get(key)
                && !members.is_empty()
            {
                let idx = (self.rng.next_u64() as usize) % members.len();
                result.push(members[idx].clone());
            }
        }

        // Fill remaining with random selection
        if result.len() < target {
            let selected: HashSet<_> = result.iter().map(|b| &b.path).collect();
            let remaining: Vec<_> = benchmarks
                .iter()
                .filter(|b| !selected.contains(&b.path))
                .collect();

            let mut indices: Vec<usize> = (0..remaining.len()).collect();
            indices.shuffle(&mut self.rng);

            for i in indices.into_iter().take(target - result.len()) {
                result.push(remaining[i].clone());
            }
        }

        result
    }
}

/// Sample benchmarks based on historical difficulty
pub fn sample_by_difficulty(
    benchmarks: &[BenchmarkMeta],
    historical_results: &[SingleResult],
    target_size: usize,
    prefer_hard: bool,
) -> Vec<BenchmarkMeta> {
    // Build map of path -> solve time
    let solve_times: HashMap<_, _> = historical_results
        .iter()
        .filter(|r| matches!(r.status, BenchmarkStatus::Sat | BenchmarkStatus::Unsat))
        .map(|r| (r.path.clone(), r.time.as_secs_f64()))
        .collect();

    // Score each benchmark by difficulty
    let mut scored: Vec<_> = benchmarks
        .iter()
        .map(|b| {
            let score = solve_times.get(&b.path).copied().unwrap_or(f64::MAX);
            (b, score)
        })
        .collect();

    // Sort by difficulty
    if prefer_hard {
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    scored
        .into_iter()
        .take(target_size)
        .map(|(b, _)| b.clone())
        .collect()
}

/// Sample summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleSummary {
    /// Original count
    pub original_count: usize,
    /// Sample count
    pub sample_count: usize,
    /// Sampling fraction
    pub fraction: f64,
    /// Logics covered
    pub logics_covered: usize,
    /// Total logics in original
    pub total_logics: usize,
    /// Coverage by logic
    pub logic_coverage: HashMap<String, (usize, usize)>, // (sampled, total)
}

impl SampleSummary {
    /// Create summary from original and sampled benchmarks
    #[must_use]
    pub fn from_benchmarks(original: &[BenchmarkMeta], sampled: &[BenchmarkMeta]) -> Self {
        let mut original_logics: HashMap<String, usize> = HashMap::new();
        let mut sampled_logics: HashMap<String, usize> = HashMap::new();

        for b in original {
            let logic = b.logic.clone().unwrap_or_else(|| "UNKNOWN".to_string());
            *original_logics.entry(logic).or_insert(0) += 1;
        }

        for b in sampled {
            let logic = b.logic.clone().unwrap_or_else(|| "UNKNOWN".to_string());
            *sampled_logics.entry(logic).or_insert(0) += 1;
        }

        let logic_coverage: HashMap<_, _> = original_logics
            .iter()
            .map(|(logic, &total)| {
                let sampled = sampled_logics.get(logic).copied().unwrap_or(0);
                (logic.clone(), (sampled, total))
            })
            .collect();

        let logics_covered = sampled_logics.len();
        let total_logics = original_logics.len();

        Self {
            original_count: original.len(),
            sample_count: sampled.len(),
            fraction: sampled.len() as f64 / original.len().max(1) as f64,
            logics_covered,
            total_logics,
            logic_coverage,
        }
    }

    /// Get coverage percentage
    #[must_use]
    pub fn logic_coverage_pct(&self) -> f64 {
        (self.logics_covered as f64 / self.total_logics.max(1) as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_meta(logic: &str, status: Option<ExpectedStatus>, size: u64) -> BenchmarkMeta {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/bench_{}.smt2", id)),
            logic: Some(logic.to_string()),
            expected_status: status,
            file_size: size,
            category: None,
        }
    }

    #[test]
    fn test_random_sampling() {
        let benchmarks: Vec<_> = (0..100)
            .map(|_| make_meta("QF_LIA", Some(ExpectedStatus::Sat), 1000))
            .collect();

        let config = SamplingConfig::with_size(10).with_strategy(SamplingStrategy::Random);
        let mut sampler = Sampler::new(config);
        let sample = sampler.sample(&benchmarks);

        assert_eq!(sample.len(), 10);
    }

    #[test]
    fn test_stratified_by_logic() {
        let mut benchmarks = Vec::new();
        for _ in 0..50 {
            benchmarks.push(make_meta("QF_LIA", None, 1000));
        }
        for _ in 0..30 {
            benchmarks.push(make_meta("QF_BV", None, 1000));
        }
        for _ in 0..20 {
            benchmarks.push(make_meta("QF_UF", None, 1000));
        }

        let config = SamplingConfig::with_size(20)
            .with_strategy(SamplingStrategy::StratifiedByLogic)
            .with_seed(42);
        let mut sampler = Sampler::new(config);
        let sample = sampler.sample(&benchmarks);

        assert_eq!(sample.len(), 20);

        // Check that all logics are represented
        let logics: HashSet<_> = sample.iter().filter_map(|b| b.logic.as_ref()).collect();
        assert!(logics.contains(&"QF_LIA".to_string()));
        assert!(logics.contains(&"QF_BV".to_string()));
        assert!(logics.contains(&"QF_UF".to_string()));
    }

    #[test]
    fn test_stratified_by_status() {
        let mut benchmarks = Vec::new();
        for _ in 0..40 {
            benchmarks.push(make_meta("QF_LIA", Some(ExpectedStatus::Sat), 1000));
        }
        for _ in 0..40 {
            benchmarks.push(make_meta("QF_LIA", Some(ExpectedStatus::Unsat), 1000));
        }
        for _ in 0..20 {
            benchmarks.push(make_meta("QF_LIA", None, 1000));
        }

        let config = SamplingConfig::with_size(15)
            .with_strategy(SamplingStrategy::StratifiedByStatus)
            .with_seed(42);
        let mut sampler = Sampler::new(config);
        let sample = sampler.sample(&benchmarks);

        // Should have representation from all status categories
        let has_sat = sample
            .iter()
            .any(|b| b.expected_status == Some(ExpectedStatus::Sat));
        let has_unsat = sample
            .iter()
            .any(|b| b.expected_status == Some(ExpectedStatus::Unsat));
        assert!(has_sat);
        assert!(has_unsat);
    }

    #[test]
    fn test_reproducibility_with_seed() {
        let benchmarks: Vec<_> = (0..100).map(|_| make_meta("QF_LIA", None, 1000)).collect();

        let config1 = SamplingConfig::with_size(10).with_seed(12345);
        let config2 = SamplingConfig::with_size(10).with_seed(12345);

        let mut sampler1 = Sampler::new(config1);
        let mut sampler2 = Sampler::new(config2);

        let sample1 = sampler1.sample(&benchmarks);
        let sample2 = sampler2.sample(&benchmarks);

        // Same seed should produce same samples
        let paths1: HashSet<_> = sample1.iter().map(|b| &b.path).collect();
        let paths2: HashSet<_> = sample2.iter().map(|b| &b.path).collect();
        assert_eq!(paths1, paths2);
    }

    #[test]
    fn test_sample_summary() {
        let original: Vec<_> = vec![
            make_meta("QF_LIA", None, 1000),
            make_meta("QF_LIA", None, 1000),
            make_meta("QF_BV", None, 1000),
        ];

        let sampled = vec![original[0].clone(), original[2].clone()];

        let summary = SampleSummary::from_benchmarks(&original, &sampled);

        assert_eq!(summary.original_count, 3);
        assert_eq!(summary.sample_count, 2);
        assert_eq!(summary.logics_covered, 2);
    }
}
