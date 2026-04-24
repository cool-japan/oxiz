//! Benchmark subset selection (representative sampling)
//!
//! This module provides functionality to select representative subsets
//! of benchmarks for efficient testing while maintaining coverage.

use crate::benchmark::{BenchmarkStatus, SingleResult};
use crate::loader::{BenchmarkMeta, ExpectedStatus};
use rand::Rng;
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
    rng: Box<dyn Rng>,
}

impl Sampler {
    /// Create a new sampler
    #[must_use]
    pub fn new(config: SamplingConfig) -> Self {
        let rng: Box<dyn Rng> = if let Some(seed) = config.seed {
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
            SamplingStrategy::DifficultyBased => self.difficulty_sample(benchmarks),
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

        // Check whether any benchmark carries structural features.
        let all_have_structural = benchmarks.iter().all(|b| b.structural_features.is_some());

        if all_have_structural {
            // Use structural feature-space distance to maximise diversity.
            // We greedily pick the benchmark that is maximally far from the
            // already-selected set, measuring distance in a 3-dimensional
            // normalised feature space: (max_term_depth, atom_count,
            // max_quantifier_nesting).
            self.structural_diversity_sample(benchmarks, target)
        } else {
            self.feature_string_diversity_sample(benchmarks, target)
        }
    }

    /// Compute a scalar feature vector for diversity scoring.
    ///
    /// Returns `[max_term_depth, atom_count, max_quantifier_nesting]` as
    /// `f64` values when structural features are present; otherwise returns
    /// a zero vector.
    fn structural_vec(bench: &BenchmarkMeta) -> [f64; 3] {
        if let Some(ref sf) = bench.structural_features {
            [
                f64::from(sf.max_term_depth),
                f64::from(sf.atom_count),
                f64::from(sf.max_quantifier_nesting),
            ]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Euclidean distance between two 3-element feature vectors.
    fn feature_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Greedy max-min diversity selection using structural feature vectors.
    ///
    /// The algorithm is the standard "furthest-point" greedy approach:
    /// 1. Seed the selection with a random benchmark.
    /// 2. Repeatedly pick the benchmark that maximises its minimum distance
    ///    to any already-selected benchmark.
    fn structural_diversity_sample(
        &mut self,
        benchmarks: &[BenchmarkMeta],
        target: usize,
    ) -> Vec<BenchmarkMeta> {
        if benchmarks.is_empty() || target == 0 {
            return Vec::new();
        }

        // Pre-compute normalised feature vectors.
        let vecs: Vec<[f64; 3]> = benchmarks.iter().map(Self::structural_vec).collect();

        // Normalise each dimension to [0, 1] to avoid dominance by large-magnitude
        // dimensions (e.g., atom_count can be thousands while quantifier_nesting is 0–5).
        let mut max_vals = [f64::NEG_INFINITY; 3];
        for v in &vecs {
            for (j, &val) in v.iter().enumerate() {
                if val > max_vals[j] {
                    max_vals[j] = val;
                }
            }
        }
        // Avoid division by zero for degenerate dimensions.
        let scale: [f64; 3] = [
            if max_vals[0] > 0.0 { max_vals[0] } else { 1.0 },
            if max_vals[1] > 0.0 { max_vals[1] } else { 1.0 },
            if max_vals[2] > 0.0 { max_vals[2] } else { 1.0 },
        ];
        let norm_vecs: Vec<[f64; 3]> = vecs
            .iter()
            .map(|v| [v[0] / scale[0], v[1] / scale[1], v[2] / scale[2]])
            .collect();

        let n = benchmarks.len();
        let actual_target = target.min(n);

        // Track which indices have been selected.
        let mut selected_indices: Vec<usize> = Vec::with_capacity(actual_target);

        // Seed: pick a random starting benchmark.
        let seed_idx = (self.rng.next_u64() as usize) % n;
        selected_indices.push(seed_idx);

        // For each candidate, maintain its minimum distance to any selected point.
        let mut min_dist: Vec<f64> = (0..n)
            .map(|j| {
                if j == seed_idx {
                    f64::NEG_INFINITY // already selected
                } else {
                    Self::feature_distance(&norm_vecs[j], &norm_vecs[seed_idx])
                }
            })
            .collect();

        while selected_indices.len() < actual_target {
            // Find the candidate with the maximum minimum distance.
            let best_idx = min_dist
                .iter()
                .enumerate()
                .filter(|(idx, _)| !selected_indices.contains(idx))
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx);

            match best_idx {
                Some(idx) => {
                    selected_indices.push(idx);
                    // Update min_dist for remaining candidates.
                    for (j, d) in min_dist.iter_mut().enumerate() {
                        if !selected_indices.contains(&j) {
                            let dist = Self::feature_distance(&norm_vecs[j], &norm_vecs[idx]);
                            if dist < *d {
                                *d = dist;
                            }
                        }
                    }
                }
                None => break,
            }
        }

        selected_indices
            .into_iter()
            .map(|idx| benchmarks[idx].clone())
            .collect()
    }

    /// Legacy diversity sampling using coarse feature strings when structural
    /// features are not available.
    fn feature_string_diversity_sample(
        &mut self,
        benchmarks: &[BenchmarkMeta],
        target: usize,
    ) -> Vec<BenchmarkMeta> {
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

    /// Difficulty-based sampling using structural complexity as a proxy.
    ///
    /// Benchmarks are ranked by their estimated difficulty (see
    /// [`structural_complexity_score`]) and the `target` hardest ones are
    /// returned.  When structural features are absent for all benchmarks the
    /// function falls back to file-size ordering, which is a weak but
    /// consistently available proxy.
    fn difficulty_sample(&mut self, benchmarks: &[BenchmarkMeta]) -> Vec<BenchmarkMeta> {
        let target = self.config.calculate_target(benchmarks.len());

        let mut scored: Vec<_> = benchmarks
            .iter()
            .map(|b| (b, structural_complexity_score(b)))
            .collect();

        // Sort hardest first.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(target)
            .map(|(b, _)| b.clone())
            .collect()
    }
}

/// Compute a structural complexity score from [`StructuralFeatures`].
///
/// The score is a weighted sum of key structural metrics that correlate with
/// solver difficulty. Used as a proxy when no historical timing data is
/// available.
///
/// The primary proxy is "high term depth + many atoms". Secondary structure
/// only acts as a light tie-breaker.
fn structural_complexity_score(meta: &BenchmarkMeta) -> f64 {
    if let Some(ref sf) = meta.structural_features {
        let depth_score = f64::from(sf.max_term_depth) * 4.0;
        let atom_score = f64::from(sf.atom_count);
        let tie_break_score = f64::from(sf.max_quantifier_nesting) * 0.25;
        depth_score + atom_score + tie_break_score
    } else {
        // No structural features: fall back to file-size proxy.
        meta.file_size as f64 / 1024.0
    }
}

/// Sample benchmarks based on historical difficulty
///
/// When historical timing data is available for a benchmark, it is used
/// directly as the difficulty score. For benchmarks without timing data, the
/// function falls back to `structural_complexity_score` so that uncharted
/// benchmarks are still ordered by estimated difficulty rather than all
/// receiving the same `f64::MAX` sentinel.
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

    // Score each benchmark by difficulty.
    // Prefer measured timing when available; fall back to structural proxy.
    let mut scored: Vec<_> = benchmarks
        .iter()
        .map(|b| {
            let score = if let Some(&t) = solve_times.get(&b.path) {
                t
            } else {
                structural_complexity_score(b)
            };
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
            structural_features: None,
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

    // --- Structural feature tests ---

    use crate::logic_detector::StructuralFeatures;

    fn make_meta_with_structural(
        logic: &str,
        max_term_depth: u32,
        atom_count: u32,
        max_quantifier_nesting: u32,
    ) -> BenchmarkMeta {
        static COUNTER2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1000);
        let id = COUNTER2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let sf = StructuralFeatures {
            max_term_depth,
            atom_count,
            max_quantifier_nesting,
            ..StructuralFeatures::default()
        };

        BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/structural_{}.smt2", id)),
            logic: Some(logic.to_string()),
            expected_status: None,
            file_size: 1000,
            category: None,
            structural_features: Some(sf),
        }
    }

    /// Diversity sampling with structural features should select a set
    /// distinct from what is selected without structural features.
    ///
    /// The test constructs two groups: one "shallow" group (low max_term_depth)
    /// and one "deep" group (high max_term_depth). The structural diversity
    /// sampler must include at least one benchmark from each extreme to
    /// maximise the feature-space spread.
    #[test]
    fn test_diversity_with_structural_features_differs_from_without() {
        // 5 shallow benchmarks: depth 1, atom_count 1, quant 0
        let mut benchmarks: Vec<BenchmarkMeta> = (0..5)
            .map(|_| make_meta_with_structural("QF_LIA", 1, 1, 0))
            .collect();
        // 5 deep benchmarks: depth 20, atom_count 100, quant 3
        let deep: Vec<BenchmarkMeta> = (0..5)
            .map(|_| make_meta_with_structural("QF_LIA", 20, 100, 3))
            .collect();
        benchmarks.extend(deep);

        let config_struct = SamplingConfig::with_size(4)
            .with_strategy(SamplingStrategy::Diversity)
            .with_seed(1);
        let mut sampler_struct = Sampler::new(config_struct);
        let sample_struct = sampler_struct.sample(&benchmarks);

        // The structural sampler must include at least one shallow AND one deep
        // benchmark to demonstrate feature-space diversity.
        let has_shallow = sample_struct
            .iter()
            .any(|b| b.structural_features.as_ref().map(|s| s.max_term_depth) == Some(1));
        let has_deep = sample_struct
            .iter()
            .any(|b| b.structural_features.as_ref().map(|s| s.max_term_depth) == Some(20));

        assert!(
            has_shallow,
            "structural diversity sampler should include shallow benchmarks"
        );
        assert!(
            has_deep,
            "structural diversity sampler should include deep benchmarks"
        );
    }

    /// Difficulty sampling should rank benchmarks with higher structural
    /// complexity scores first when `prefer_hard` is `true`.
    #[test]
    fn test_difficulty_sampling_with_structural_features() {
        // 5 simple benchmarks: depth 1, no quantifiers
        let simple: Vec<BenchmarkMeta> = (0..5)
            .map(|_| make_meta_with_structural("QF_LIA", 1, 1, 0))
            .collect();
        // 5 complex benchmarks: deep nesting + quantifiers
        let complex: Vec<BenchmarkMeta> = (0..5)
            .map(|_| make_meta_with_structural("LIA", 10, 50, 5))
            .collect();

        let all_benchmarks: Vec<BenchmarkMeta> =
            simple.iter().chain(complex.iter()).cloned().collect();

        let selected = sample_by_difficulty(&all_benchmarks, &[], 5, true);

        // All selected benchmarks should be from the complex group.
        let all_complex = selected.iter().all(|b| {
            b.structural_features
                .as_ref()
                .map(|s| s.max_quantifier_nesting > 0)
                .unwrap_or(false)
        });
        assert!(
            all_complex,
            "difficulty sampling with prefer_hard=true should select complex benchmarks"
        );
    }
}
