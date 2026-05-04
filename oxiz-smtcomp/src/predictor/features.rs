//! Feature extraction from benchmark metadata.
//!
//! [`Features`] is a fixed-size 22-dimensional real vector computed from a
//! [`BenchmarkMeta`].  [`FeatureNormalizer`] fits per-dimension min/max
//! statistics over a training corpus and maps raw features to the `[0, 1]`
//! hypercube.

use crate::loader::BenchmarkMeta;
use crate::logic_detector::TheoryBits;

/// Total dimensionality of the feature vector.
///
/// Layout: 10 theory bits (one-hot f64) + 12 scalar features = 22 dimensions.
pub const FEATURE_DIM: usize = 22;

/// Fixed-size feature vector for a single SMT benchmark.
///
/// All fields are `f64` to simplify linear-algebra operations.
/// The vector is split into two conceptual groups:
///
/// * `theory_bits[0..10]` — one-hot encoding of detected theories
///   (order: uf, int, real, bv, array, string, fp, dt, nonlinear, quantifier)
/// * 12 scalar features derived from structural analysis and file metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Features {
    /// One-hot theory presence vector (10 dimensions, values 0.0 or 1.0).
    pub theory_bits: [f64; 10],
    /// Base-10 logarithm of `(file_size + 1)`.
    pub log_file_size: f64,
    /// Number of atomic predicate applications in the formula.
    pub atom_count: f64,
    /// Number of top-level `assert` commands.
    pub clause_count: f64,
    /// Maximum nesting depth of terms.
    pub max_term_depth: f64,
    /// Maximum nesting depth of quantifiers.
    pub max_quantifier_nesting: f64,
    /// Maximum nesting depth of `ite` expressions.
    pub max_ite_depth: f64,
    /// Maximum nesting depth of `let` binders.
    pub max_let_depth: f64,
    /// Maximum bit-vector width seen in the formula (0.0 if none).
    pub max_bv_width: f64,
    /// Sum of `width * count` across the BV width histogram.
    pub total_bv_volume: f64,
    /// Maximum array dimension seen (0.0 if no arrays).
    pub max_array_dim: f64,
    /// Sum of `dim * count` across the array dimension histogram.
    pub total_array_volume: f64,
    /// Count of theory bits that are set (0–10).
    pub theory_combination_score: f64,
}

impl Default for Features {
    fn default() -> Self {
        Self {
            theory_bits: [0.0; 10],
            log_file_size: 0.0,
            atom_count: 0.0,
            clause_count: 0.0,
            max_term_depth: 0.0,
            max_quantifier_nesting: 0.0,
            max_ite_depth: 0.0,
            max_let_depth: 0.0,
            max_bv_width: 0.0,
            total_bv_volume: 0.0,
            max_array_dim: 0.0,
            total_array_volume: 0.0,
            theory_combination_score: 0.0,
        }
    }
}

/// Extract [`TheoryBits`] from an SMT-LIB logic name string such as `"QF_LIA"`.
///
/// The logic name is parsed by looking for well-known fragments in the body
/// (the portion after an optional `"QF_"` prefix):
///
/// * `"UF"` → `has_uf`
/// * `"LIA"`, `"NIA"`, `"ALIA"`, `"LIRA"`, `"NIRA"`, `"SLIA"` → `has_int`
/// * `"LRA"`, `"NRA"`, `"LIRA"`, `"NIRA"` → `has_real`
/// * `"BV"` → `has_bv`
/// * `"A"` prefix or `"AX"` → `has_array`
/// * `"S"` body, `"SLIA"` → `has_string`
/// * `"FP"` → `has_fp`
/// * `"DT"` → `has_dt`
/// * `"NIA"`, `"NRA"`, `"NIRA"`, `"UFNIA"`, `"UFNRA"` → `has_nonlinear`
/// * No `"QF_"` prefix → `has_quantifier`
fn theory_bits_from_logic_name(logic: &str) -> TheoryBits {
    let mut bits = TheoryBits::default();
    if logic.is_empty() {
        return bits;
    }

    // Quantifier-free prefix
    let body = logic.strip_prefix("QF_").unwrap_or(logic);
    bits.has_quantifier = !logic.starts_with("QF_");

    // BV and FP
    bits.has_bv = body.contains("BV");
    bits.has_fp = body.contains("FP");

    // Datatypes
    bits.has_dt = body.contains("DT");

    // Uninterpreted Functions
    bits.has_uf = body.contains("UF");

    // Arrays: body starts with 'A' (covering AUFBV, AUFLIA, ALIA, ABV, AX, etc.)
    // or body is exactly "AX"
    bits.has_array = body.starts_with('A') && !body.starts_with("AU") // ALIA, ABV, AX
        || body.contains("AUF")   // AUFBV, AUFLIA, AUFNIRA
        || body == "AX"
        || body.starts_with("AU") && body.contains('A'); // catch-all for AU* with array

    // Integer
    bits.has_int = body.contains("LIA")
        || body.contains("NIA")
        || body.contains("LIRA")
        || body.contains("NIRA")
        || body.contains("SLIA");

    // Real
    bits.has_real = body.contains("LRA")
        || body.contains("NRA")
        || body.contains("LIRA")
        || body.contains("NIRA");

    // Nonlinear
    bits.has_nonlinear = body.contains("NIA")
        || body.contains("NRA")
        || body.contains("NIRA");

    // Strings: body is "S", "SLIA", or similar
    bits.has_string = body == "S" || body.contains("SLIA");

    bits
}

impl Features {
    /// Build a feature vector from benchmark metadata.
    ///
    /// * Theory bits are derived from the logic name string using
    ///   [`theory_bits_from_logic_name`]; if the logic is `None` all bits are 0.
    /// * Structural features default to 0.0 when
    ///   `meta.structural_features` is `None`.
    /// * `log_file_size = (file_size as f64 + 1.0).log10()`
    #[must_use]
    pub fn from_meta(meta: &BenchmarkMeta) -> Self {
        let bits = theory_bits_from_logic_name(meta.logic.as_deref().unwrap_or(""));

        let theory_bits = [
            if bits.has_uf { 1.0 } else { 0.0 },
            if bits.has_int { 1.0 } else { 0.0 },
            if bits.has_real { 1.0 } else { 0.0 },
            if bits.has_bv { 1.0 } else { 0.0 },
            if bits.has_array { 1.0 } else { 0.0 },
            if bits.has_string { 1.0 } else { 0.0 },
            if bits.has_fp { 1.0 } else { 0.0 },
            if bits.has_dt { 1.0 } else { 0.0 },
            if bits.has_nonlinear { 1.0 } else { 0.0 },
            if bits.has_quantifier { 1.0 } else { 0.0 },
        ];

        let theory_combination_score = theory_bits.iter().filter(|&&v| v > 0.0).count() as f64;
        let log_file_size = (meta.file_size as f64 + 1.0).log10();

        let (atom_count, clause_count, max_term_depth, max_quantifier_nesting,
             max_ite_depth, max_let_depth, max_bv_width, total_bv_volume,
             max_array_dim, total_array_volume) =
            if let Some(sf) = &meta.structural_features {
                let max_bv_width = sf
                    .bv_width_histogram
                    .iter()
                    .map(|(w, _)| *w)
                    .max()
                    .unwrap_or(0) as f64;
                let total_bv_volume = sf
                    .bv_width_histogram
                    .iter()
                    .map(|(w, c)| (*w as f64) * (*c as f64))
                    .sum::<f64>();
                let max_array_dim = sf
                    .array_dim_histogram
                    .iter()
                    .map(|(d, _)| *d)
                    .max()
                    .unwrap_or(0) as f64;
                let total_array_volume = sf
                    .array_dim_histogram
                    .iter()
                    .map(|(d, c)| (*d as f64) * (*c as f64))
                    .sum::<f64>();
                (
                    sf.atom_count as f64,
                    sf.clause_count as f64,
                    sf.max_term_depth as f64,
                    sf.max_quantifier_nesting as f64,
                    sf.max_ite_depth as f64,
                    sf.max_let_depth as f64,
                    max_bv_width,
                    total_bv_volume,
                    max_array_dim,
                    total_array_volume,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            };

        Self {
            theory_bits,
            log_file_size,
            atom_count,
            clause_count,
            max_term_depth,
            max_quantifier_nesting,
            max_ite_depth,
            max_let_depth,
            max_bv_width,
            total_bv_volume,
            max_array_dim,
            total_array_volume,
            theory_combination_score,
        }
    }

    /// Return the feature vector as a flat `Vec<f64>` in canonical order.
    ///
    /// Index layout:
    /// `[0..10]` theory bits, `[10]` log_file_size, `[11]` atom_count,
    /// `[12]` clause_count, `[13]` max_term_depth, `[14]` max_quantifier_nesting,
    /// `[15]` max_ite_depth, `[16]` max_let_depth, `[17]` max_bv_width,
    /// `[18]` total_bv_volume, `[19]` max_array_dim, `[20]` total_array_volume,
    /// `[21]` theory_combination_score.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(FEATURE_DIM);
        v.extend_from_slice(&self.theory_bits);
        v.push(self.log_file_size);
        v.push(self.atom_count);
        v.push(self.clause_count);
        v.push(self.max_term_depth);
        v.push(self.max_quantifier_nesting);
        v.push(self.max_ite_depth);
        v.push(self.max_let_depth);
        v.push(self.max_bv_width);
        v.push(self.total_bv_volume);
        v.push(self.max_array_dim);
        v.push(self.total_array_volume);
        v.push(self.theory_combination_score);
        debug_assert_eq!(v.len(), FEATURE_DIM);
        v
    }

    /// Reconstruct a `Features` value from a flat vector.
    ///
    /// Returns `None` if `v.len() != FEATURE_DIM`.
    #[must_use]
    pub fn from_vec(v: &[f64]) -> Option<Self> {
        if v.len() != FEATURE_DIM {
            return None;
        }
        let mut theory_bits = [0.0f64; 10];
        theory_bits.copy_from_slice(&v[0..10]);
        Some(Self {
            theory_bits,
            log_file_size: v[10],
            atom_count: v[11],
            clause_count: v[12],
            max_term_depth: v[13],
            max_quantifier_nesting: v[14],
            max_ite_depth: v[15],
            max_let_depth: v[16],
            max_bv_width: v[17],
            total_bv_volume: v[18],
            max_array_dim: v[19],
            total_array_volume: v[20],
            theory_combination_score: v[21],
        })
    }
}

/// Per-dimension min-max statistics for feature normalization.
///
/// Fitted on a training corpus; transforms raw feature vectors to `[0, 1]`
/// by clamping and rescaling.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatureNormalizer {
    /// Per-dimension minimum values (length == `FEATURE_DIM`).
    pub min: Vec<f64>,
    /// Per-dimension maximum values (length == `FEATURE_DIM`).
    pub max: Vec<f64>,
}

impl Default for FeatureNormalizer {
    fn default() -> Self {
        Self {
            min: vec![0.0; FEATURE_DIM],
            max: vec![1.0; FEATURE_DIM],
        }
    }
}

impl FeatureNormalizer {
    /// Compute per-dimension min and max over `samples`.
    ///
    /// Falls back to `min = 0, max = 1` for each dimension if `samples` is empty
    /// to avoid division-by-zero later.
    #[must_use]
    pub fn fit(samples: &[Features]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let first = samples[0].to_vec();
        let mut min = first.clone();
        let mut max = first;

        for s in samples.iter().skip(1) {
            let v = s.to_vec();
            for (i, val) in v.iter().enumerate() {
                if *val < min[i] {
                    min[i] = *val;
                }
                if *val > max[i] {
                    max[i] = *val;
                }
            }
        }

        Self { min, max }
    }

    /// Normalize `f` to `[0, 1]` per dimension, clamped.
    ///
    /// Dimensions where `max == min` are mapped to 0.0.
    #[must_use]
    pub fn normalize(&self, f: &Features) -> Vec<f64> {
        let v = f.to_vec();
        v.iter()
            .enumerate()
            .map(|(i, &val)| {
                let range = self.max[i] - self.min[i];
                if range.abs() < f64::EPSILON {
                    0.0
                } else {
                    ((val - self.min[i]) / range).clamp(0.0, 1.0)
                }
            })
            .collect()
    }

    /// Inverse-transform a normalized vector back to raw feature space.
    ///
    /// Returns `None` if the slice length is not `FEATURE_DIM`.
    #[must_use]
    pub fn denormalize(&self, normalized: &[f64]) -> Option<Features> {
        if normalized.len() != FEATURE_DIM {
            return None;
        }
        let raw: Vec<f64> = normalized
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let range = self.max[i] - self.min[i];
                v * range + self.min[i]
            })
            .collect();
        Features::from_vec(&raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::BenchmarkMeta;
    use std::path::PathBuf;

    fn make_meta_no_structural(logic: &str, size: u64) -> BenchmarkMeta {
        BenchmarkMeta {
            path: PathBuf::from("/tmp/test.smt2"),
            logic: Some(logic.to_string()),
            expected_status: None,
            file_size: size,
            category: None,
            structural_features: None,
        }
    }

    #[test]
    fn test_feature_dim() {
        let f = Features::default();
        assert_eq!(f.to_vec().len(), FEATURE_DIM);
    }

    #[test]
    fn test_from_meta_none_structural() {
        let meta = make_meta_no_structural("QF_LIA", 1000);
        let f = Features::from_meta(&meta);
        assert_eq!(f.atom_count, 0.0);
        assert_eq!(f.clause_count, 0.0);
        // QF_LIA has int
        assert_eq!(f.theory_bits[1], 1.0);
    }

    #[test]
    fn test_log_file_size() {
        let meta = make_meta_no_structural("QF_LIA", 999);
        let f = Features::from_meta(&meta);
        let expected = 1000.0f64.log10();
        assert!((f.log_file_size - expected).abs() < 1e-9);
    }

    #[test]
    fn test_round_trip_to_from_vec() {
        let f = Features {
            theory_bits: [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            log_file_size: 3.5,
            atom_count: 42.0,
            clause_count: 10.0,
            max_term_depth: 5.0,
            max_quantifier_nesting: 2.0,
            max_ite_depth: 1.0,
            max_let_depth: 3.0,
            max_bv_width: 32.0,
            total_bv_volume: 128.0,
            max_array_dim: 2.0,
            total_array_volume: 6.0,
            theory_combination_score: 2.0,
        };
        let v = f.to_vec();
        let f2 = Features::from_vec(&v).expect("round-trip failed");
        assert!((f.log_file_size - f2.log_file_size).abs() < 1e-12);
        assert!((f.atom_count - f2.atom_count).abs() < 1e-12);
        assert!((f.total_array_volume - f2.total_array_volume).abs() < 1e-12);
    }
}
