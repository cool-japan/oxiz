//! Integration tests for `predictor::features`.

use oxiz_smtcomp::loader::BenchmarkMeta;
use oxiz_smtcomp::logic_detector::StructuralFeatures;
use oxiz_smtcomp::predictor::{FEATURE_DIM, FeatureNormalizer, Features};
use std::path::PathBuf;

fn make_meta(logic: &str, size: u64, sf: Option<StructuralFeatures>) -> BenchmarkMeta {
    BenchmarkMeta {
        path: PathBuf::from("/tmp/test.smt2"),
        logic: Some(logic.to_string()),
        expected_status: None,
        file_size: size,
        category: None,
        structural_features: sf,
    }
}

#[test]
fn test_features_from_meta_qf_lia() {
    let meta = make_meta("QF_LIA", 1000, None);
    let f = Features::from_meta(&meta);
    // QF_LIA should set has_int bit (index 1) and nothing else from theories
    assert_eq!(f.theory_bits[1], 1.0, "QF_LIA should set int bit");
    assert_eq!(f.theory_bits[0], 0.0, "QF_LIA should not set UF bit");
    assert_eq!(f.theory_bits[3], 0.0, "QF_LIA should not set BV bit");
    // log_file_size = log10(1001)
    let expected_log = 1001.0f64.log10();
    assert!((f.log_file_size - expected_log).abs() < 1e-9);
}

#[test]
fn test_features_from_meta_qf_bv() {
    let sf = StructuralFeatures {
        max_term_depth: 10,
        atom_count: 50,
        clause_count: 20,
        max_quantifier_nesting: 0,
        bv_width_histogram: vec![(32, 5), (64, 3)],
        array_dim_histogram: vec![],
        max_ite_depth: 2,
        max_let_depth: 1,
    };
    let meta = make_meta("QF_BV", 5000, Some(sf));
    let f = Features::from_meta(&meta);
    // QF_BV should set has_bv bit (index 3)
    assert_eq!(f.theory_bits[3], 1.0, "QF_BV should set BV bit");
    assert_eq!(f.max_bv_width, 64.0, "Max BV width should be 64");
    assert_eq!(f.total_bv_volume, 32.0 * 5.0 + 64.0 * 3.0);
    assert_eq!(f.atom_count, 50.0);
    assert_eq!(f.clause_count, 20.0);
    assert_eq!(f.max_term_depth, 10.0);
}

#[test]
fn test_features_handles_missing_structural() {
    let meta = make_meta("QF_LIA", 500, None);
    let f = Features::from_meta(&meta);
    assert_eq!(f.atom_count, 0.0);
    assert_eq!(f.clause_count, 0.0);
    assert_eq!(f.max_term_depth, 0.0);
    assert_eq!(f.max_bv_width, 0.0);
    assert_eq!(f.total_bv_volume, 0.0);
    assert_eq!(f.max_array_dim, 0.0);
}

#[test]
fn test_features_theory_combination_score_qf_aufbv() {
    let meta = make_meta("QF_AUFBV", 1000, None);
    let f = Features::from_meta(&meta);
    // QF_AUFBV has: array, UF (implicit in some forms), BV
    // At minimum: bv + array bits should be set → score >= 2
    assert!(
        f.theory_combination_score >= 1.0,
        "QF_AUFBV should have at least 1 theory bit set"
    );
}

#[test]
fn test_feature_dim_constant() {
    let f = Features::default();
    assert_eq!(f.to_vec().len(), FEATURE_DIM);
}

#[test]
fn test_features_to_from_vec_round_trip() {
    let original = Features {
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
    let v = original.to_vec();
    assert_eq!(v.len(), FEATURE_DIM);
    let restored = Features::from_vec(&v).expect("round-trip failed");
    assert!((original.log_file_size - restored.log_file_size).abs() < 1e-12);
    assert!((original.atom_count - restored.atom_count).abs() < 1e-12);
    assert!((original.theory_combination_score - restored.theory_combination_score).abs() < 1e-12);
}

#[test]
fn test_normalize_round_trip() {
    let samples: Vec<Features> = (0..10)
        .map(|i| Features {
            atom_count: i as f64 * 10.0,
            clause_count: i as f64 * 2.0,
            log_file_size: (i as f64 + 1.0).log10(),
            ..Default::default()
        })
        .collect();

    let normalizer = FeatureNormalizer::fit(&samples);

    // Normalised values should be in [0, 1]
    for s in &samples {
        let norm = normalizer.normalize(s);
        assert_eq!(norm.len(), FEATURE_DIM);
        for v in &norm {
            assert!(
                *v >= 0.0 && *v <= 1.0 + 1e-9,
                "Normalised value out of range: {v}"
            );
        }
    }
}

#[test]
fn test_normalizer_empty_samples_gives_default() {
    let normalizer = FeatureNormalizer::fit(&[]);
    let f = Features::default();
    let norm = normalizer.normalize(&f);
    assert_eq!(norm.len(), FEATURE_DIM);
    // All zeros → all zeros in [0,1] space (min=0, max=1 by default)
    for v in &norm {
        assert!(v.is_finite());
    }
}
