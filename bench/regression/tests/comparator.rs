use bench_regression::z3_compare::{compute_ratio, detect_z3};

#[test]
fn z3_comparison_skips_gracefully_when_absent() {
    let _version = detect_z3();
}

#[test]
fn ratio_computation_correct() {
    let within_target = compute_ratio(100.0, 120.0).expect("ratio should exist");
    let exceeds_target = compute_ratio(100.0, 80.0).expect("ratio should exist");

    assert!((within_target - 0.833_333_333).abs() < 0.001);
    assert!(within_target <= 1.2);

    assert!((exceeds_target - 1.25).abs() < f64::EPSILON);
    assert!(exceeds_target > 1.2);
}
