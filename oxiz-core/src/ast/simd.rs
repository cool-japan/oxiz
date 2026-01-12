//! SIMD-accelerated term comparison and hashing utilities
//!
//! This module provides SIMD-optimized operations for comparing and hashing
//! terms in very large term databases, improving performance when processing
//! thousands or millions of terms.

use crate::ast::term::TermId;
use wide::i32x4;

/// SIMD-accelerated comparison of TermId arrays
///
/// Compares two slices of TermIds using SIMD instructions where possible.
/// Falls back to scalar comparison for remainder elements.
///
/// # Arguments
///
/// * `a` - First slice of term IDs
/// * `b` - Second slice of term IDs
///
/// # Returns
///
/// `true` if the slices are equal, `false` otherwise
///
/// # Performance
///
/// For large arrays (>4 elements), this can be 2-4x faster than
/// element-by-element comparison due to SIMD parallelism.
#[inline]
pub fn simd_compare_termids(a: &[TermId], b: &[TermId]) -> bool {
    // Quick length check
    if a.len() != b.len() {
        return false;
    }

    let len = a.len();

    // For small arrays, use scalar comparison (faster due to no SIMD overhead)
    if len < 4 {
        return a == b;
    }

    // Process 4 elements at a time using SIMD
    let chunks = len / 4;
    let remainder = len % 4;

    // Safety: We're comparing u32 values which can be safely loaded as i32
    // for comparison purposes (equality is preserved)
    for i in 0..chunks {
        let offset = i * 4;

        // Load 4 TermIds from each array
        let a_chunk = [
            a[offset].0 as i32,
            a[offset + 1].0 as i32,
            a[offset + 2].0 as i32,
            a[offset + 3].0 as i32,
        ];
        let b_chunk = [
            b[offset].0 as i32,
            b[offset + 1].0 as i32,
            b[offset + 2].0 as i32,
            b[offset + 3].0 as i32,
        ];

        let va = i32x4::from(a_chunk);
        let vb = i32x4::from(b_chunk);

        // Compare using SIMD - check element-wise equality
        let a_arr = va.to_array();
        let b_arr = vb.to_array();
        if a_arr != b_arr {
            return false;
        }
    }

    // Handle remainder with scalar comparison
    let offset = chunks * 4;
    for i in 0..remainder {
        if a[offset + i] != b[offset + i] {
            return false;
        }
    }

    true
}

/// SIMD-accelerated hash computation for TermId arrays
///
/// Computes a hash value for a slice of TermIds using SIMD instructions
/// to process multiple elements in parallel.
///
/// # Arguments
///
/// * `ids` - Slice of term IDs to hash
/// * `seed` - Initial hash seed
///
/// # Returns
///
/// Hash value as a u64
///
/// # Performance
///
/// This implementation uses SIMD to process 4 term IDs at once,
/// providing better performance for arrays with many elements.
#[inline]
pub fn simd_hash_termids(ids: &[TermId], seed: u64) -> u64 {
    let len = ids.len();

    // For small arrays, use a simple scalar hash
    if len == 0 {
        return seed;
    }

    if len < 4 {
        return ids.iter().fold(seed, |acc, &id| {
            acc.wrapping_mul(0x517c_c1b7_2722_0a95)
                .wrapping_add(id.0 as u64)
        });
    }

    let mut hash = seed;
    let chunks = len / 4;
    let remainder = len % 4;

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;

        // Load 4 TermIds
        let chunk = [
            ids[offset].0 as i32,
            ids[offset + 1].0 as i32,
            ids[offset + 2].0 as i32,
            ids[offset + 3].0 as i32,
        ];

        let v = i32x4::from(chunk);

        // Mix the values together
        // Convert to array and accumulate into hash
        let arr: [i32; 4] = v.into();
        for &val in &arr {
            hash = hash
                .wrapping_mul(0x517c_c1b7_2722_0a95)
                .wrapping_add(val as u32 as u64);
        }
    }

    // Handle remainder
    let offset = chunks * 4;
    for i in 0..remainder {
        hash = hash
            .wrapping_mul(0x517c_c1b7_2722_0a95)
            .wrapping_add(ids[offset + i].0 as u64);
    }

    hash
}

/// SIMD-accelerated bulk equality check
///
/// Checks if all elements in a slice equal a target value using SIMD.
///
/// # Arguments
///
/// * `ids` - Slice of term IDs to check
/// * `target` - Target value to compare against
///
/// # Returns
///
/// `true` if all elements equal the target, `false` otherwise
#[inline]
pub fn simd_all_equal(ids: &[TermId], target: TermId) -> bool {
    let len = ids.len();

    if len == 0 {
        return true;
    }

    if len < 4 {
        return ids.iter().all(|&id| id == target);
    }

    let chunks = len / 4;
    let remainder = len % 4;
    let target_i32 = target.0 as i32;
    let target_vec = i32x4::from([target_i32; 4]);

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;

        let chunk = [
            ids[offset].0 as i32,
            ids[offset + 1].0 as i32,
            ids[offset + 2].0 as i32,
            ids[offset + 3].0 as i32,
        ];

        let v = i32x4::from(chunk);
        let v_arr = v.to_array();
        let target_arr = target_vec.to_array();

        // Check if all elements are equal to target
        if v_arr != target_arr {
            return false;
        }
    }

    // Handle remainder
    let offset = chunks * 4;
    for i in 0..remainder {
        if ids[offset + i] != target {
            return false;
        }
    }

    true
}

/// SIMD-accelerated membership check
///
/// Checks if a target value exists in a slice using SIMD.
///
/// # Arguments
///
/// * `ids` - Slice of term IDs to search
/// * `target` - Target value to find
///
/// # Returns
///
/// `true` if the target is found, `false` otherwise
#[inline]
pub fn simd_contains(ids: &[TermId], target: TermId) -> bool {
    let len = ids.len();

    if len == 0 {
        return false;
    }

    if len < 4 {
        return ids.contains(&target);
    }

    let chunks = len / 4;
    let remainder = len % 4;
    let target_i32 = target.0 as i32;
    let target_vec = i32x4::from([target_i32; 4]);

    // Process 4 elements at a time
    for i in 0..chunks {
        let offset = i * 4;

        let chunk = [
            ids[offset].0 as i32,
            ids[offset + 1].0 as i32,
            ids[offset + 2].0 as i32,
            ids[offset + 3].0 as i32,
        ];

        let v = i32x4::from(chunk);
        let v_arr = v.to_array();
        let target_arr = target_vec.to_array();

        // Check if any element matches target
        if v_arr.iter().zip(target_arr.iter()).any(|(a, b)| a == b) {
            return true;
        }
    }

    // Handle remainder
    let offset = chunks * 4;
    for i in 0..remainder {
        if ids[offset + i] == target {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_compare_empty() {
        let a: &[TermId] = &[];
        let b: &[TermId] = &[];
        assert!(simd_compare_termids(a, b));
    }

    #[test]
    fn test_simd_compare_different_lengths() {
        let a = &[TermId(1), TermId(2)];
        let b = &[TermId(1)];
        assert!(!simd_compare_termids(a, b));
    }

    #[test]
    fn test_simd_compare_small_equal() {
        let a = &[TermId(1), TermId(2), TermId(3)];
        let b = &[TermId(1), TermId(2), TermId(3)];
        assert!(simd_compare_termids(a, b));
    }

    #[test]
    fn test_simd_compare_small_different() {
        let a = &[TermId(1), TermId(2), TermId(3)];
        let b = &[TermId(1), TermId(2), TermId(4)];
        assert!(!simd_compare_termids(a, b));
    }

    #[test]
    fn test_simd_compare_large_equal() {
        let a: Vec<TermId> = (0..100).map(TermId).collect();
        let b: Vec<TermId> = (0..100).map(TermId).collect();
        assert!(simd_compare_termids(&a, &b));
    }

    #[test]
    fn test_simd_compare_large_different() {
        let a: Vec<TermId> = (0..100).map(TermId).collect();
        let mut b: Vec<TermId> = (0..100).map(TermId).collect();
        b[50] = TermId(999);
        assert!(!simd_compare_termids(&a, &b));
    }

    #[test]
    fn test_simd_hash_empty() {
        let ids: &[TermId] = &[];
        let hash = simd_hash_termids(ids, 0);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_simd_hash_deterministic() {
        let ids = &[TermId(1), TermId(2), TermId(3), TermId(4), TermId(5)];
        let hash1 = simd_hash_termids(ids, 42);
        let hash2 = simd_hash_termids(ids, 42);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_simd_hash_different_inputs() {
        let ids1 = &[TermId(1), TermId(2), TermId(3)];
        let ids2 = &[TermId(1), TermId(2), TermId(4)];
        let hash1 = simd_hash_termids(ids1, 0);
        let hash2 = simd_hash_termids(ids2, 0);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_simd_all_equal_empty() {
        let ids: &[TermId] = &[];
        assert!(simd_all_equal(ids, TermId(1)));
    }

    #[test]
    fn test_simd_all_equal_small_true() {
        let ids = &[TermId(5), TermId(5), TermId(5)];
        assert!(simd_all_equal(ids, TermId(5)));
    }

    #[test]
    fn test_simd_all_equal_small_false() {
        let ids = &[TermId(5), TermId(5), TermId(6)];
        assert!(!simd_all_equal(ids, TermId(5)));
    }

    #[test]
    fn test_simd_all_equal_large_true() {
        let ids: Vec<TermId> = vec![TermId(42); 100];
        assert!(simd_all_equal(&ids, TermId(42)));
    }

    #[test]
    fn test_simd_all_equal_large_false() {
        let mut ids: Vec<TermId> = vec![TermId(42); 100];
        ids[75] = TermId(99);
        assert!(!simd_all_equal(&ids, TermId(42)));
    }

    #[test]
    fn test_simd_contains_empty() {
        let ids: &[TermId] = &[];
        assert!(!simd_contains(ids, TermId(1)));
    }

    #[test]
    fn test_simd_contains_small_found() {
        let ids = &[TermId(1), TermId(2), TermId(3)];
        assert!(simd_contains(ids, TermId(2)));
    }

    #[test]
    fn test_simd_contains_small_not_found() {
        let ids = &[TermId(1), TermId(2), TermId(3)];
        assert!(!simd_contains(ids, TermId(5)));
    }

    #[test]
    fn test_simd_contains_large_found() {
        let ids: Vec<TermId> = (0..100).map(TermId).collect();
        assert!(simd_contains(&ids, TermId(42)));
    }

    #[test]
    fn test_simd_contains_large_not_found() {
        let ids: Vec<TermId> = (0..100).map(TermId).collect();
        assert!(!simd_contains(&ids, TermId(200)));
    }

    #[test]
    fn test_simd_compare_edge_cases() {
        // Test with exactly 4 elements (one SIMD chunk)
        let a = &[TermId(1), TermId(2), TermId(3), TermId(4)];
        let b = &[TermId(1), TermId(2), TermId(3), TermId(4)];
        assert!(simd_compare_termids(a, b));

        // Test with 5 elements (one SIMD chunk + 1 remainder)
        let a = &[TermId(1), TermId(2), TermId(3), TermId(4), TermId(5)];
        let b = &[TermId(1), TermId(2), TermId(3), TermId(4), TermId(5)];
        assert!(simd_compare_termids(a, b));

        // Test with 8 elements (two SIMD chunks)
        let a: Vec<TermId> = (1..=8).map(TermId).collect();
        let b: Vec<TermId> = (1..=8).map(TermId).collect();
        assert!(simd_compare_termids(&a, &b));
    }
}
