//! SIMD-Accelerated BitVector Word Operations.
//!
//! Provides explicitly unrolled (4×) word-level bitwise operations on packed
//! `&[u64]` slices.  The unrolled loops are intentionally structured so that
//! LLVM's auto-vectoriser can emit SSE2/AVX2/NEON instructions depending on
//! the target CPU — no nightly `std::simd` or unsafe code required.
//!
//! These functions are the hot path for constraint propagation when bit-vector
//! widths exceed 64 bits (i.e., when the bitvector is stored as multiple u64
//! words rather than a single integer).
//!
//! # Design
//!
//! Every loop is written as a four-way unrolled body followed by a scalar
//! remainder.  LLVM reliably vectorises these patterns with `opt-level = 3`.
//!
//! # Example
//! ```rust
//! use oxiz_theories::bv::simd_ops::{bv_and_words, bv_or_words, bv_count_ones};
//!
//! let a = [0xFFFF_FFFF_FFFF_FFFFu64; 4];
//! let b = [0x0F0F_0F0F_0F0F_0F0Fu64; 4];
//! let mut out = [0u64; 4];
//!
//! bv_and_words(&a, &b, &mut out);
//! assert_eq!(out, b);
//!
//! let popcount = bv_count_ones(&a);
//! assert_eq!(popcount, 256);
//! ```

/// Compute element-wise bitwise AND of two packed u64 word slices.
///
/// `a`, `b`, and `out` must all have the same length.  The loop is 4×
/// unrolled so LLVM auto-vectorises to 256-bit AVX2 or 128-bit SSE2 ops.
///
/// # Panics
/// Panics in debug mode if lengths differ.
#[inline]
pub fn bv_and_words(a: &[u64], b: &[u64], out: &mut [u64]) {
    debug_assert_eq!(a.len(), b.len(), "bv_and_words: slice length mismatch");
    debug_assert_eq!(
        a.len(),
        out.len(),
        "bv_and_words: output slice length mismatch"
    );

    let n = a.len().min(b.len()).min(out.len());
    let chunks = n / 4;
    let remainder = n % 4;

    // 4× unrolled body — LLVM fuses these into a single vectorised store
    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] & b[base];
        out[base + 1] = a[base + 1] & b[base + 1];
        out[base + 2] = a[base + 2] & b[base + 2];
        out[base + 3] = a[base + 3] & b[base + 3];
    }

    // Scalar remainder
    let base = chunks * 4;
    for j in 0..remainder {
        out[base + j] = a[base + j] & b[base + j];
    }
}

/// Compute element-wise bitwise OR of two packed u64 word slices.
///
/// Same length and unrolling contract as [`bv_and_words`].
#[inline]
pub fn bv_or_words(a: &[u64], b: &[u64], out: &mut [u64]) {
    debug_assert_eq!(a.len(), b.len(), "bv_or_words: slice length mismatch");
    debug_assert_eq!(
        a.len(),
        out.len(),
        "bv_or_words: output slice length mismatch"
    );

    let n = a.len().min(b.len()).min(out.len());
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] | b[base];
        out[base + 1] = a[base + 1] | b[base + 1];
        out[base + 2] = a[base + 2] | b[base + 2];
        out[base + 3] = a[base + 3] | b[base + 3];
    }

    let base = chunks * 4;
    for j in 0..remainder {
        out[base + j] = a[base + j] | b[base + j];
    }
}

/// Compute element-wise bitwise XOR of two packed u64 word slices.
///
/// Same length and unrolling contract as [`bv_and_words`].
#[inline]
pub fn bv_xor_words(a: &[u64], b: &[u64], out: &mut [u64]) {
    debug_assert_eq!(a.len(), b.len(), "bv_xor_words: slice length mismatch");
    debug_assert_eq!(
        a.len(),
        out.len(),
        "bv_xor_words: output slice length mismatch"
    );

    let n = a.len().min(b.len()).min(out.len());
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] ^ b[base];
        out[base + 1] = a[base + 1] ^ b[base + 1];
        out[base + 2] = a[base + 2] ^ b[base + 2];
        out[base + 3] = a[base + 3] ^ b[base + 3];
    }

    let base = chunks * 4;
    for j in 0..remainder {
        out[base + j] = a[base + j] ^ b[base + j];
    }
}

/// Compute element-wise bitwise NOT (complement) of a packed u64 word slice.
///
/// `a` and `out` must have the same length.
#[inline]
pub fn bv_not_words(a: &[u64], out: &mut [u64]) {
    debug_assert_eq!(
        a.len(),
        out.len(),
        "bv_not_words: output slice length mismatch"
    );

    let n = a.len().min(out.len());
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = !a[base];
        out[base + 1] = !a[base + 1];
        out[base + 2] = !a[base + 2];
        out[base + 3] = !a[base + 3];
    }

    let base = chunks * 4;
    for j in 0..remainder {
        out[base + j] = !a[base + j];
    }
}

/// Count the total number of set bits (population count) across all words.
///
/// This is equivalent to Hamming weight of the entire multi-word bit-vector.
/// The loop is 4× unrolled so the CPU can execute multiple `popcnt`
/// instructions in parallel on super-scalar pipelines.
#[inline]
pub fn bv_count_ones(a: &[u64]) -> u32 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut count: u32 = 0;

    for i in 0..chunks {
        let base = i * 4;
        // These four statements are independent — the CPU executes them
        // simultaneously on ports that support popcnt.
        count += a[base].count_ones();
        count += a[base + 1].count_ones();
        count += a[base + 2].count_ones();
        count += a[base + 3].count_ones();
    }

    let base = chunks * 4;
    for j in 0..remainder {
        count += a[base + j].count_ones();
    }

    count
}

/// Test whether two word-slices are equal in constant time (all words equal).
///
/// Returns `true` if every word at index `i` is identical.
#[inline]
pub fn bv_eq_words(a: &[u64], b: &[u64]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Accumulate differences — any non-zero difference means inequality.
    let mut diff: u64 = 0;

    for i in 0..chunks {
        let base = i * 4;
        diff |= a[base] ^ b[base];
        diff |= a[base + 1] ^ b[base + 1];
        diff |= a[base + 2] ^ b[base + 2];
        diff |= a[base + 3] ^ b[base + 3];
    }

    let base = chunks * 4;
    for j in 0..remainder {
        diff |= a[base + j] ^ b[base + j];
    }

    diff == 0
}

/// Compute element-wise bitwise AND-NOT: `out[i] = a[i] & !b[i]`.
///
/// Useful in BV constraint propagation for clearing bits.
#[inline]
pub fn bv_and_not_words(a: &[u64], b: &[u64], out: &mut [u64]) {
    debug_assert_eq!(a.len(), b.len(), "bv_and_not_words: slice length mismatch");
    debug_assert_eq!(
        a.len(),
        out.len(),
        "bv_and_not_words: output slice length mismatch"
    );

    let n = a.len().min(b.len()).min(out.len());
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] & !b[base];
        out[base + 1] = a[base + 1] & !b[base + 1];
        out[base + 2] = a[base + 2] & !b[base + 2];
        out[base + 3] = a[base + 3] & !b[base + 3];
    }

    let base = chunks * 4;
    for j in 0..remainder {
        out[base + j] = a[base + j] & !b[base + j];
    }
}

/// Check whether all bits in a word slice are zero.
///
/// Returns `true` if every word is zero.
#[inline]
pub fn bv_is_zero(a: &[u64]) -> bool {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc: u64 = 0;

    for i in 0..chunks {
        let base = i * 4;
        acc |= a[base];
        acc |= a[base + 1];
        acc |= a[base + 2];
        acc |= a[base + 3];
    }

    let base = chunks * 4;
    for j in 0..remainder {
        acc |= a[base + j];
    }

    acc == 0
}

/// Compute element-wise bitwise OR-NOT: `out[i] = a[i] | !b[i]`.
///
/// Useful in BV constraint propagation for setting bits.
#[inline]
pub fn bv_or_not_words(a: &[u64], b: &[u64], out: &mut [u64]) {
    debug_assert_eq!(a.len(), b.len(), "bv_or_not_words: slice length mismatch");
    debug_assert_eq!(
        a.len(),
        out.len(),
        "bv_or_not_words: output slice length mismatch"
    );

    let n = a.len().min(b.len()).min(out.len());
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] | !b[base];
        out[base + 1] = a[base + 1] | !b[base + 1];
        out[base + 2] = a[base + 2] | !b[base + 2];
        out[base + 3] = a[base + 3] | !b[base + 3];
    }

    let base = chunks * 4;
    for j in 0..remainder {
        out[base + j] = a[base + j] | !b[base + j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_words(val: u64, n: usize) -> Vec<u64> {
        vec![val; n]
    }

    #[test]
    fn test_bv_and_words_basic() {
        let a = make_words(0xFF00_FF00_FF00_FF00, 4);
        let b = make_words(0x0F0F_0F0F_0F0F_0F0F, 4);
        let mut out = vec![0u64; 4];
        bv_and_words(&a, &b, &mut out);
        assert!(out.iter().all(|&w| w == 0x0F00_0F00_0F00_0F00));
    }

    #[test]
    fn test_bv_or_words_basic() {
        let a = make_words(0xFF00_FF00_FF00_FF00, 4);
        let b = make_words(0x0F0F_0F0F_0F0F_0F0F, 4);
        let mut out = vec![0u64; 4];
        bv_or_words(&a, &b, &mut out);
        assert!(out.iter().all(|&w| w == 0xFF0F_FF0F_FF0F_FF0F));
    }

    #[test]
    fn test_bv_xor_words_basic() {
        let a = make_words(0xAAAA_AAAA_AAAA_AAAA, 4);
        let b = make_words(0x5555_5555_5555_5555, 4);
        let mut out = vec![0u64; 4];
        bv_xor_words(&a, &b, &mut out);
        assert!(out.iter().all(|&w| w == u64::MAX));
    }

    #[test]
    fn test_bv_not_words_basic() {
        let a = make_words(0x0000_0000_0000_0000, 4);
        let mut out = vec![0u64; 4];
        bv_not_words(&a, &mut out);
        assert!(out.iter().all(|&w| w == u64::MAX));
    }

    #[test]
    fn test_bv_count_ones_all_set() {
        let a = make_words(u64::MAX, 4); // 4 * 64 = 256 bits set
        assert_eq!(bv_count_ones(&a), 256);
    }

    #[test]
    fn test_bv_count_ones_empty() {
        assert_eq!(bv_count_ones(&[]), 0);
    }

    #[test]
    fn test_bv_count_ones_partial() {
        // 3 words (not a multiple of 4) to exercise the remainder path
        let a = vec![u64::MAX, 0x0F0F_0F0F_0F0F_0F0F, 0x0000_0000_FFFF_FFFF];
        let expected = 64u32 + 32u32 + 32u32;
        assert_eq!(bv_count_ones(&a), expected);
    }

    #[test]
    fn test_bv_eq_words_equal() {
        let a = make_words(0xDEAD_BEEF_CAFE_BABE, 8);
        let b = make_words(0xDEAD_BEEF_CAFE_BABE, 8);
        assert!(bv_eq_words(&a, &b));
    }

    #[test]
    fn test_bv_eq_words_not_equal() {
        let a = make_words(0xDEAD_BEEF_CAFE_BABE, 8);
        let mut b = make_words(0xDEAD_BEEF_CAFE_BABE, 8);
        b[5] = 0xFFFF_FFFF_FFFF_FFFF;
        assert!(!bv_eq_words(&a, &b));
    }

    #[test]
    fn test_bv_and_not_words() {
        // a & !b: clear bits in a that are set in b
        let a = vec![0xFF; 1];
        let b = vec![0x0F; 1];
        let mut out = vec![0u64; 1];
        bv_and_not_words(&a, &b, &mut out);
        // 0xFF & !0x0F = 0xFF & 0xF0 = 0xF0
        assert_eq!(out[0], 0xF0);
    }

    #[test]
    fn test_bv_is_zero() {
        let a = vec![0u64; 5];
        assert!(bv_is_zero(&a));
        let mut b = vec![0u64; 5];
        b[4] = 1;
        assert!(!bv_is_zero(&b));
    }

    #[test]
    fn test_remainder_path_single_word() {
        // n = 1 → chunks = 0, remainder = 1: exercises only the scalar path
        let a = [0xABCD_EF01_2345_6789u64];
        let b = [0x1111_1111_1111_1111u64];
        let mut out = [0u64; 1];
        bv_and_words(&a, &b, &mut out);
        assert_eq!(out[0], a[0] & b[0]);
    }

    #[test]
    fn test_remainder_path_five_words() {
        // n = 5 → chunks = 1, remainder = 1
        let a = vec![u64::MAX; 5];
        let b = vec![0x5555_5555_5555_5555u64; 5];
        let mut out = vec![0u64; 5];
        bv_xor_words(&a, &b, &mut out);
        assert!(out.iter().all(|&w| w == 0xAAAA_AAAA_AAAA_AAAA));
    }
}
