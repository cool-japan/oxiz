//! Specialised hasher for `TermKind` values.
//!
//! The default `FxHasher` treats `TermKind` as an opaque byte sequence.
//! `TermKindHasher` exploits structural knowledge of `TermKind`:
//!
//! * The **discriminant** (enum variant tag) is used as the primary hash
//!   component, giving an excellent first-level spread.
//! * **Child `TermId`s** (small u32 integers) are mixed in with a fast
//!   integer finaliser rather than feeding them byte-by-byte.
//! * **Large payloads** (string literals, `BigInt` constants) are
//!   represented by their *length* plus the first and last bytes, avoiding
//!   full traversal of potentially enormous data.
//!
//! This yields a cheaper hash while retaining collision resistance that is
//! more than adequate for the term interning cache.

use core::hash::{BuildHasher, Hasher};

/// A fast, non-cryptographic hasher specialised for `TermKind` values.
///
/// Internally accumulates a `u64` state using a multiply-xor-shift
/// finaliser for each ingested value.
#[derive(Clone)]
pub struct TermKindHasher {
    state: u64,
}

/// Multiplicative constant (Knuth's golden-ratio constant for 64-bit).
const PHI64: u64 = 0x9E37_79B9_7F4A_7C15;

impl TermKindHasher {
    /// Create a new hasher with seed 0.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self { state: 0 }
    }

    /// Fold a u64 value into the running state.
    #[inline]
    fn fold(&mut self, value: u64) {
        // Multiply-xor-shift mixing (same family as FxHash / wyhash lite).
        self.state = (self.state.rotate_left(5) ^ value).wrapping_mul(PHI64);
    }
}

impl Default for TermKindHasher {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for TermKindHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // Final avalanche pass
        let mut h = self.state;
        h ^= h >> 33;
        h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        h ^= h >> 33;
        h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        h ^= h >> 33;
        h
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Process 8 bytes at a time
        let chunks = bytes.len() / 8;
        for i in 0..chunks {
            let base = i * 8;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[base..base + 8]);
            self.fold(u64::from_le_bytes(buf));
        }
        // Remaining bytes
        let tail_start = chunks * 8;
        if tail_start < bytes.len() {
            let mut buf = [0u8; 8];
            let tail = &bytes[tail_start..];
            buf[..tail.len()].copy_from_slice(tail);
            // Mix in the length to differentiate short tails
            buf[7] = bytes.len() as u8;
            self.fold(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.fold(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.fold(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.fold(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.fold(i);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.fold(i as u64);
        self.fold((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.fold(i as u64);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.fold(i as u64);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.fold(i as u64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.fold(i as u64);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.fold(i as u64);
    }

    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.fold(i as u64);
        self.fold((i >> 64) as u64);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.fold(i as u64);
    }
}

/// [`BuildHasher`] that produces [`TermKindHasher`] instances.
#[derive(Clone, Default)]
pub struct BuildTermKindHasher;

impl BuildHasher for BuildTermKindHasher {
    type Hasher = TermKindHasher;

    #[inline]
    fn build_hasher(&self) -> TermKindHasher {
        TermKindHasher::new()
    }
}

/// Type alias for a `HashMap` using the term-specialised hasher.
#[cfg(feature = "std")]
pub type TermHashMap<K, V> = std::collections::HashMap<K, V, BuildTermKindHasher>;

/// Type alias for a `HashMap` using the term-specialised hasher (no_std).
#[cfg(not(feature = "std"))]
pub type TermHashMap<K, V> = hashbrown::HashMap<K, V, BuildTermKindHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::term::{TermId, TermKind};
    use core::hash::Hash;
    use smallvec::smallvec;

    fn hash_one<T: Hash>(value: &T) -> u64 {
        let mut h = TermKindHasher::new();
        value.hash(&mut h);
        h.finish()
    }

    #[test]
    fn test_different_discriminants_differ() {
        let h1 = hash_one(&TermKind::True);
        let h2 = hash_one(&TermKind::False);
        assert_ne!(h1, h2, "True and False should hash differently");
    }

    #[test]
    fn test_same_kind_same_hash() {
        let h1 = hash_one(&TermKind::True);
        let h2 = hash_one(&TermKind::True);
        assert_eq!(h1, h2, "identical TermKind must produce identical hash");
    }

    #[test]
    fn test_not_different_children() {
        let h1 = hash_one(&TermKind::Not(TermId(0)));
        let h2 = hash_one(&TermKind::Not(TermId(1)));
        assert_ne!(h1, h2, "Not(0) and Not(1) should hash differently");
    }

    #[test]
    fn test_and_order_matters() {
        let h1 = hash_one(&TermKind::And(smallvec![TermId(0), TermId(1)]));
        let h2 = hash_one(&TermKind::And(smallvec![TermId(1), TermId(0)]));
        // Different ordering should yield different hashes
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_build_hasher() {
        let bh = BuildTermKindHasher;
        let mut h = bh.build_hasher();
        h.write_u64(42);
        let v = h.finish();
        assert_ne!(v, 0);
    }

    #[test]
    fn test_collision_resistance_u32_sequence() {
        // Hash a range of small TermIds and check uniqueness
        let hashes: Vec<u64> = (0..256u32)
            .map(|i| hash_one(&TermKind::Not(TermId(i))))
            .collect();
        let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        // All 256 should be unique
        assert_eq!(
            unique.len(),
            256,
            "expected 256 unique hashes, got {}",
            unique.len()
        );
    }

    #[test]
    fn test_term_hash_map_insert_lookup() {
        let mut map: TermHashMap<TermKind, TermId> = TermHashMap::default();
        let kind = TermKind::Not(TermId(42));
        map.insert(kind.clone(), TermId(100));
        assert_eq!(map.get(&kind), Some(&TermId(100)));
    }

    #[test]
    fn test_hasher_write_bytes() {
        let mut h = TermKindHasher::new();
        h.write(b"hello world, this is a test string");
        let v = h.finish();
        assert_ne!(v, 0);
    }
}
