//! Term fingerprinting for fast matching
//!
//! This module provides efficient fingerprinting (hashing) of terms for quick
//! matching and lookup. Fingerprints are used to quickly filter out non-matching
//! candidates before performing full structural matching.
//!
//! # Design
//!
//! - **Hash-based**: Uses efficient hashing to create unique fingerprints
//! - **Cached**: Fingerprints are cached to avoid recomputation
//! - **Incremental**: Supports incremental updates when terms are modified
//!
//! # Algorithm
//!
//! Based on Z3's fingerprint implementation in src/ast/ast_pp_util.cpp

use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A fingerprint of a term
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TermFingerprint(pub u64);

impl TermFingerprint {
    /// Create a new fingerprint from a hash value
    pub const fn new(hash: u64) -> Self {
        Self(hash)
    }

    /// Get the raw hash value
    pub const fn hash(&self) -> u64 {
        self.0
    }

    /// Combine two fingerprints
    pub fn combine(self, other: Self) -> Self {
        // Use a simple combining function similar to boost::hash_combine
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        other.0.hash(&mut hasher);
        TermFingerprint(hasher.finish())
    }

    /// Combine multiple fingerprints
    pub fn combine_many(fingerprints: &[Self]) -> Self {
        let mut hasher = DefaultHasher::new();
        for fp in fingerprints {
            fp.0.hash(&mut hasher);
        }
        TermFingerprint(hasher.finish())
    }
}

/// Configuration for fingerprinting
#[derive(Debug, Clone)]
pub struct FingerprintConfig {
    /// Whether to cache fingerprints
    pub enable_cache: bool,
    /// Maximum cache size (0 = unlimited)
    pub max_cache_size: usize,
    /// Whether to use structural hashing (slower but more precise)
    pub structural_hashing: bool,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            max_cache_size: 100000,
            structural_hashing: true,
        }
    }
}

/// Cache for term fingerprints
#[derive(Debug)]
pub struct FingerprintCache {
    /// Configuration
    config: FingerprintConfig,
    /// Cached fingerprints
    cache: FxHashMap<TermId, TermFingerprint>,
    /// Cache hit statistics
    hits: usize,
    /// Cache miss statistics
    misses: usize,
}

impl FingerprintCache {
    /// Create a new fingerprint cache
    pub fn new(config: FingerprintConfig) -> Self {
        Self {
            config,
            cache: FxHashMap::default(),
            hits: 0,
            misses: 0,
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(FingerprintConfig::default())
    }

    /// Compute fingerprint for a term
    pub fn compute(&mut self, term_id: TermId, manager: &TermManager) -> TermFingerprint {
        // Check cache first
        if self.config.enable_cache {
            if let Some(&fp) = self.cache.get(&term_id) {
                self.hits += 1;
                return fp;
            }
            self.misses += 1;
        }

        // Compute the fingerprint
        let fp = if self.config.structural_hashing {
            self.compute_structural(term_id, manager)
        } else {
            self.compute_simple(term_id)
        };

        // Cache it if enabled
        if self.config.enable_cache
            && (self.config.max_cache_size == 0 || self.cache.len() < self.config.max_cache_size)
        {
            self.cache.insert(term_id, fp);
        }

        fp
    }

    /// Compute a simple fingerprint based on term ID only
    fn compute_simple(&self, term_id: TermId) -> TermFingerprint {
        let mut hasher = DefaultHasher::new();
        term_id.hash(&mut hasher);
        TermFingerprint(hasher.finish())
    }

    /// Compute a structural fingerprint based on term structure
    fn compute_structural(&mut self, term_id: TermId, manager: &TermManager) -> TermFingerprint {
        let Some(term) = manager.get(term_id) else {
            return self.compute_simple(term_id);
        };

        let mut hasher = DefaultHasher::new();

        // Hash the term kind discriminant
        std::mem::discriminant(&term.kind).hash(&mut hasher);

        // Hash based on term structure
        match &term.kind {
            TermKind::Var(name) => {
                name.hash(&mut hasher);
            }
            TermKind::True => {
                1u8.hash(&mut hasher);
            }
            TermKind::False => {
                0u8.hash(&mut hasher);
            }
            TermKind::IntConst(val) => {
                val.hash(&mut hasher);
            }
            TermKind::RealConst(rational) => {
                // Hash numerator and denominator of the rational
                rational.numer().hash(&mut hasher);
                rational.denom().hash(&mut hasher);
            }
            TermKind::BitVecConst { value, width } => {
                value.hash(&mut hasher);
                width.hash(&mut hasher);
            }
            TermKind::StringLit(s) => {
                s.hash(&mut hasher);
            }
            TermKind::Apply { func, args } => {
                func.hash(&mut hasher);
                for &arg in args.iter() {
                    let arg_fp = self.compute(arg, manager);
                    arg_fp.0.hash(&mut hasher);
                }
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs)
            | TermKind::Sub(lhs, rhs)
            | TermKind::Div(lhs, rhs) => {
                let lhs_fp = self.compute(*lhs, manager);
                let rhs_fp = self.compute(*rhs, manager);
                lhs_fp.0.hash(&mut hasher);
                rhs_fp.0.hash(&mut hasher);
            }
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                for &arg in args.iter() {
                    let arg_fp = self.compute(arg, manager);
                    arg_fp.0.hash(&mut hasher);
                }
            }
            TermKind::Not(inner) | TermKind::Neg(inner) => {
                let inner_fp = self.compute(*inner, manager);
                inner_fp.0.hash(&mut hasher);
            }
            TermKind::Implies(lhs, rhs) | TermKind::Xor(lhs, rhs) => {
                let lhs_fp = self.compute(*lhs, manager);
                let rhs_fp = self.compute(*rhs, manager);
                lhs_fp.0.hash(&mut hasher);
                rhs_fp.0.hash(&mut hasher);
            }
            TermKind::Ite(c, t, e) => {
                let c_fp = self.compute(*c, manager);
                let t_fp = self.compute(*t, manager);
                let e_fp = self.compute(*e, manager);
                c_fp.0.hash(&mut hasher);
                t_fp.0.hash(&mut hasher);
                e_fp.0.hash(&mut hasher);
            }
            TermKind::Select(arr, idx) => {
                let arr_fp = self.compute(*arr, manager);
                let idx_fp = self.compute(*idx, manager);
                arr_fp.0.hash(&mut hasher);
                idx_fp.0.hash(&mut hasher);
            }
            TermKind::Store(arr, idx, val) => {
                let arr_fp = self.compute(*arr, manager);
                let idx_fp = self.compute(*idx, manager);
                let val_fp = self.compute(*val, manager);
                arr_fp.0.hash(&mut hasher);
                idx_fp.0.hash(&mut hasher);
                val_fp.0.hash(&mut hasher);
            }
            TermKind::Forall { vars, body, .. } | TermKind::Exists { vars, body, .. } => {
                // Hash number of variables
                vars.len().hash(&mut hasher);
                // Hash variable sorts
                for (_, sort) in vars.iter() {
                    sort.hash(&mut hasher);
                }
                // Hash body
                let body_fp = self.compute(*body, manager);
                body_fp.0.hash(&mut hasher);
            }
            TermKind::Mod(lhs, rhs) => {
                let lhs_fp = self.compute(*lhs, manager);
                let rhs_fp = self.compute(*rhs, manager);
                lhs_fp.0.hash(&mut hasher);
                rhs_fp.0.hash(&mut hasher);
            }
            TermKind::BvNot(inner) => {
                let inner_fp = self.compute(*inner, manager);
                inner_fp.0.hash(&mut hasher);
            }
            TermKind::BvAnd(lhs, rhs)
            | TermKind::BvOr(lhs, rhs)
            | TermKind::BvXor(lhs, rhs)
            | TermKind::BvAdd(lhs, rhs)
            | TermKind::BvSub(lhs, rhs)
            | TermKind::BvMul(lhs, rhs)
            | TermKind::BvUdiv(lhs, rhs)
            | TermKind::BvSdiv(lhs, rhs)
            | TermKind::BvUrem(lhs, rhs)
            | TermKind::BvSrem(lhs, rhs)
            | TermKind::BvShl(lhs, rhs)
            | TermKind::BvLshr(lhs, rhs)
            | TermKind::BvAshr(lhs, rhs)
            | TermKind::BvUlt(lhs, rhs)
            | TermKind::BvUle(lhs, rhs)
            | TermKind::BvSlt(lhs, rhs)
            | TermKind::BvSle(lhs, rhs)
            | TermKind::BvConcat(lhs, rhs) => {
                let lhs_fp = self.compute(*lhs, manager);
                let rhs_fp = self.compute(*rhs, manager);
                lhs_fp.0.hash(&mut hasher);
                rhs_fp.0.hash(&mut hasher);
            }
            TermKind::BvExtract { arg, high, low } => {
                let val_fp = self.compute(*arg, manager);
                val_fp.0.hash(&mut hasher);
                high.hash(&mut hasher);
                low.hash(&mut hasher);
            }
            TermKind::StrLen(inner) | TermKind::StrToInt(inner) => {
                let inner_fp = self.compute(*inner, manager);
                inner_fp.0.hash(&mut hasher);
            }
            TermKind::StrConcat(lhs, rhs)
            | TermKind::StrAt(lhs, rhs)
            | TermKind::StrContains(lhs, rhs)
            | TermKind::StrPrefixOf(lhs, rhs)
            | TermKind::StrSuffixOf(lhs, rhs)
            | TermKind::StrInRe(lhs, rhs) => {
                let lhs_fp = self.compute(*lhs, manager);
                let rhs_fp = self.compute(*rhs, manager);
                lhs_fp.0.hash(&mut hasher);
                rhs_fp.0.hash(&mut hasher);
            }
            TermKind::StrSubstr(s, start, len)
            | TermKind::StrReplace(s, start, len)
            | TermKind::StrIndexOf(s, start, len) => {
                let s_fp = self.compute(*s, manager);
                let start_fp = self.compute(*start, manager);
                let len_fp = self.compute(*len, manager);
                s_fp.0.hash(&mut hasher);
                start_fp.0.hash(&mut hasher);
                len_fp.0.hash(&mut hasher);
            }
            TermKind::StrReplaceAll(s, pattern, replacement) => {
                let s_fp = self.compute(*s, manager);
                let p_fp = self.compute(*pattern, manager);
                let r_fp = self.compute(*replacement, manager);
                s_fp.0.hash(&mut hasher);
                p_fp.0.hash(&mut hasher);
                r_fp.0.hash(&mut hasher);
            }
            TermKind::FpAbs(inner)
            | TermKind::FpNeg(inner)
            | TermKind::FpIsNormal(inner)
            | TermKind::FpIsSubnormal(inner)
            | TermKind::FpIsZero(inner)
            | TermKind::FpIsInfinite(inner)
            | TermKind::FpIsNaN(inner)
            | TermKind::FpIsNegative(inner)
            | TermKind::FpIsPositive(inner)
            | TermKind::FpToReal(inner) => {
                let inner_fp = self.compute(*inner, manager);
                inner_fp.0.hash(&mut hasher);
            }
            TermKind::FpAdd(rm, lhs, rhs)
            | TermKind::FpSub(rm, lhs, rhs)
            | TermKind::FpMul(rm, lhs, rhs)
            | TermKind::FpDiv(rm, lhs, rhs) => {
                rm.hash(&mut hasher);
                let lhs_fp = self.compute(*lhs, manager);
                let rhs_fp = self.compute(*rhs, manager);
                lhs_fp.0.hash(&mut hasher);
                rhs_fp.0.hash(&mut hasher);
            }
            TermKind::FpRem(lhs, rhs) | TermKind::FpMin(lhs, rhs) | TermKind::FpMax(lhs, rhs) => {
                let lhs_fp = self.compute(*lhs, manager);
                let rhs_fp = self.compute(*rhs, manager);
                lhs_fp.0.hash(&mut hasher);
                rhs_fp.0.hash(&mut hasher);
            }
            TermKind::FpFma(rm, a, b, c) => {
                rm.hash(&mut hasher);
                let a_fp = self.compute(*a, manager);
                let b_fp = self.compute(*b, manager);
                let c_fp = self.compute(*c, manager);
                a_fp.0.hash(&mut hasher);
                b_fp.0.hash(&mut hasher);
                c_fp.0.hash(&mut hasher);
            }
            TermKind::FpSqrt(rm, val) | TermKind::FpRoundToIntegral(rm, val) => {
                rm.hash(&mut hasher);
                let val_fp = self.compute(*val, manager);
                val_fp.0.hash(&mut hasher);
            }
            TermKind::FpEq(lhs, rhs)
            | TermKind::FpLt(lhs, rhs)
            | TermKind::FpLeq(lhs, rhs)
            | TermKind::FpGt(lhs, rhs)
            | TermKind::FpGeq(lhs, rhs) => {
                let lhs_fp = self.compute(*lhs, manager);
                let rhs_fp = self.compute(*rhs, manager);
                lhs_fp.0.hash(&mut hasher);
                rhs_fp.0.hash(&mut hasher);
            }
            TermKind::DtConstructor { constructor, args } => {
                constructor.hash(&mut hasher);
                for &arg in args.iter() {
                    let arg_fp = self.compute(arg, manager);
                    arg_fp.0.hash(&mut hasher);
                }
            }
            TermKind::DtTester { constructor, arg } => {
                constructor.hash(&mut hasher);
                let arg_fp = self.compute(*arg, manager);
                arg_fp.0.hash(&mut hasher);
            }
            TermKind::Distinct(args) => {
                for &arg in args.iter() {
                    let arg_fp = self.compute(arg, manager);
                    arg_fp.0.hash(&mut hasher);
                }
            }
            TermKind::IntToStr(inner) => {
                let inner_fp = self.compute(*inner, manager);
                inner_fp.0.hash(&mut hasher);
            }
            TermKind::FpLit { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpNaN { .. } => {
                // FP literals: hash their discriminant (fields are already in hash)
                // No child terms to recurse on
            }
            TermKind::FpToFp { rm, arg, .. }
            | TermKind::FpToSBV { rm, arg, .. }
            | TermKind::FpToUBV { rm, arg, .. }
            | TermKind::RealToFp { rm, arg, .. }
            | TermKind::SBVToFp { rm, arg, .. }
            | TermKind::UBVToFp { rm, arg, .. } => {
                rm.hash(&mut hasher);
                let arg_fp = self.compute(*arg, manager);
                arg_fp.0.hash(&mut hasher);
            }
            TermKind::DtSelector { selector, arg } => {
                selector.hash(&mut hasher);
                let arg_fp = self.compute(*arg, manager);
                arg_fp.0.hash(&mut hasher);
            }
            TermKind::Let { bindings, body } => {
                for (name, value) in bindings.iter() {
                    name.hash(&mut hasher);
                    let val_fp = self.compute(*value, manager);
                    val_fp.0.hash(&mut hasher);
                }
                let body_fp = self.compute(*body, manager);
                body_fp.0.hash(&mut hasher);
            }
            TermKind::Match { scrutinee, cases } => {
                let scrutinee_fp = self.compute(*scrutinee, manager);
                scrutinee_fp.0.hash(&mut hasher);
                // Hash case bodies (patterns would require more complex handling)
                for case in cases.iter() {
                    let body_fp = self.compute(case.body, manager);
                    body_fp.0.hash(&mut hasher);
                }
            }
        }

        TermFingerprint(hasher.finish())
    }

    /// Invalidate cached fingerprint for a term
    pub fn invalidate(&mut self, term_id: TermId) {
        self.cache.remove(&term_id);
    }

    /// Clear all cached fingerprints
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.cache.len(), self.hits, self.misses)
    }
}

/// Compute a fingerprint for a term without caching
pub fn compute_fingerprint(term_id: TermId, manager: &TermManager) -> TermFingerprint {
    let mut cache = FingerprintCache::new(FingerprintConfig {
        enable_cache: false,
        ..Default::default()
    });
    cache.compute(term_id, manager)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    fn setup() -> TermManager {
        TermManager::new()
    }

    #[test]
    fn test_fingerprint_creation() {
        let fp = TermFingerprint::new(12345);
        assert_eq!(fp.hash(), 12345);
    }

    #[test]
    fn test_fingerprint_combine() {
        let fp1 = TermFingerprint::new(111);
        let fp2 = TermFingerprint::new(222);
        let combined = fp1.combine(fp2);

        // Combined fingerprint should be different from originals
        assert_ne!(combined.hash(), fp1.hash());
        assert_ne!(combined.hash(), fp2.hash());
    }

    #[test]
    fn test_fingerprint_config_default() {
        let config = FingerprintConfig::default();
        assert!(config.enable_cache);
        assert_eq!(config.max_cache_size, 100000);
        assert!(config.structural_hashing);
    }

    #[test]
    fn test_cache_creation() {
        let cache = FingerprintCache::new_default();
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 0);
    }

    #[test]
    fn test_compute_simple_fingerprint() {
        let mut manager = setup();
        let mut cache = FingerprintCache::new(FingerprintConfig {
            enable_cache: false,
            structural_hashing: false,
            ..Default::default()
        });

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let fp = cache.compute(x, &manager);

        // Should get consistent fingerprints
        let fp2 = cache.compute(x, &manager);
        assert_eq!(fp.hash(), fp2.hash());
    }

    #[test]
    fn test_compute_structural_fingerprint() {
        let mut manager = setup();
        let mut cache = FingerprintCache::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        // Different terms should have different fingerprints
        let fp_x = cache.compute(x, &manager);
        let fp_five = cache.compute(five, &manager);
        assert_ne!(fp_x.hash(), fp_five.hash());

        // Same term should have same fingerprint
        let fp_x2 = cache.compute(x, &manager);
        assert_eq!(fp_x.hash(), fp_x2.hash());
    }

    #[test]
    fn test_fingerprint_caching() {
        let mut manager = setup();
        let mut cache = FingerprintCache::new_default();

        let x = manager.mk_var("x", manager.sorts.int_sort);

        // First access should be a miss
        cache.compute(x, &manager);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);

        // Second access should be a hit
        cache.compute(x, &manager);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let mut manager = setup();
        let mut cache = FingerprintCache::new_default();

        let x = manager.mk_var("x", manager.sorts.int_sort);

        // Compute and cache
        cache.compute(x, &manager);
        assert_eq!(cache.cache.len(), 1);

        // Invalidate
        cache.invalidate(x);
        assert_eq!(cache.cache.len(), 0);

        // Next access should be a miss again
        let old_misses = cache.misses;
        cache.compute(x, &manager);
        assert_eq!(cache.misses, old_misses + 1);
    }

    #[test]
    fn test_cache_clear() {
        let mut manager = setup();
        let mut cache = FingerprintCache::new_default();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        cache.compute(x, &manager);
        cache.compute(y, &manager);

        assert_eq!(cache.cache.len(), 2);
        assert!(cache.misses > 0);

        cache.clear();
        assert_eq!(cache.cache.len(), 0);
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 0);
    }

    #[test]
    fn test_hit_rate() {
        let mut manager = setup();
        let mut cache = FingerprintCache::new_default();

        let x = manager.mk_var("x", manager.sorts.int_sort);

        // 1 miss
        cache.compute(x, &manager);
        // 2 hits
        cache.compute(x, &manager);
        cache.compute(x, &manager);

        let hit_rate = cache.hit_rate();
        assert!((hit_rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_structural_vs_simple() {
        let mut manager = setup();

        let mut cache_structural = FingerprintCache::new(FingerprintConfig {
            structural_hashing: true,
            ..Default::default()
        });

        let mut cache_simple = FingerprintCache::new(FingerprintConfig {
            structural_hashing: false,
            ..Default::default()
        });

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);

        let fp_struct = cache_structural.compute(f_x, &manager);
        let fp_simple = cache_simple.compute(f_x, &manager);

        // They may be different due to different hashing strategies
        // (not guaranteed, but likely in practice)
        // Main point: both should be consistent
        let fp_struct2 = cache_structural.compute(f_x, &manager);
        let fp_simple2 = cache_simple.compute(f_x, &manager);

        assert_eq!(fp_struct.hash(), fp_struct2.hash());
        assert_eq!(fp_simple.hash(), fp_simple2.hash());
    }

    #[test]
    fn test_compute_fingerprint_function() {
        let mut manager = setup();
        let x = manager.mk_var("x", manager.sorts.int_sort);

        let fp1 = compute_fingerprint(x, &manager);
        let fp2 = compute_fingerprint(x, &manager);

        // Should be consistent
        assert_eq!(fp1.hash(), fp2.hash());
    }

    #[test]
    fn test_complex_term_fingerprint() {
        let mut manager = setup();
        let mut cache = FingerprintCache::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let sum = manager.mk_add([x, y]);
        let product = manager.mk_mul([x, y]);

        let fp_sum = cache.compute(sum, &manager);
        let fp_product = cache.compute(product, &manager);

        // Different operations should have different fingerprints
        assert_ne!(fp_sum.hash(), fp_product.hash());
    }

    #[test]
    fn test_max_cache_size() {
        let mut manager = setup();
        let mut cache = FingerprintCache::new(FingerprintConfig {
            enable_cache: true,
            max_cache_size: 2,
            ..Default::default()
        });

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let z = manager.mk_var("z", int_sort);

        cache.compute(x, &manager);
        cache.compute(y, &manager);
        assert_eq!(cache.cache.len(), 2);

        // Third term should not be cached (at limit)
        cache.compute(z, &manager);
        assert_eq!(cache.cache.len(), 2);
    }
}
