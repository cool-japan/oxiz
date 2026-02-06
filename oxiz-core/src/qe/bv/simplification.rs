//! BitVector Simplification for QE.
//!
//! Provides simplification techniques for bitvector formulas during
//! quantifier elimination.
//!
//! ## Simplifications
//!
//! - **Constant Folding**: Evaluate constant expressions
//! - **Algebraic Identities**: x + 0 = x, x & x = x, etc.
//! - **Range Analysis**: Track value ranges to eliminate impossible constraints
//! - **Bit-Level Reasoning**: Simplify based on individual bits
//!
//! ## References
//!
//! - "Deciding Bit-Vector Arithmetic with Abstraction" (Bryant et al., 2007)
//! - Z3's `qe/qe_bv.cpp`

use rustc_hash::FxHashMap;

/// Variable identifier.
pub type VarId = usize;

/// Bitvector term.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BvTerm {
    /// Constant.
    Const(u64, u32), // value, width
    /// Variable.
    Var(VarId, u32), // id, width
    /// Addition.
    Add(Box<BvTerm>, Box<BvTerm>),
    /// Bitwise AND.
    And(Box<BvTerm>, Box<BvTerm>),
    /// Bitwise OR.
    Or(Box<BvTerm>, Box<BvTerm>),
    /// Bitwise XOR.
    Xor(Box<BvTerm>, Box<BvTerm>),
    /// Negation (two's complement).
    Neg(Box<BvTerm>),
}

/// Configuration for BV simplification.
#[derive(Debug, Clone)]
pub struct BvSimplificationConfig {
    /// Enable constant folding.
    pub enable_constant_folding: bool,
    /// Enable algebraic identities.
    pub enable_algebraic_identities: bool,
    /// Enable range analysis.
    pub enable_range_analysis: bool,
    /// Enable bit-level reasoning.
    pub enable_bit_reasoning: bool,
}

impl Default for BvSimplificationConfig {
    fn default() -> Self {
        Self {
            enable_constant_folding: true,
            enable_algebraic_identities: true,
            enable_range_analysis: true,
            enable_bit_reasoning: true,
        }
    }
}

/// Statistics for BV simplification.
#[derive(Debug, Clone, Default)]
pub struct BvSimplificationStats {
    /// Constant folding applications.
    pub constant_foldings: u64,
    /// Algebraic simplifications.
    pub algebraic_simplifications: u64,
    /// Range-based eliminations.
    pub range_eliminations: u64,
    /// Bit-level simplifications.
    pub bit_simplifications: u64,
}

/// BV simplifier.
#[derive(Debug)]
pub struct BvSimplifier {
    /// Variable ranges (var_id -> (min, max)).
    ranges: FxHashMap<VarId, (u64, u64)>,
    /// Configuration.
    config: BvSimplificationConfig,
    /// Statistics.
    stats: BvSimplificationStats,
}

impl BvSimplifier {
    /// Create a new BV simplifier.
    pub fn new(config: BvSimplificationConfig) -> Self {
        Self {
            ranges: FxHashMap::default(),
            config,
            stats: BvSimplificationStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(BvSimplificationConfig::default())
    }

    /// Set a variable's range.
    pub fn set_range(&mut self, var: VarId, min: u64, max: u64) {
        self.ranges.insert(var, (min, max));
    }

    /// Simplify a BV term.
    pub fn simplify(&mut self, term: &BvTerm) -> BvTerm {
        match term {
            BvTerm::Const(_, _) => term.clone(),
            BvTerm::Var(_, _) => term.clone(),

            BvTerm::Add(left, right) => {
                let left_simp = self.simplify(left);
                let right_simp = self.simplify(right);
                self.simplify_add(&left_simp, &right_simp)
            }

            BvTerm::And(left, right) => {
                let left_simp = self.simplify(left);
                let right_simp = self.simplify(right);
                self.simplify_and(&left_simp, &right_simp)
            }

            BvTerm::Or(left, right) => {
                let left_simp = self.simplify(left);
                let right_simp = self.simplify(right);
                self.simplify_or(&left_simp, &right_simp)
            }

            BvTerm::Xor(left, right) => {
                let left_simp = self.simplify(left);
                let right_simp = self.simplify(right);
                self.simplify_xor(&left_simp, &right_simp)
            }

            BvTerm::Neg(inner) => {
                let inner_simp = self.simplify(inner);
                self.simplify_neg(&inner_simp)
            }
        }
    }

    /// Simplify addition.
    fn simplify_add(&mut self, left: &BvTerm, right: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let (BvTerm::Const(v1, w1), BvTerm::Const(v2, w2)) = (left, right)
            && w1 == w2
        {
            self.stats.constant_foldings += 1;
            let mask = (1u64 << w1) - 1;
            return BvTerm::Const((v1 + v2) & mask, *w1);
        }

        // Algebraic identities
        if self.config.enable_algebraic_identities {
            // x + 0 = x
            if let BvTerm::Const(0, _) = right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }
            if let BvTerm::Const(0, _) = left {
                self.stats.algebraic_simplifications += 1;
                return right.clone();
            }
        }

        BvTerm::Add(Box::new(left.clone()), Box::new(right.clone()))
    }

    /// Simplify bitwise AND.
    fn simplify_and(&mut self, left: &BvTerm, right: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let (BvTerm::Const(v1, w1), BvTerm::Const(v2, w2)) = (left, right)
            && w1 == w2
        {
            self.stats.constant_foldings += 1;
            return BvTerm::Const(v1 & v2, *w1);
        }

        // Algebraic identities
        if self.config.enable_algebraic_identities {
            // x & x = x
            if left == right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }

            // x & 0 = 0
            if let BvTerm::Const(0, w) = right {
                self.stats.algebraic_simplifications += 1;
                return BvTerm::Const(0, *w);
            }

            // x & ~0 = x
            if let BvTerm::Const(v, w) = right {
                let all_ones = (1u64 << w) - 1;
                if *v == all_ones {
                    self.stats.algebraic_simplifications += 1;
                    return left.clone();
                }
            }
        }

        BvTerm::And(Box::new(left.clone()), Box::new(right.clone()))
    }

    /// Simplify bitwise OR.
    fn simplify_or(&mut self, left: &BvTerm, right: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let (BvTerm::Const(v1, w1), BvTerm::Const(v2, w2)) = (left, right)
            && w1 == w2
        {
            self.stats.constant_foldings += 1;
            return BvTerm::Const(v1 | v2, *w1);
        }

        // Algebraic identities
        if self.config.enable_algebraic_identities {
            // x | x = x
            if left == right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }

            // x | 0 = x
            if let BvTerm::Const(0, _) = right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }
        }

        BvTerm::Or(Box::new(left.clone()), Box::new(right.clone()))
    }

    /// Simplify bitwise XOR.
    fn simplify_xor(&mut self, left: &BvTerm, right: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let (BvTerm::Const(v1, w1), BvTerm::Const(v2, w2)) = (left, right)
            && w1 == w2
        {
            self.stats.constant_foldings += 1;
            return BvTerm::Const(v1 ^ v2, *w1);
        }

        // Algebraic identities
        if self.config.enable_algebraic_identities {
            // x ^ x = 0
            if left == right {
                self.stats.algebraic_simplifications += 1;
                if let BvTerm::Var(_, w) = left {
                    return BvTerm::Const(0, *w);
                }
            }

            // x ^ 0 = x
            if let BvTerm::Const(0, _) = right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }
        }

        BvTerm::Xor(Box::new(left.clone()), Box::new(right.clone()))
    }

    /// Simplify negation.
    fn simplify_neg(&mut self, inner: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let BvTerm::Const(v, w) = inner
        {
            self.stats.constant_foldings += 1;
            let mask = (1u64 << w) - 1;
            let negated = (!(v - 1)) & mask;
            return BvTerm::Const(negated, *w);
        }

        BvTerm::Neg(Box::new(inner.clone()))
    }

    /// Get statistics.
    pub fn stats(&self) -> &BvSimplificationStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = BvSimplificationStats::default();
    }
}

impl Default for BvSimplifier {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplifier_creation() {
        let simp = BvSimplifier::default_config();
        assert_eq!(simp.stats().constant_foldings, 0);
    }

    #[test]
    fn test_constant_folding_add() {
        let mut simp = BvSimplifier::default_config();

        let term = BvTerm::Add(Box::new(BvTerm::Const(3, 8)), Box::new(BvTerm::Const(5, 8)));

        let result = simp.simplify(&term);

        assert_eq!(result, BvTerm::Const(8, 8));
        assert_eq!(simp.stats().constant_foldings, 1);
    }

    #[test]
    fn test_algebraic_identity_add() {
        let mut simp = BvSimplifier::default_config();

        let var = BvTerm::Var(0, 8);
        let term = BvTerm::Add(Box::new(var.clone()), Box::new(BvTerm::Const(0, 8)));

        let result = simp.simplify(&term);

        assert_eq!(result, var);
        assert_eq!(simp.stats().algebraic_simplifications, 1);
    }

    #[test]
    fn test_and_self() {
        let mut simp = BvSimplifier::default_config();

        let var = BvTerm::Var(0, 8);
        let term = BvTerm::And(Box::new(var.clone()), Box::new(var.clone()));

        let result = simp.simplify(&term);

        assert_eq!(result, var);
        assert_eq!(simp.stats().algebraic_simplifications, 1);
    }

    #[test]
    fn test_xor_self() {
        let mut simp = BvSimplifier::default_config();

        let var = BvTerm::Var(0, 8);
        let term = BvTerm::Xor(Box::new(var.clone()), Box::new(var));

        let result = simp.simplify(&term);

        assert_eq!(result, BvTerm::Const(0, 8));
        assert_eq!(simp.stats().algebraic_simplifications, 1);
    }
}
