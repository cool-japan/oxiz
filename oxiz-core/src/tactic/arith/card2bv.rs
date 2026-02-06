//! Cardinality to BitVector Encoding Tactic.
#![allow(missing_docs, dead_code)] // Under development
//!
//! This tactic converts cardinality constraints (at-most-k, at-least-k)
//! into bitvector constraints, enabling efficient solving via bit-blasting.
//!
//! ## Transformations
//!
//! 1. **At-Most-K**: (b1 + b2 + ... + bn ≤ k) → BV encoding with carry
//! 2. **At-Least-K**: (b1 + b2 + ... + bn ≥ k) → Negation + at-most
//! 3. **Exactly-K**: Conjunction of at-most and at-least
//!
//! ## Encoding Strategies
//!
//! 1. **Unary**: Direct bit representation (efficient for small k)
//! 2. **Binary**: Log encoding (efficient for large k)
//! 3. **Mixed**: Hybrid approach based on k/n ratio
//!
//! ## Benefits
//!
//! - Reduces search space for cardinality constraints
//! - Enables SAT solver preprocessing on encoded form
//! - Often faster than clause-based encodings
//!
//! ## References
//!
//! - Bailleux & Boufkhad: "Efficient CNF Encoding of Boolean Cardinality Constraints" (CP 2003)
//! - Z3's `tactic/arith/card2bv_tactic.cpp`

/// Boolean literal (variable or negation).
pub type BoolLit = i32;

/// BitVector width type.
pub type BvWidth = u32;

/// Configuration for card2bv tactic.
#[derive(Debug, Clone)]
pub struct Card2BvConfig {
    /// Use unary encoding for small k.
    pub use_unary: bool,
    /// Use binary encoding for large k.
    pub use_binary: bool,
    /// Threshold for unary vs binary (k/n ratio).
    pub unary_threshold: f64,
    /// Maximum cardinality to encode.
    pub max_cardinality: u32,
}

impl Default for Card2BvConfig {
    fn default() -> Self {
        Self {
            use_unary: true,
            use_binary: true,
            unary_threshold: 0.3, // Use unary if k < 0.3*n
            max_cardinality: 10000,
        }
    }
}

/// Statistics for card2bv tactic.
#[derive(Debug, Clone, Default)]
pub struct Card2BvStats {
    /// Constraints encoded.
    pub constraints_encoded: u64,
    /// Unary encodings used.
    pub unary_encodings: u64,
    /// Binary encodings used.
    pub binary_encodings: u64,
    /// Bitvector variables created.
    pub bv_vars_created: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Cardinality constraint type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardinalityConstraint {
    /// At most k literals are true.
    AtMost { lits: Vec<BoolLit>, k: u32 },
    /// At least k literals are true.
    AtLeast { lits: Vec<BoolLit>, k: u32 },
    /// Exactly k literals are true.
    Exactly { lits: Vec<BoolLit>, k: u32 },
}

/// Bitvector variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BvVar {
    /// Variable ID.
    pub id: u32,
    /// Width in bits.
    pub width: BvWidth,
}

/// Bitvector term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvTerm {
    /// Variable.
    Var(BvVar),
    /// Constant.
    Const { value: u64, width: BvWidth },
    /// Addition.
    Add(Box<BvTerm>, Box<BvTerm>),
    /// Bitwise AND.
    And(Box<BvTerm>, Box<BvTerm>),
    /// Bitwise OR.
    Or(Box<BvTerm>, Box<BvTerm>),
    /// Comparison (≤).
    Le(Box<BvTerm>, Box<BvTerm>),
    /// Comparison (≥).
    Ge(Box<BvTerm>, Box<BvTerm>),
}

/// Encoding result.
#[derive(Debug, Clone)]
pub struct EncodingResult {
    /// Bitvector term representing the constraint.
    pub term: BvTerm,
    /// Auxiliary variables created.
    pub aux_vars: Vec<BvVar>,
    /// Encoding metadata.
    pub metadata: EncodingMetadata,
}

/// Metadata about encoding.
#[derive(Debug, Clone)]
pub struct EncodingMetadata {
    /// Encoding strategy used.
    pub strategy: EncodingStrategy,
    /// Number of bits used.
    pub bits_used: u32,
}

/// Encoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingStrategy {
    /// Unary encoding.
    Unary,
    /// Binary encoding.
    Binary,
    /// Mixed encoding.
    Mixed,
}

/// Card2BV tactic.
pub struct Card2BvTactic {
    config: Card2BvConfig,
    stats: Card2BvStats,
    /// Next fresh variable ID.
    next_var: u32,
}

impl Card2BvTactic {
    /// Create new tactic.
    pub fn new() -> Self {
        Self::with_config(Card2BvConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: Card2BvConfig) -> Self {
        Self {
            config,
            stats: Card2BvStats::default(),
            next_var: 0,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &Card2BvStats {
        &self.stats
    }

    /// Encode cardinality constraint to bitvector.
    pub fn encode(&mut self, constraint: &CardinalityConstraint) -> EncodingResult {
        let start = std::time::Instant::now();

        let result = match constraint {
            CardinalityConstraint::AtMost { lits, k } => self.encode_at_most(lits, *k),
            CardinalityConstraint::AtLeast { lits, k } => self.encode_at_least(lits, *k),
            CardinalityConstraint::Exactly { lits, k } => self.encode_exactly(lits, *k),
        };

        self.stats.constraints_encoded += 1;
        self.stats.time_us += start.elapsed().as_micros() as u64;

        result
    }

    /// Encode at-most-k constraint.
    fn encode_at_most(&mut self, lits: &[BoolLit], k: u32) -> EncodingResult {
        if k > self.config.max_cardinality {
            // Too large - use trivial encoding
            return self.encode_trivial(lits, k);
        }

        let n = lits.len() as u32;
        let ratio = k as f64 / n as f64;

        if self.config.use_unary && ratio <= self.config.unary_threshold {
            self.encode_unary_at_most(lits, k)
        } else if self.config.use_binary {
            self.encode_binary_at_most(lits, k)
        } else {
            self.encode_unary_at_most(lits, k)
        }
    }

    /// Encode at-least-k constraint.
    fn encode_at_least(&mut self, lits: &[BoolLit], k: u32) -> EncodingResult {
        let n = lits.len() as u32;

        // at-least-k = at-most-(n-k) on negated literals
        let negated: Vec<BoolLit> = lits.iter().map(|&lit| -lit).collect();

        self.encode_at_most(&negated, n - k)
    }

    /// Encode exactly-k constraint.
    fn encode_exactly(&mut self, lits: &[BoolLit], k: u32) -> EncodingResult {
        // exactly-k = at-most-k ∧ at-least-k

        let at_most = self.encode_at_most(lits, k);
        let at_least = self.encode_at_least(lits, k);

        let term = BvTerm::And(Box::new(at_most.term), Box::new(at_least.term));

        let mut aux_vars = at_most.aux_vars;
        aux_vars.extend(at_least.aux_vars);

        EncodingResult {
            term,
            aux_vars,
            metadata: EncodingMetadata {
                strategy: EncodingStrategy::Mixed,
                bits_used: at_most.metadata.bits_used + at_least.metadata.bits_used,
            },
        }
    }

    /// Unary encoding for at-most-k.
    fn encode_unary_at_most(&mut self, lits: &[BoolLit], k: u32) -> EncodingResult {
        self.stats.unary_encodings += 1;

        let n = lits.len();

        // Create n-bit bitvector where each bit corresponds to a literal
        let bv_width = n as BvWidth;
        let bv_var = self.fresh_bv_var(bv_width);

        // Sum of bits
        let sum_term = self.build_sum_term(lits);

        // sum ≤ k
        let k_term = BvTerm::Const {
            value: k as u64,
            width: bv_width,
        };

        let constraint_term = BvTerm::Le(Box::new(sum_term), Box::new(k_term));

        EncodingResult {
            term: constraint_term,
            aux_vars: vec![bv_var],
            metadata: EncodingMetadata {
                strategy: EncodingStrategy::Unary,
                bits_used: bv_width,
            },
        }
    }

    /// Binary encoding for at-most-k.
    fn encode_binary_at_most(&mut self, lits: &[BoolLit], k: u32) -> EncodingResult {
        self.stats.binary_encodings += 1;

        let n = lits.len();

        // Log encoding: ceil(log2(n+1)) bits
        let bv_width = ((n + 1) as f64).log2().ceil() as BvWidth;
        let bv_var = self.fresh_bv_var(bv_width);

        // Build binary sum
        let sum_term = self.build_binary_sum(lits, bv_width);

        // sum ≤ k
        let k_term = BvTerm::Const {
            value: k as u64,
            width: bv_width,
        };

        let constraint_term = BvTerm::Le(Box::new(sum_term), Box::new(k_term));

        EncodingResult {
            term: constraint_term,
            aux_vars: vec![bv_var],
            metadata: EncodingMetadata {
                strategy: EncodingStrategy::Binary,
                bits_used: bv_width,
            },
        }
    }

    /// Trivial encoding (always true).
    fn encode_trivial(&mut self, _lits: &[BoolLit], _k: u32) -> EncodingResult {
        // Return constant true (all comparisons succeed)
        EncodingResult {
            term: BvTerm::Const { value: 1, width: 1 },
            aux_vars: Vec::new(),
            metadata: EncodingMetadata {
                strategy: EncodingStrategy::Binary,
                bits_used: 1,
            },
        }
    }

    /// Build sum term from literals.
    fn build_sum_term(&self, lits: &[BoolLit]) -> BvTerm {
        if lits.is_empty() {
            return BvTerm::Const { value: 0, width: 1 };
        }

        // Convert each literal to 1-bit BV, then sum
        let width = lits.len() as BvWidth;

        // Simplified: return a sum representation
        // In full implementation, build adder tree
        BvTerm::Const { value: 0, width }
    }

    /// Build binary sum with logarithmic width.
    fn build_binary_sum(&self, lits: &[BoolLit], width: BvWidth) -> BvTerm {
        if lits.is_empty() {
            return BvTerm::Const { value: 0, width };
        }

        // Build adder tree with log width
        // Simplified: return placeholder
        BvTerm::Const { value: 0, width }
    }

    /// Allocate fresh bitvector variable.
    fn fresh_bv_var(&mut self, width: BvWidth) -> BvVar {
        let var = BvVar {
            id: self.next_var,
            width,
        };
        self.next_var += 1;
        self.stats.bv_vars_created += 1;
        var
    }
}

impl Default for Card2BvTactic {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tactic_creation() {
        let tactic = Card2BvTactic::new();
        assert_eq!(tactic.stats().constraints_encoded, 0);
    }

    #[test]
    fn test_fresh_bv_var() {
        let mut tactic = Card2BvTactic::new();

        let v1 = tactic.fresh_bv_var(4);
        let v2 = tactic.fresh_bv_var(8);

        assert_ne!(v1.id, v2.id);
        assert_eq!(v1.width, 4);
        assert_eq!(v2.width, 8);
    }

    #[test]
    fn test_encode_at_most() {
        let mut tactic = Card2BvTactic::new();

        let lits = vec![1, 2, 3, 4];
        let constraint = CardinalityConstraint::AtMost { lits, k: 2 };

        let result = tactic.encode(&constraint);

        assert!(!result.aux_vars.is_empty());
        assert_eq!(tactic.stats().constraints_encoded, 1);
    }

    #[test]
    fn test_encode_exactly() {
        let mut tactic = Card2BvTactic::new();

        let lits = vec![1, 2, 3];
        let constraint = CardinalityConstraint::Exactly { lits, k: 2 };

        let result = tactic.encode(&constraint);

        assert_eq!(result.metadata.strategy, EncodingStrategy::Mixed);
    }

    #[test]
    fn test_unary_strategy() {
        let config = Card2BvConfig {
            use_unary: true,
            use_binary: false,
            unary_threshold: 1.0, // Always unary
            max_cardinality: 10000,
        };

        let mut tactic = Card2BvTactic::with_config(config);

        let lits = vec![1, 2, 3, 4, 5];
        let constraint = CardinalityConstraint::AtMost { lits, k: 2 };

        let result = tactic.encode(&constraint);

        assert_eq!(result.metadata.strategy, EncodingStrategy::Unary);
        assert_eq!(tactic.stats().unary_encodings, 1);
    }

    #[test]
    fn test_binary_strategy() {
        let config = Card2BvConfig {
            use_unary: false,
            use_binary: true,
            unary_threshold: 0.0, // Always binary
            max_cardinality: 10000,
        };

        let mut tactic = Card2BvTactic::with_config(config);

        let lits = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let constraint = CardinalityConstraint::AtMost { lits, k: 3 };

        let result = tactic.encode(&constraint);

        assert_eq!(result.metadata.strategy, EncodingStrategy::Binary);
        assert_eq!(tactic.stats().binary_encodings, 1);
    }

    #[test]
    fn test_encode_at_least() {
        let mut tactic = Card2BvTactic::new();

        let lits = vec![1, 2, 3, 4];
        let constraint = CardinalityConstraint::AtLeast { lits, k: 3 };

        let result = tactic.encode(&constraint);

        assert!(!result.aux_vars.is_empty());
    }

    #[test]
    fn test_trivial_encoding() {
        let mut tactic = Card2BvTactic::new();

        let lits = vec![1; 20000]; // Exceeds max_cardinality
        let result = tactic.encode_trivial(&lits, 10000);

        assert_eq!(result.aux_vars.len(), 0);
    }
}
