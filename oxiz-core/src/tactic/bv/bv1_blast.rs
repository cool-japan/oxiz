//! Bit-Level Blasting Tactic for BitVectors.
#![allow(dead_code)] // Under development - not yet fully integrated
//!
//! This tactic decomposes bitvector operations into Boolean operations
//! on individual bits, enabling SAT solver-based reasoning.
//!
//! ## Transformations
//!
//! 1. **Addition**: Implement as ripple-carry adder circuit
//! 2. **Multiplication**: Implement as array multiplier or Wallace tree
//! 3. **Comparison**: Bit-by-bit comparison with carry propagation
//! 4. **Shifts**: Mux-based implementation
//! 5. **Arithmetic**: Signed operations via two's complement
//!
//! ## Example
//!
//! ```text
//! x + y (4-bit)  becomes:
//! s[0] = x[0] ⊕ y[0]
//! c[0] = x[0] ∧ y[0]
//! s[1] = x[1] ⊕ y[1] ⊕ c[0]
//! c[1] = (x[1] ∧ y[1]) ∨ (x[1] ∧ c[0]) ∨ (y[1] ∧ c[0])
//! ...
//! ```
//!
//! ## Benefits
//!
//! - Enables powerful SAT solver preprocessing
//! - Allows bit-level optimizations
//! - Simplifies some bitvector constraints
//! - Can expose structure for other tactics
//!
//! ## References
//!
//! - Brummayer & Biere: "Boolector: An Efficient SMT Solver for Bit-Vectors and Arrays" (TACAS 2009)
//! - Z3's `tactic/bv/bv1_blast_tactic.cpp`

/// A Boolean literal (variable or negated variable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoolLit {
    /// Variable ID.
    pub var: u32,
    /// Is negated.
    pub negated: bool,
}

impl BoolLit {
    /// Create positive literal.
    pub fn pos(var: u32) -> Self {
        Self {
            var,
            negated: false,
        }
    }

    /// Create negative literal.
    pub fn neg(var: u32) -> Self {
        Self { var, negated: true }
    }

    /// Negate literal.
    pub fn negate(self) -> Self {
        Self {
            var: self.var,
            negated: !self.negated,
        }
    }
}

/// A Boolean expression (combination of literals).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoolExpr {
    /// Literal.
    Lit(BoolLit),
    /// AND of expressions.
    And(Vec<BoolExpr>),
    /// OR of expressions.
    Or(Vec<BoolExpr>),
    /// XOR of two expressions.
    Xor(Box<BoolExpr>, Box<BoolExpr>),
    /// NOT of expression.
    Not(Box<BoolExpr>),
    /// If-then-else.
    Ite(Box<BoolExpr>, Box<BoolExpr>, Box<BoolExpr>),
}

/// A bitvector represented as vector of Boolean expressions.
pub type BitVec = Vec<BoolExpr>;

/// Configuration for bit-blasting.
#[derive(Debug, Clone)]
pub struct Bv1BlastConfig {
    /// Enable carry-lookahead for addition (faster but more clauses).
    pub use_carry_lookahead: bool,
    /// Enable Wallace tree for multiplication (faster but more complex).
    pub use_wallace_tree: bool,
    /// Maximum width for full blasting (larger widths use mixed approach).
    pub max_full_blast_width: u32,
}

impl Default for Bv1BlastConfig {
    fn default() -> Self {
        Self {
            use_carry_lookahead: false, // Ripple carry is simpler
            use_wallace_tree: false,    // Array multiplier is simpler
            max_full_blast_width: 64,
        }
    }
}

/// Statistics for bit-blasting.
#[derive(Debug, Clone, Default)]
pub struct Bv1BlastStats {
    /// Bitvectors blasted.
    pub bitvectors_blasted: u64,
    /// Total bits generated.
    pub bits_generated: u64,
    /// Boolean clauses generated.
    pub clauses_generated: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Bit-blasting tactic.
pub struct Bv1BlastTactic {
    config: Bv1BlastConfig,
    stats: Bv1BlastStats,
    /// Next fresh variable ID.
    next_var: u32,
}

impl Bv1BlastTactic {
    /// Create new bit-blasting tactic.
    pub fn new() -> Self {
        Self::with_config(Bv1BlastConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: Bv1BlastConfig) -> Self {
        Self {
            config,
            stats: Bv1BlastStats::default(),
            next_var: 0,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &Bv1BlastStats {
        &self.stats
    }

    /// Allocate fresh Boolean variable.
    fn fresh_var(&mut self) -> BoolLit {
        let var = self.next_var;
        self.next_var += 1;
        BoolLit::pos(var)
    }

    /// Blast bitvector constant to bits.
    pub fn blast_constant(&mut self, value: u64, width: u32) -> BitVec {
        let mut bits = Vec::with_capacity(width as usize);

        for i in 0..width {
            let bit = (value >> i) & 1;
            let lit = self.fresh_var();

            bits.push(if bit == 1 {
                BoolExpr::Lit(lit)
            } else {
                BoolExpr::Lit(lit.negate())
            });
        }

        self.stats.bits_generated += width as u64;
        bits
    }

    /// Blast bitvector addition: a + b.
    pub fn blast_add(&mut self, a: &BitVec, b: &BitVec) -> BitVec {
        assert_eq!(a.len(), b.len(), "Bitvector widths must match");

        let width = a.len();
        let mut result = Vec::with_capacity(width);

        // Initialize carry variable to false (0)
        let carry_var = self.next_var;
        self.next_var += 1;
        let mut carry = BoolExpr::Lit(BoolLit::neg(carry_var));

        for i in 0..width {
            // Full adder: sum = a ⊕ b ⊕ carry
            let sum = BoolExpr::Xor(
                Box::new(BoolExpr::Xor(
                    Box::new(a[i].clone()),
                    Box::new(b[i].clone()),
                )),
                Box::new(carry.clone()),
            );

            // Carry = (a ∧ b) ∨ (a ∧ carry) ∨ (b ∧ carry)
            let new_carry = BoolExpr::Or(vec![
                BoolExpr::And(vec![a[i].clone(), b[i].clone()]),
                BoolExpr::And(vec![a[i].clone(), carry.clone()]),
                BoolExpr::And(vec![b[i].clone(), carry.clone()]),
            ]);

            result.push(sum);
            carry = new_carry;

            self.stats.clauses_generated += 4; // Approximate clause count
        }

        self.stats.bitvectors_blasted += 1;
        self.stats.bits_generated += width as u64;

        result
    }

    /// Blast bitvector multiplication: a * b.
    pub fn blast_mul(&mut self, a: &BitVec, b: &BitVec) -> BitVec {
        assert_eq!(a.len(), b.len(), "Bitvector widths must match");

        let width = a.len();

        // Simple array multiplier (shift-and-add)
        let mut result = vec![BoolExpr::Lit(BoolLit::neg(0)); width];

        for i in 0..width {
            // Shift a by i positions
            let mut shifted = vec![BoolExpr::Lit(BoolLit::neg(0)); width];

            for j in 0..(width - i) {
                // If b[i] is 1, add shifted a to result
                shifted[i + j] = BoolExpr::And(vec![b[i].clone(), a[j].clone()]);
            }

            // Add shifted to result
            result = self.blast_add(&result, &shifted);
        }

        self.stats.bitvectors_blasted += 1;
        self.stats.bits_generated += width as u64;

        result
    }

    /// Blast bitvector unsigned less-than: a < b.
    pub fn blast_ult(&mut self, a: &BitVec, b: &BitVec) -> BoolExpr {
        assert_eq!(a.len(), b.len(), "Bitvector widths must match");

        let width = a.len();

        // Compare from MSB to LSB
        let mut less_than = BoolExpr::Lit(BoolLit::neg(0)); // Initially false

        for i in (0..width).rev() {
            // less_than = (!a[i] ∧ b[i]) ∨ (a[i] = b[i] ∧ less_than)
            let a_lt_b_at_i =
                BoolExpr::And(vec![BoolExpr::Not(Box::new(a[i].clone())), b[i].clone()]);

            let a_eq_b_at_i = BoolExpr::And(vec![BoolExpr::Xor(
                Box::new(BoolExpr::Xor(
                    Box::new(a[i].clone()),
                    Box::new(b[i].clone()),
                )),
                Box::new(BoolExpr::Lit(BoolLit::pos(0))),
            )]);

            less_than = BoolExpr::Or(vec![
                a_lt_b_at_i,
                BoolExpr::And(vec![a_eq_b_at_i, less_than]),
            ]);
        }

        self.stats.clauses_generated += width as u64 * 2;

        less_than
    }

    /// Blast bitvector left shift: a << n.
    pub fn blast_shl(&mut self, a: &BitVec, shift_amount: u32) -> BitVec {
        let width = a.len();
        let mut result = Vec::with_capacity(width);

        for i in 0..width {
            if i < shift_amount as usize {
                // Shifted-in bits are 0
                result.push(BoolExpr::Lit(BoolLit::neg(0)));
            } else {
                result.push(a[i - shift_amount as usize].clone());
            }
        }

        self.stats.bits_generated += width as u64;

        result
    }

    /// Blast bitvector extraction: a\[high:low\].
    pub fn blast_extract(&mut self, a: &BitVec, high: u32, low: u32) -> BitVec {
        assert!(high < a.len() as u32 && low <= high);

        let mut result = Vec::new();

        for i in low..=high {
            result.push(a[i as usize].clone());
        }

        self.stats.bits_generated += (high - low + 1) as u64;

        result
    }
}

impl Default for Bv1BlastTactic {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tactic_creation() {
        let tactic = Bv1BlastTactic::new();
        assert_eq!(tactic.stats().bitvectors_blasted, 0);
    }

    #[test]
    fn test_blast_constant() {
        let mut tactic = Bv1BlastTactic::new();

        // Blast constant 5 (0b101) as 3-bit vector
        let bits = tactic.blast_constant(5, 3);

        assert_eq!(bits.len(), 3);
        assert_eq!(tactic.stats().bits_generated, 3);
    }

    #[test]
    fn test_fresh_var() {
        let mut tactic = Bv1BlastTactic::new();

        let v1 = tactic.fresh_var();
        let v2 = tactic.fresh_var();

        assert_ne!(v1.var, v2.var);
    }

    #[test]
    fn test_blast_add() {
        let mut tactic = Bv1BlastTactic::new();

        let a = tactic.blast_constant(3, 4); // 0011
        let b = tactic.blast_constant(5, 4); // 0101

        let result = tactic.blast_add(&a, &b); // Should be 1000 (8)

        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_blast_extract() {
        let mut tactic = Bv1BlastTactic::new();

        let a = tactic.blast_constant(0b11010110, 8);

        // Extract bits [5:2] = 0b1001
        let extracted = tactic.blast_extract(&a, 5, 2);

        assert_eq!(extracted.len(), 4);
    }

    #[test]
    fn test_blast_shl() {
        let mut tactic = Bv1BlastTactic::new();

        let a = tactic.blast_constant(0b0011, 4);

        // Shift left by 1: 0b0011 << 1 = 0b0110
        let shifted = tactic.blast_shl(&a, 1);

        assert_eq!(shifted.len(), 4);
    }
}
