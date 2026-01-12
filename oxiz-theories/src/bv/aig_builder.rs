//! AIG-based BitVector circuit builder
//!
//! Provides a high-level API for building BitVector circuits using AIG representation,
//! which can then be efficiently converted to CNF for SAT solving.

use super::aig::{AigCircuit, AigEdge};
use oxiz_core::ast::TermId;
use oxiz_sat::{Solver as SatSolver, Var};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// BitVector circuit builder using AIG representation
#[derive(Debug)]
pub struct AigBvBuilder {
    /// The underlying AIG circuit
    circuit: AigCircuit,
    /// Mapping from TermId to bit vectors (represented as AIG edges)
    term_to_bits: FxHashMap<TermId, SmallVec<[AigEdge; 32]>>,
    /// Mapping from AIG nodes to SAT variables (populated after to_cnf)
    node_to_var: Option<FxHashMap<super::aig::NodeId, Var>>,
}

impl Default for AigBvBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AigBvBuilder {
    /// Create a new AIG-based BitVector builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            circuit: AigCircuit::new(),
            term_to_bits: FxHashMap::default(),
            node_to_var: None,
        }
    }

    /// Create a new bit vector variable
    pub fn new_bv(&mut self, term: TermId, width: u32, prefix: &str) -> &[AigEdge] {
        if !self.term_to_bits.contains_key(&term) {
            let mut bits = SmallVec::new();
            for i in 0..width {
                let name = format!("{}_{}", prefix, i);
                let edge = self.circuit.new_input(name);
                bits.push(edge);
            }
            self.term_to_bits.insert(term, bits);
        }
        self.term_to_bits
            .get(&term)
            .expect("BitVector should exist after insertion")
    }

    /// Get the bits for a term
    #[must_use]
    pub fn get_bits(&self, term: TermId) -> Option<&[AigEdge]> {
        self.term_to_bits.get(&term).map(|v| v.as_slice())
    }

    /// Assert equality: a = b (bitwise)
    pub fn assert_eq(&mut self, a: TermId, b: TermId) {
        if let (Some(bits_a), Some(bits_b)) = (
            self.term_to_bits.get(&a).cloned(),
            self.term_to_bits.get(&b).cloned(),
        ) {
            assert_eq!(bits_a.len(), bits_b.len());

            // For each bit: assert a[i] IFF b[i]
            for i in 0..bits_a.len() {
                let equiv = self.circuit.iff(bits_a[i], bits_b[i]);
                self.circuit.add_output(equiv);
            }
        }
    }

    /// Bitwise AND: result = a & b
    pub fn bv_and(&mut self, result: TermId, a: TermId, b: TermId, prefix: &str) {
        if let (Some(bits_a), Some(bits_b)) = (
            self.term_to_bits.get(&a).cloned(),
            self.term_to_bits.get(&b).cloned(),
        ) {
            assert_eq!(bits_a.len(), bits_b.len());
            let width = bits_a.len();

            let mut result_bits = SmallVec::new();
            for i in 0..width {
                let and_bit = self.circuit.and(bits_a[i], bits_b[i]);
                result_bits.push(and_bit);
            }

            self.term_to_bits.insert(result, result_bits);
        } else {
            // Create new bits if they don't exist
            let width = 32; // Default width
            self.new_bv(result, width, prefix);
        }
    }

    /// Bitwise OR: result = a | b
    pub fn bv_or(&mut self, result: TermId, a: TermId, b: TermId, prefix: &str) {
        if let (Some(bits_a), Some(bits_b)) = (
            self.term_to_bits.get(&a).cloned(),
            self.term_to_bits.get(&b).cloned(),
        ) {
            assert_eq!(bits_a.len(), bits_b.len());
            let width = bits_a.len();

            let mut result_bits = SmallVec::new();
            for i in 0..width {
                let or_bit = self.circuit.or(bits_a[i], bits_b[i]);
                result_bits.push(or_bit);
            }

            self.term_to_bits.insert(result, result_bits);
        } else {
            let width = 32;
            self.new_bv(result, width, prefix);
        }
    }

    /// Bitwise XOR: result = a ^ b
    pub fn bv_xor(&mut self, result: TermId, a: TermId, b: TermId, prefix: &str) {
        if let (Some(bits_a), Some(bits_b)) = (
            self.term_to_bits.get(&a).cloned(),
            self.term_to_bits.get(&b).cloned(),
        ) {
            assert_eq!(bits_a.len(), bits_b.len());
            let width = bits_a.len();

            let mut result_bits = SmallVec::new();
            for i in 0..width {
                let xor_bit = self.circuit.xor(bits_a[i], bits_b[i]);
                result_bits.push(xor_bit);
            }

            self.term_to_bits.insert(result, result_bits);
        } else {
            let width = 32;
            self.new_bv(result, width, prefix);
        }
    }

    /// Bitwise NOT: result = ~a
    pub fn bv_not(&mut self, result: TermId, a: TermId, prefix: &str) {
        if let Some(bits_a) = self.term_to_bits.get(&a).cloned() {
            let mut result_bits = SmallVec::new();
            for bit in bits_a {
                let not_bit = self.circuit.not(bit);
                result_bits.push(not_bit);
            }

            self.term_to_bits.insert(result, result_bits);
        } else {
            let width = 32;
            self.new_bv(result, width, prefix);
        }
    }

    /// Assert constant value for a bit vector
    pub fn assert_const(&mut self, term: TermId, value: u64, width: u32) {
        let prefix = format!("const_{}", value);
        self.new_bv(term, width, &prefix);

        if let Some(bits) = self.term_to_bits.get(&term).cloned() {
            for (i, &bit) in bits.iter().enumerate().take(width as usize) {
                let bit_value = (value >> i) & 1;
                let constraint = if bit_value == 1 {
                    bit
                } else {
                    self.circuit.not(bit)
                };
                self.circuit.add_output(constraint);
            }
        }
    }

    /// Convert the AIG circuit to CNF and add to SAT solver
    pub fn to_cnf(&mut self, sat: &mut SatSolver) {
        let node_to_var = self.circuit.to_cnf(sat);
        self.node_to_var = Some(node_to_var);
    }

    /// Get the AIG circuit (for inspection/debugging)
    #[must_use]
    pub fn circuit(&self) -> &AigCircuit {
        &self.circuit
    }

    /// Get statistics about the circuit
    #[must_use]
    pub fn stats(&self) -> super::aig::AigStats {
        self.circuit.stats()
    }

    /// BitVector addition: result = a + b
    pub fn bv_add(&mut self, result: TermId, a: TermId, b: TermId, prefix: &str) {
        if let (Some(bits_a), Some(bits_b)) = (
            self.term_to_bits.get(&a).cloned(),
            self.term_to_bits.get(&b).cloned(),
        ) {
            assert_eq!(bits_a.len(), bits_b.len());

            let result_bits = self.circuit.ripple_carry_adder(&bits_a, &bits_b);
            self.term_to_bits.insert(result, result_bits.into());
        } else {
            let width = 32;
            self.new_bv(result, width, prefix);
        }
    }

    /// BitVector negation (two's complement): result = -a
    pub fn bv_neg(&mut self, result: TermId, a: TermId, prefix: &str) {
        if let Some(bits_a) = self.term_to_bits.get(&a).cloned() {
            let result_bits = self.circuit.negate_bitvector(&bits_a);
            self.term_to_bits.insert(result, result_bits.into());
        } else {
            let width = 32;
            self.new_bv(result, width, prefix);
        }
    }

    /// BitVector subtraction: result = a - b
    pub fn bv_sub(&mut self, result: TermId, a: TermId, b: TermId, prefix: &str) {
        if let (Some(bits_a), Some(bits_b)) = (
            self.term_to_bits.get(&a).cloned(),
            self.term_to_bits.get(&b).cloned(),
        ) {
            assert_eq!(bits_a.len(), bits_b.len());

            // a - b = a + (-b)
            let neg_b = self.circuit.negate_bitvector(&bits_b);
            let result_bits = self.circuit.ripple_carry_adder(&bits_a, &neg_b);
            self.term_to_bits.insert(result, result_bits.into());
        } else {
            let width = 32;
            self.new_bv(result, width, prefix);
        }
    }

    /// Assert unsigned less than: a < b
    pub fn assert_ult(&mut self, a: TermId, b: TermId) {
        if let (Some(bits_a), Some(bits_b)) = (
            self.term_to_bits.get(&a).cloned(),
            self.term_to_bits.get(&b).cloned(),
        ) {
            let lt = self.circuit.unsigned_less_than(&bits_a, &bits_b);
            self.circuit.add_output(lt);
        }
    }

    /// Assert disequality: a != b
    pub fn assert_neq(&mut self, a: TermId, b: TermId) {
        if let (Some(bits_a), Some(bits_b)) = (
            self.term_to_bits.get(&a).cloned(),
            self.term_to_bits.get(&b).cloned(),
        ) {
            let eq = self.circuit.equal(&bits_a, &bits_b);
            let neq = self.circuit.not(eq);
            self.circuit.add_output(neq);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_sat::SolverResult;

    #[test]
    fn test_aig_builder_basic() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);

        builder.new_bv(a, 4, "a");
        builder.new_bv(b, 4, "b");

        builder.assert_eq(a, b);

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_builder_and() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        builder.new_bv(a, 4, "a");
        builder.new_bv(b, 4, "b");

        builder.bv_and(c, a, b, "c");

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_builder_const() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);

        builder.assert_const(a, 5, 4); // a = 0b0101

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_builder_stats() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        builder.new_bv(a, 4, "a");
        builder.new_bv(b, 4, "b");
        builder.bv_and(c, a, b, "c");

        let stats = builder.stats();
        assert_eq!(stats.num_inputs, 8); // 4 bits for a, 4 bits for b
        assert!(stats.num_and_gates >= 4); // At least 4 AND gates (one per bit)
    }

    #[test]
    fn test_aig_builder_add() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        builder.assert_const(a, 3, 4); // a = 3
        builder.assert_const(b, 5, 4); // b = 5
        builder.bv_add(c, a, b, "c"); // c = a + b = 8

        // Assert c = 8
        builder.assert_const(TermId::new(4), 8, 4);
        builder.assert_eq(c, TermId::new(4));

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_builder_sub() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);
        let c = TermId::new(3);

        builder.assert_const(a, 7, 4); // a = 7
        builder.assert_const(b, 3, 4); // b = 3
        builder.bv_sub(c, a, b, "c"); // c = a - b = 4

        // Assert c = 4
        builder.assert_const(TermId::new(4), 4, 4);
        builder.assert_eq(c, TermId::new(4));

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_builder_neg() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);

        builder.assert_const(a, 5, 4); // a = 5
        builder.bv_neg(b, a, "b"); // b = -5 = 11 in 4-bit unsigned

        // Assert b = 11
        builder.assert_const(TermId::new(3), 11, 4);
        builder.assert_eq(b, TermId::new(3));

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_builder_ult() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);

        builder.assert_const(a, 3, 4); // a = 3
        builder.assert_const(b, 5, 4); // b = 5
        builder.assert_ult(a, b); // assert a < b (should be SAT)

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_builder_ult_false() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);

        builder.assert_const(a, 7, 4); // a = 7
        builder.assert_const(b, 3, 4); // b = 3
        builder.assert_ult(a, b); // assert a < b (should be UNSAT since 7 >= 3)

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Unsat));
    }

    #[test]
    fn test_aig_builder_neq() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);

        builder.assert_const(a, 3, 4); // a = 3
        builder.assert_const(b, 5, 4); // b = 5
        builder.assert_neq(a, b); // assert a != b (should be SAT)

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_builder_neq_false() {
        let mut builder = AigBvBuilder::new();
        let a = TermId::new(1);
        let b = TermId::new(2);

        builder.assert_const(a, 5, 4); // a = 5
        builder.assert_const(b, 5, 4); // b = 5
        builder.assert_neq(a, b); // assert a != b (should be UNSAT since a == b)

        let mut sat = SatSolver::new();
        builder.to_cnf(&mut sat);

        let result = sat.solve();
        assert!(matches!(result, SolverResult::Unsat));
    }
}
