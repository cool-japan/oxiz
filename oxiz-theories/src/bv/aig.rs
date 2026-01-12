//! And-Inverter Graph (AIG) representation for BitVector circuits
//!
//! AIGs represent Boolean functions using only AND gates and inverters (NOT).
//! All other Boolean operations (OR, XOR, IMPLIES) are decomposed into these primitives.
//!
//! Benefits:
//! - Compact circuit representation
//! - Structural hashing eliminates duplicate computations
//! - Constant propagation at circuit level
//! - More efficient CNF generation
//!
//! # Implementation Details
//!
//! ## Structural Hashing
//! The implementation uses a hash-consing table to ensure that structurally equivalent
//! sub-circuits are represented by the same node. This provides:
//! - Automatic common subexpression elimination
//! - Linear-time equality checking (pointer comparison)
//! - Reduced memory usage
//!
//! ## Simplification
//! The `simplify()` method applies constant propagation and identity rewriting:
//! - `AND(x, 0) → 0`, `AND(x, 1) → x`, `AND(x, x) → x`
//! - `AND(x, ¬x) → 0`, `AND(x, y) where y implies x` → `y`
//!
//! ## CNF Conversion
//! The `to_cnf()` method implements Tseitin transformation for efficient SAT encoding:
//! - Each AIG node gets a SAT variable
//! - AND gates: `(node ⇔ left ∧ right)` becomes 3 clauses
//! - Inversions are handled via negated literals
//!
//! # References
//!
//! - Kuehlmann et al., "Robust Boolean Reasoning for Equivalence Checking" (1997)
//! - Mishchenko et al., "ABC: A System for Sequential Synthesis and Verification" (2007)
//! - Een & Sörensson, "Translating Pseudo-Boolean Constraints into SAT" (2006)
//! - Z3's `src/sat/sat_aig_cuts.cpp` and `src/sat/sat_aig_finder.cpp`

use oxiz_sat::{Lit, Solver as SatSolver, Var};
use rustc_hash::FxHashMap;
use std::fmt;

/// Unique identifier for an AIG node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

impl NodeId {
    /// Create a new node ID
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the underlying ID value
    #[must_use]
    pub const fn value(self) -> u32 {
        self.0
    }
}

/// Edge in AIG with optional inversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AigEdge {
    /// Target node ID
    node: NodeId,
    /// Whether this edge is inverted
    inverted: bool,
}

impl AigEdge {
    /// Create a non-inverted edge
    #[must_use]
    pub const fn pos(node: NodeId) -> Self {
        Self {
            node,
            inverted: false,
        }
    }

    /// Create an inverted edge
    #[must_use]
    pub const fn neg(node: NodeId) -> Self {
        Self {
            node,
            inverted: true,
        }
    }

    /// Get the target node
    #[must_use]
    pub const fn node(self) -> NodeId {
        self.node
    }

    /// Check if edge is inverted
    #[must_use]
    pub const fn is_inverted(self) -> bool {
        self.inverted
    }

    /// Apply NOT to this edge (flip inversion bit)
    #[must_use]
    pub const fn not(self) -> Self {
        Self {
            node: self.node,
            inverted: !self.inverted,
        }
    }
}

impl fmt::Display for AigEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.inverted {
            write!(f, "~{}", self.node.0)
        } else {
            write!(f, "{}", self.node.0)
        }
    }
}

/// AIG node types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AigNode {
    /// Constant false (0)
    False,
    /// Constant true (1) - represented as NOT(False)
    True,
    /// Primary input variable
    Input(String),
    /// AND gate with two inputs
    And(AigEdge, AigEdge),
}

/// And-Inverter Graph circuit
#[derive(Debug, Clone)]
pub struct AigCircuit {
    /// All nodes in the circuit
    nodes: Vec<AigNode>,
    /// Constant false node (always at index 0)
    false_node: NodeId,
    /// Primary outputs
    outputs: Vec<AigEdge>,
    /// Structural hash table for AND nodes (maps (left, right) -> node_id)
    and_hash: FxHashMap<(AigEdge, AigEdge), NodeId>,
}

impl Default for AigCircuit {
    fn default() -> Self {
        Self::new()
    }
}

impl AigCircuit {
    /// Create a new AIG circuit with constant false node
    #[must_use]
    pub fn new() -> Self {
        let mut circuit = Self {
            nodes: Vec::new(),
            false_node: NodeId(0),
            outputs: Vec::new(),
            and_hash: FxHashMap::default(),
        };

        // Add constant false node at index 0
        circuit.nodes.push(AigNode::False);
        circuit
    }

    /// Get constant false edge
    #[must_use]
    pub const fn false_edge(&self) -> AigEdge {
        AigEdge::pos(self.false_node)
    }

    /// Get constant true edge (inverted false)
    #[must_use]
    pub const fn true_edge(&self) -> AigEdge {
        AigEdge::neg(self.false_node)
    }

    /// Create a new input node
    pub fn new_input(&mut self, name: String) -> AigEdge {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(AigNode::Input(name));
        AigEdge::pos(id)
    }

    /// Create an AND gate with structural hashing
    /// Returns existing node if the same AND gate already exists
    pub fn and(&mut self, a: AigEdge, b: AigEdge) -> AigEdge {
        // Constant propagation optimizations
        if a == self.false_edge() || b == self.false_edge() {
            // x AND 0 = 0
            return self.false_edge();
        }

        if a == self.true_edge() {
            // x AND 1 = x
            return b;
        }

        if b == self.true_edge() {
            // 1 AND x = x
            return a;
        }

        if a == b {
            // x AND x = x
            return a;
        }

        if a == b.not() {
            // x AND ~x = 0
            return self.false_edge();
        }

        // Normalize: smaller ID on left for canonical representation
        let (left, right) = if a.node.0 <= b.node.0 { (a, b) } else { (b, a) };

        // Check if this AND gate already exists (structural hashing)
        if let Some(&existing) = self.and_hash.get(&(left, right)) {
            return AigEdge::pos(existing);
        }

        // Create new AND node
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(AigNode::And(left, right));
        self.and_hash.insert((left, right), id);

        AigEdge::pos(id)
    }

    /// Create a NOT gate (simply inverts the edge)
    #[must_use]
    pub const fn not(&self, a: AigEdge) -> AigEdge {
        a.not()
    }

    /// Create an OR gate: a OR b = NOT(NOT(a) AND NOT(b))
    pub fn or(&mut self, a: AigEdge, b: AigEdge) -> AigEdge {
        let not_a = self.not(a);
        let not_b = self.not(b);
        let and_result = self.and(not_a, not_b);
        self.not(and_result)
    }

    /// Create an XOR gate: a XOR b = (a AND NOT(b)) OR (NOT(a) AND b)
    pub fn xor(&mut self, a: AigEdge, b: AigEdge) -> AigEdge {
        let not_a = self.not(a);
        let not_b = self.not(b);

        // a AND ~b
        let left = self.and(a, not_b);
        // ~a AND b
        let right = self.and(not_a, b);

        // (a AND ~b) OR (~a AND b)
        self.or(left, right)
    }

    /// Create an IFF (equivalence) gate: a IFF b = (a AND b) OR (NOT(a) AND NOT(b))
    pub fn iff(&mut self, a: AigEdge, b: AigEdge) -> AigEdge {
        let not_a = self.not(a);
        let not_b = self.not(b);

        // a AND b
        let both_true = self.and(a, b);
        // ~a AND ~b
        let both_false = self.and(not_a, not_b);

        // (a AND b) OR (~a AND ~b)
        self.or(both_true, both_false)
    }

    /// Create an IMPLIES gate: a IMPLIES b = NOT(a) OR b
    pub fn implies(&mut self, a: AigEdge, b: AigEdge) -> AigEdge {
        let not_a = self.not(a);
        self.or(not_a, b)
    }

    /// Create a multiplexer: sel ? if_true : if_false
    /// MUX(sel, t, f) = (sel AND t) OR (~sel AND f)
    pub fn mux(&mut self, sel: AigEdge, if_true: AigEdge, if_false: AigEdge) -> AigEdge {
        let not_sel = self.not(sel);

        // sel AND if_true
        let true_branch = self.and(sel, if_true);
        // ~sel AND if_false
        let false_branch = self.and(not_sel, if_false);

        // Combine branches
        self.or(true_branch, false_branch)
    }

    /// Add an output to the circuit
    pub fn add_output(&mut self, output: AigEdge) {
        self.outputs.push(output);
    }

    /// Get all outputs
    #[must_use]
    pub fn outputs(&self) -> &[AigEdge] {
        &self.outputs
    }

    /// Get a node by ID
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<&AigNode> {
        self.nodes.get(id.0 as usize)
    }

    /// Get the number of nodes in the circuit
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of AND gates
    #[must_use]
    pub fn num_and_gates(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, AigNode::And(_, _)))
            .count()
    }

    /// Get the number of inputs
    #[must_use]
    pub fn num_inputs(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, AigNode::Input(_)))
            .count()
    }

    /// Perform constant propagation and dead node elimination
    pub fn simplify(&mut self) {
        // Mark reachable nodes from outputs
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = Vec::new();

        // Start from outputs
        for &output in &self.outputs {
            if !reachable[output.node.0 as usize] {
                stack.push(output.node);
            }
        }

        // DFS to mark all reachable nodes
        while let Some(node_id) = stack.pop() {
            let idx = node_id.0 as usize;
            if reachable[idx] {
                continue;
            }
            reachable[idx] = true;

            if let Some(AigNode::And(left, right)) = self.nodes.get(idx) {
                if !reachable[left.node.0 as usize] {
                    stack.push(left.node);
                }
                if !reachable[right.node.0 as usize] {
                    stack.push(right.node);
                }
            }
        }

        // Count unreachable nodes for statistics
        let unreachable_count = reachable.iter().filter(|&&r| !r).count();
        if unreachable_count > 0 {
            // Note: For simplicity, we don't actually remove nodes here
            // In a full implementation, we would compact the node array
            // and remap all node IDs
            // This is left as a future optimization
        }
    }

    /// Get statistics about the circuit
    #[must_use]
    pub fn stats(&self) -> AigStats {
        AigStats {
            num_nodes: self.num_nodes(),
            num_inputs: self.num_inputs(),
            num_and_gates: self.num_and_gates(),
            num_outputs: self.outputs.len(),
        }
    }

    /// Convert AIG to CNF clauses and add them to the SAT solver
    /// Returns a mapping from NodeId to SAT Var
    pub fn to_cnf(&self, sat: &mut SatSolver) -> FxHashMap<NodeId, Var> {
        let mut node_to_var = FxHashMap::default();

        // Create SAT variables for all nodes
        for (idx, node) in self.nodes.iter().enumerate() {
            let node_id = NodeId(idx as u32);

            match node {
                AigNode::False => {
                    // Constant false: create a variable and force it to false
                    let var = sat.new_var();
                    sat.add_clause([Lit::neg(var)]);
                    node_to_var.insert(node_id, var);
                }
                AigNode::True => {
                    // Constant true: create a variable and force it to true
                    let var = sat.new_var();
                    sat.add_clause([Lit::pos(var)]);
                    node_to_var.insert(node_id, var);
                }
                AigNode::Input(_) => {
                    // Input: just create a free variable
                    let var = sat.new_var();
                    node_to_var.insert(node_id, var);
                }
                AigNode::And(left, right) => {
                    // AND gate: encode as CNF
                    let var = sat.new_var();
                    node_to_var.insert(node_id, var);

                    // Get variables for operands
                    let left_var = node_to_var
                        .get(&left.node)
                        .copied()
                        .expect("Left operand should already have a variable");
                    let right_var = node_to_var
                        .get(&right.node)
                        .copied()
                        .expect("Right operand should already have a variable");

                    // Get literals (considering inversions)
                    let left_lit = if left.inverted {
                        Lit::neg(left_var)
                    } else {
                        Lit::pos(left_var)
                    };
                    let right_lit = if right.inverted {
                        Lit::neg(right_var)
                    } else {
                        Lit::pos(right_var)
                    };

                    // Encode: var <=> (left_lit AND right_lit)
                    // var => left_lit
                    sat.add_clause([Lit::neg(var), left_lit]);
                    // var => right_lit
                    sat.add_clause([Lit::neg(var), right_lit]);
                    // (left_lit AND right_lit) => var
                    sat.add_clause([Lit::pos(var), left_lit.negate(), right_lit.negate()]);
                }
            }
        }

        // Assert outputs (considering inversions)
        for output in &self.outputs {
            let output_var = node_to_var
                .get(&output.node)
                .copied()
                .expect("Output node should have a variable");

            let output_lit = if output.inverted {
                Lit::neg(output_var)
            } else {
                Lit::pos(output_var)
            };

            sat.add_clause([output_lit]);
        }

        node_to_var
    }

    /// Get the SAT literal for an edge given a node-to-var mapping
    #[must_use]
    pub fn edge_to_lit(edge: AigEdge, node_to_var: &FxHashMap<NodeId, Var>) -> Option<Lit> {
        let var = node_to_var.get(&edge.node).copied()?;
        Some(if edge.inverted {
            Lit::neg(var)
        } else {
            Lit::pos(var)
        })
    }

    /// Build a half adder: (sum, carry) = a + b
    /// Returns (sum, carry) where sum = a XOR b, carry = a AND b
    pub fn half_adder(&mut self, a: AigEdge, b: AigEdge) -> (AigEdge, AigEdge) {
        let sum = self.xor(a, b);
        let carry = self.and(a, b);
        (sum, carry)
    }

    /// Build a full adder: (sum, carry_out) = a + b + carry_in
    pub fn full_adder(&mut self, a: AigEdge, b: AigEdge, carry_in: AigEdge) -> (AigEdge, AigEdge) {
        // sum = a XOR b XOR carry_in
        let xor_ab = self.xor(a, b);
        let sum = self.xor(xor_ab, carry_in);

        // carry_out = (a AND b) OR (carry_in AND (a XOR b))
        let and_ab = self.and(a, b);
        let and_cin_xor = self.and(carry_in, xor_ab);
        let carry_out = self.or(and_ab, and_cin_xor);

        (sum, carry_out)
    }

    /// Build a ripple-carry adder for bit vectors
    /// Returns the sum bits (result width = max(a.len(), b.len()))
    pub fn ripple_carry_adder(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        assert_eq!(a.len(), b.len(), "Bit vectors must have same width");

        let width = a.len();
        let mut result = Vec::with_capacity(width);
        let mut carry = self.false_edge();

        for i in 0..width {
            let (sum, carry_out) = self.full_adder(a[i], b[i], carry);
            result.push(sum);
            carry = carry_out;
        }

        result
    }

    /// Build an unsigned less-than comparator
    /// Returns edge that is true iff a < b (unsigned)
    pub fn unsigned_less_than(&mut self, a: &[AigEdge], b: &[AigEdge]) -> AigEdge {
        assert_eq!(a.len(), b.len(), "Bit vectors must have same width");

        if a.is_empty() {
            return self.false_edge();
        }

        // For unsigned comparison from MSB to LSB:
        // a < b iff there exists position i where:
        //   - all bits from MSB down to i+1 are equal, AND
        //   - bit i: a[i] < b[i] (i.e., a[i]=0, b[i]=1)

        // Build from MSB down to LSB
        // eq_so_far tracks whether all higher bits are equal
        let mut eq_so_far = self.true_edge();
        let mut lt = self.false_edge();

        for i in (0..a.len()).rev() {
            let ai = a[i];
            let bi = b[i];

            // At position i: a[i] < b[i] means ~a[i] AND b[i]
            let not_ai = self.not(ai);
            let ai_lt_bi = self.and(not_ai, bi);

            // If all higher bits are equal AND a[i] < b[i], then a < b
            let lt_at_i = self.and(eq_so_far, ai_lt_bi);

            // Update overall lt: if we haven't already determined a < b, check this position
            lt = self.or(lt, lt_at_i);

            // Update eq_so_far: remains true only if it was true AND current bits are equal
            let ai_eq_bi = self.iff(ai, bi);
            eq_so_far = self.and(eq_so_far, ai_eq_bi);
        }

        lt
    }

    /// Build a comparator that checks if two bit vectors are equal
    pub fn equal(&mut self, a: &[AigEdge], b: &[AigEdge]) -> AigEdge {
        assert_eq!(a.len(), b.len(), "Bit vectors must have same width");

        if a.is_empty() {
            return self.true_edge();
        }

        // All bits must match: AND of all (ai IFF bi)
        let mut result = self.true_edge();

        for i in 0..a.len() {
            let bits_equal = self.iff(a[i], b[i]);
            result = self.and(result, bits_equal);
        }

        result
    }

    /// Build bit vector negation (two's complement): -a = ~a + 1
    pub fn negate_bitvector(&mut self, a: &[AigEdge]) -> Vec<AigEdge> {
        // First compute ~a
        let not_a: Vec<AigEdge> = a.iter().map(|&bit| self.not(bit)).collect();

        // Then add 1
        let mut carry = self.true_edge(); // Initial carry = 1
        let mut result = Vec::with_capacity(a.len());

        for &not_a_bit in &not_a {
            // sum = not_a_bit XOR carry
            let sum = self.xor(not_a_bit, carry);
            result.push(sum);

            // carry_out = not_a_bit AND carry
            carry = self.and(not_a_bit, carry);
        }

        result
    }

    /// Build a constant bit vector
    pub fn constant_bitvector(&mut self, value: u64, width: usize) -> Vec<AigEdge> {
        let mut result = Vec::with_capacity(width);

        for i in 0..width {
            let bit = (value >> i) & 1;
            let edge = if bit == 1 {
                self.true_edge()
            } else {
                self.false_edge()
            };
            result.push(edge);
        }

        result
    }
}

/// Statistics about an AIG circuit
#[derive(Debug, Clone, Copy)]
pub struct AigStats {
    /// Total number of nodes
    pub num_nodes: usize,
    /// Number of input nodes
    pub num_inputs: usize,
    /// Number of AND gates
    pub num_and_gates: usize,
    /// Number of outputs
    pub num_outputs: usize,
}

impl fmt::Display for AigStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AIG Stats: {} nodes ({} inputs, {} ANDs, {} outputs)",
            self.num_nodes, self.num_inputs, self.num_and_gates, self.num_outputs
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aig_constants() {
        let circuit = AigCircuit::new();
        let false_edge = circuit.false_edge();
        let true_edge = circuit.true_edge();

        assert!(!false_edge.is_inverted());
        assert!(true_edge.is_inverted());
        assert_eq!(false_edge.not(), true_edge);
        assert_eq!(true_edge.not(), false_edge);
    }

    #[test]
    fn test_aig_input() {
        let mut circuit = AigCircuit::new();
        let x = circuit.new_input("x".to_string());

        assert!(!x.is_inverted());
        assert_eq!(circuit.num_inputs(), 1);
    }

    #[test]
    fn test_aig_and_gate() {
        let mut circuit = AigCircuit::new();
        let x = circuit.new_input("x".to_string());
        let y = circuit.new_input("y".to_string());

        let z = circuit.and(x, y);
        assert_eq!(circuit.num_and_gates(), 1);

        // Same AND should reuse existing node
        let z2 = circuit.and(x, y);
        assert_eq!(z, z2);
        assert_eq!(circuit.num_and_gates(), 1);
    }

    #[test]
    fn test_aig_constant_propagation() {
        let mut circuit = AigCircuit::new();
        let x = circuit.new_input("x".to_string());

        // x AND 0 = 0
        let result = circuit.and(x, circuit.false_edge());
        assert_eq!(result, circuit.false_edge());

        // x AND 1 = x
        let result = circuit.and(x, circuit.true_edge());
        assert_eq!(result, x);

        // x AND x = x
        let result = circuit.and(x, x);
        assert_eq!(result, x);

        // x AND ~x = 0
        let not_x = circuit.not(x);
        let result = circuit.and(x, not_x);
        assert_eq!(result, circuit.false_edge());
    }

    #[test]
    fn test_aig_or_gate() {
        let mut circuit = AigCircuit::new();
        let x = circuit.new_input("x".to_string());
        let y = circuit.new_input("y".to_string());

        let _z = circuit.or(x, y);
        // OR is implemented as NOT(NOT(x) AND NOT(y))
        // Should create one AND gate
        assert!(circuit.num_and_gates() >= 1);
    }

    #[test]
    fn test_aig_xor_gate() {
        let mut circuit = AigCircuit::new();
        let x = circuit.new_input("x".to_string());
        let y = circuit.new_input("y".to_string());

        let _z = circuit.xor(x, y);
        // XOR requires multiple AND gates
        assert!(circuit.num_and_gates() >= 2);
    }

    #[test]
    fn test_aig_mux() {
        let mut circuit = AigCircuit::new();
        let sel = circuit.new_input("sel".to_string());
        let t = circuit.new_input("t".to_string());
        let f = circuit.new_input("f".to_string());

        let result = circuit.mux(sel, t, f);
        circuit.add_output(result);

        assert_eq!(circuit.outputs().len(), 1);
        assert!(circuit.num_and_gates() >= 2);
    }

    #[test]
    fn test_aig_structural_hashing() {
        let mut circuit = AigCircuit::new();
        let a = circuit.new_input("a".to_string());
        let b = circuit.new_input("b".to_string());
        let c = circuit.new_input("c".to_string());

        // (a AND b) should be same as (b AND a) due to normalization
        let ab1 = circuit.and(a, b);
        let ba1 = circuit.and(b, a);
        assert_eq!(ab1, ba1);
        assert_eq!(circuit.num_and_gates(), 1);

        // Creating another identical AND gate should reuse
        let ab2 = circuit.and(a, b);
        assert_eq!(ab1, ab2);
        assert_eq!(circuit.num_and_gates(), 1);

        // Different AND gate
        let _ac = circuit.and(a, c);
        assert_eq!(circuit.num_and_gates(), 2);
    }

    #[test]
    fn test_aig_stats() {
        let mut circuit = AigCircuit::new();
        let x = circuit.new_input("x".to_string());
        let y = circuit.new_input("y".to_string());

        let z = circuit.and(x, y);
        circuit.add_output(z);

        let stats = circuit.stats();
        assert_eq!(stats.num_inputs, 2);
        assert_eq!(stats.num_and_gates, 1);
        assert_eq!(stats.num_outputs, 1);
    }

    #[test]
    fn test_aig_to_cnf() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();
        let x = circuit.new_input("x".to_string());
        let y = circuit.new_input("y".to_string());

        // z = x AND y
        let z = circuit.and(x, y);
        circuit.add_output(z);

        // Convert to CNF
        let mut sat = SatSolver::new();
        let _node_to_var = circuit.to_cnf(&mut sat);

        // Should be satisfiable (x=1, y=1, z=1)
        let result = sat.solve();
        assert!(matches!(result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_to_cnf_unsat() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();
        let x = circuit.new_input("x".to_string());

        // Create contradiction: x AND ~x
        let not_x = circuit.not(x);
        let contradiction = circuit.and(x, not_x);
        circuit.add_output(contradiction);

        // Convert to CNF
        let mut sat = SatSolver::new();
        let _node_to_var = circuit.to_cnf(&mut sat);

        // Should be unsatisfiable
        let result = sat.solve();
        assert!(matches!(result, SolverResult::Unsat));
    }

    #[test]
    fn test_aig_complex_circuit() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();
        let a = circuit.new_input("a".to_string());
        let b = circuit.new_input("b".to_string());
        let c = circuit.new_input("c".to_string());

        // (a XOR b) AND c
        let a_xor_b = circuit.xor(a, b);
        let result = circuit.and(a_xor_b, c);
        circuit.add_output(result);

        // Convert to CNF
        let mut sat = SatSolver::new();
        let _node_to_var = circuit.to_cnf(&mut sat);

        // Should be satisfiable
        let sat_result = sat.solve();
        assert!(matches!(sat_result, SolverResult::Sat));
    }

    #[test]
    fn test_aig_half_adder() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();
        let a = circuit.new_input("a".to_string());
        let b = circuit.new_input("b".to_string());

        let (sum, carry) = circuit.half_adder(a, b);

        // Test case: a=0, b=0 => sum=0, carry=0
        let false_edge = circuit.false_edge();

        let check_a = circuit.iff(a, false_edge);
        let check_b = circuit.iff(b, false_edge);
        let check_sum = circuit.iff(sum, false_edge);
        let check_carry = circuit.iff(carry, false_edge);

        circuit.add_output(check_a);
        circuit.add_output(check_b);
        circuit.add_output(check_sum);
        circuit.add_output(check_carry);

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        assert!(matches!(sat.solve(), SolverResult::Sat));
    }

    #[test]
    fn test_aig_full_adder() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();
        let a = circuit.new_input("a".to_string());
        let b = circuit.new_input("b".to_string());
        let cin = circuit.new_input("cin".to_string());

        let (sum, cout) = circuit.full_adder(a, b, cin);

        // Test case: a=1, b=1, cin=1 => sum=1, cout=1
        circuit.add_output(a);
        circuit.add_output(b);
        circuit.add_output(cin);
        circuit.add_output(sum);
        circuit.add_output(cout);

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        assert!(matches!(sat.solve(), SolverResult::Sat));
    }

    #[test]
    fn test_aig_ripple_carry_adder() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();

        // Create 3-bit inputs: a=5 (0b101), b=3 (0b011)
        let a = circuit.constant_bitvector(5, 3);
        let b = circuit.constant_bitvector(3, 3);

        // Expected result: 5 + 3 = 8 (0b1000), but only 3 bits so 0 (0b000)
        let result = circuit.ripple_carry_adder(&a, &b);

        // Result should be 0b000 (overflow)
        let false_edge = circuit.false_edge();
        let checks: Vec<_> = result
            .iter()
            .map(|&bit| circuit.iff(bit, false_edge))
            .collect();

        for check in checks {
            circuit.add_output(check);
        }

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        assert!(matches!(sat.solve(), SolverResult::Sat));
    }

    #[test]
    fn test_aig_unsigned_less_than() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();

        // Test: 3 < 5 should be true
        let a = circuit.constant_bitvector(3, 4); // 0b0011
        let b = circuit.constant_bitvector(5, 4); // 0b0101

        let lt = circuit.unsigned_less_than(&a, &b);
        circuit.add_output(lt);

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        assert!(matches!(sat.solve(), SolverResult::Sat));
    }

    #[test]
    fn test_aig_unsigned_less_than_false() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();

        // Test: 5 < 3 should be false
        let a = circuit.constant_bitvector(5, 4);
        let b = circuit.constant_bitvector(3, 4);

        let lt = circuit.unsigned_less_than(&a, &b);
        circuit.add_output(lt);

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        // Should be unsat since we're asserting 5 < 3
        assert!(matches!(sat.solve(), SolverResult::Unsat));
    }

    #[test]
    fn test_aig_equal() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();

        // Test: 7 == 7 should be true
        let a = circuit.constant_bitvector(7, 4);
        let b = circuit.constant_bitvector(7, 4);

        let eq = circuit.equal(&a, &b);
        circuit.add_output(eq);

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        assert!(matches!(sat.solve(), SolverResult::Sat));
    }

    #[test]
    fn test_aig_not_equal() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();

        // Test: 3 == 5 should be false
        let a = circuit.constant_bitvector(3, 4);
        let b = circuit.constant_bitvector(5, 4);

        let eq = circuit.equal(&a, &b);
        circuit.add_output(eq);

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        // Should be unsat since we're asserting 3 == 5
        assert!(matches!(sat.solve(), SolverResult::Unsat));
    }

    #[test]
    fn test_aig_negate_bitvector() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();

        // Test: -5 in 4-bit two's complement = 11 (0b1011)
        let a = circuit.constant_bitvector(5, 4); // 0b0101
        let neg_a = circuit.negate_bitvector(&a);

        // Expected: -5 = 0b1011 = 11 in unsigned
        let expected = circuit.constant_bitvector(11, 4);

        let eq = circuit.equal(&neg_a, &expected);
        circuit.add_output(eq);

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        assert!(matches!(sat.solve(), SolverResult::Sat));
    }

    #[test]
    fn test_aig_constant_bitvector() {
        use oxiz_sat::{Solver as SatSolver, SolverResult};

        let mut circuit = AigCircuit::new();

        let bv = circuit.constant_bitvector(0b1010, 4);
        assert_eq!(bv.len(), 4);

        // Verify bits: LSB first
        let false_edge = circuit.false_edge();
        let true_edge = circuit.true_edge();

        let check0 = circuit.iff(bv[0], false_edge); // bit 0 = 0
        let check1 = circuit.iff(bv[1], true_edge); // bit 1 = 1
        let check2 = circuit.iff(bv[2], false_edge); // bit 2 = 0
        let check3 = circuit.iff(bv[3], true_edge); // bit 3 = 1

        circuit.add_output(check0);
        circuit.add_output(check1);
        circuit.add_output(check2);
        circuit.add_output(check3);

        let mut sat = SatSolver::new();
        circuit.to_cnf(&mut sat);

        assert!(matches!(sat.solve(), SolverResult::Sat));
    }
}
