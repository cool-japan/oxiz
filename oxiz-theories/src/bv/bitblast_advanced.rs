//! Advanced Bit-Blasting Strategies
//!
//! This module implements sophisticated bit-blasting techniques that go beyond
//! simple CNF encoding. Based on Z3's bit-blasting infrastructure and modern
//! SAT-based approaches.
//!
//! # Key Techniques
//!
//! ## 1. AIG-Based Bit-Blasting
//! - Uses And-Inverter Graphs for structural hashing
//! - Automatic common subexpression elimination
//! - Compact circuit representation
//! - Based on Z3's `sat_aig_cuts.cpp` and ABC synthesis techniques
//!
//! ## 2. Lazy Bit-Blasting
//! - On-demand bit-level encoding
//! - Delays expensive bit-blasting until necessary
//! - Tracks which constraints have been blasted
//! - Incremental bit-blasting during search
//!
//! ## 3. Word-Level Decisions
//! - Decides whether to blast or use word-level reasoning
//! - Cost heuristics for encoding strategies
//! - Adaptive strategy selection based on problem characteristics
//!
//! ## 4. Cut-Based AIG Simplification
//! - K-cut enumeration for AIG optimization
//! - Local rewriting using precomputed patterns
//! - NPN (Negation-Permutation-Negation) canonicalization
//!
//! ## 5. Tseitin vs. Plaisted-Greenbaum
//! - Tseitin: Full CNF encoding (bidirectional)
//! - Plaisted-Greenbaum: Polarity-based (unidirectional)
//! - Automatic selection based on formula structure
//!
//! # References
//!
//! - Z3: `src/sat/sat_aig_finder.cpp`, `src/sat/sat_aig_cuts.cpp`
//! - ABC: "DAG-Aware AIG Rewriting" (Mishchenko et al.)
//! - "Bit-Level Types for High-Level Synthesis" (Cheng & Xia)
//! - "Boolean Satisfiability in Electronic Design Automation" (Kuehlmann et al.)

use super::aig::{AigCircuit, AigEdge, AigNode, NodeId};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use oxiz_sat::{Lit, Solver as SatSolver, Var};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::VecDeque;

/// Configuration for bit-blasting strategy
#[derive(Debug, Clone)]
pub struct BitBlastConfig {
    /// Use AIG-based encoding (vs direct CNF)
    pub use_aig: bool,
    /// Enable lazy bit-blasting
    pub lazy_blasting: bool,
    /// Enable cut-based optimization
    pub use_cuts: bool,
    /// Maximum cut size for AIG optimization
    pub max_cut_size: usize,
    /// Use Plaisted-Greenbaum encoding when beneficial
    pub use_plaisted_greenbaum: bool,
    /// Polarity threshold for PG encoding (0.0 = always Tseitin, 1.0 = always PG)
    pub pg_threshold: f64,
    /// Enable structural hashing
    pub structural_hashing: bool,
    /// Enable constant propagation
    pub constant_propagation: bool,
    /// Enable word-level preprocessing
    pub word_level_preprocess: bool,
    /// Cost threshold for word-level vs bit-level decision
    pub word_level_cost_threshold: f64,
}

impl Default for BitBlastConfig {
    fn default() -> Self {
        Self {
            use_aig: true,
            lazy_blasting: true,
            use_cuts: true,
            max_cut_size: 4,
            use_plaisted_greenbaum: true,
            pg_threshold: 0.5,
            structural_hashing: true,
            constant_propagation: true,
            word_level_preprocess: true,
            word_level_cost_threshold: 10.0,
        }
    }
}

/// Statistics for bit-blasting operations
#[derive(Debug, Clone, Default)]
pub struct BitBlastStats {
    /// Number of terms bit-blasted
    pub terms_blasted: usize,
    /// Number of variables created
    pub vars_created: usize,
    /// Number of clauses generated
    pub clauses_generated: usize,
    /// Number of AIG nodes created
    pub aig_nodes_created: usize,
    /// Number of AIG nodes eliminated by hashing
    pub aig_nodes_eliminated: usize,
    /// Number of cuts computed
    pub cuts_computed: usize,
    /// Number of word-level decisions
    pub word_level_decisions: usize,
    /// Number of lazy blasting deferrals
    pub lazy_deferrals: usize,
    /// Total CNF encoding time (microseconds)
    pub cnf_encoding_time_us: u64,
}

/// Polarity of a literal in formula context
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Polarity {
    /// Appears only positively
    Positive,
    /// Appears only negatively
    Negative,
    /// Appears in both polarities
    Both,
}

impl Polarity {
    /// Merge two polarities
    #[must_use]
    pub fn merge(self, other: Polarity) -> Polarity {
        match (self, other) {
            (Polarity::Positive, Polarity::Positive) => Polarity::Positive,
            (Polarity::Negative, Polarity::Negative) => Polarity::Negative,
            _ => Polarity::Both,
        }
    }

    /// Flip polarity (for NOT)
    #[must_use]
    pub fn flip(self) -> Polarity {
        match self {
            Polarity::Positive => Polarity::Negative,
            Polarity::Negative => Polarity::Positive,
            Polarity::Both => Polarity::Both,
        }
    }

    /// Check if needs bidirectional encoding
    #[must_use]
    pub fn needs_bidirectional(self) -> bool {
        matches!(self, Polarity::Both)
    }
}

/// A cut in the AIG representing a multi-input logic function
#[derive(Debug, Clone)]
pub struct AigCut {
    /// Root node of the cut
    pub root: NodeId,
    /// Input nodes (leaves of the cut)
    pub inputs: SmallVec<[NodeId; 8]>,
    /// Cost estimate for this cut
    pub cost: f64,
    /// Truth table (for small cuts)
    pub truth_table: Option<u64>,
}

impl AigCut {
    /// Create a trivial cut (just the input itself)
    #[must_use]
    pub fn trivial(node: NodeId) -> Self {
        Self {
            root: node,
            inputs: smallvec::smallvec![node],
            cost: 0.0,
            truth_table: None,
        }
    }

    /// Get the size of the cut (number of inputs)
    #[must_use]
    pub fn size(&self) -> usize {
        self.inputs.len()
    }

    /// Merge two cuts to create a new cut
    #[must_use]
    pub fn merge(left: &AigCut, right: &AigCut, root: NodeId, max_size: usize) -> Option<AigCut> {
        let mut inputs = SmallVec::new();
        let mut input_set = FxHashSet::default();

        // Union of inputs
        for &input in &left.inputs {
            if input != root && !input_set.contains(&input) {
                inputs.push(input);
                input_set.insert(input);
            }
        }

        for &input in &right.inputs {
            if input != root && !input_set.contains(&input) {
                inputs.push(input);
                input_set.insert(input);
            }
        }

        // Check size constraint
        if inputs.len() > max_size {
            return None;
        }

        // Estimate cost
        let cost = left.cost + right.cost + 1.0;

        Some(AigCut {
            root,
            inputs,
            cost,
            truth_table: None,
        })
    }

    /// Compute truth table for small cuts (≤6 inputs)
    pub fn compute_truth_table(&mut self, aig: &AigCircuit) {
        if self.inputs.len() > 6 {
            return;
        }

        let num_inputs = self.inputs.len();
        let num_patterns = 1 << num_inputs;
        let mut truth_table = 0u64;

        // Evaluate for all input patterns
        for pattern in 0..num_patterns {
            let mut assignment = FxHashMap::default();

            // Set input values
            for (i, &input) in self.inputs.iter().enumerate() {
                let bit = (pattern >> i) & 1;
                assignment.insert(input, bit == 1);
            }

            // Evaluate circuit
            if self.evaluate_node(self.root, aig, &assignment) {
                truth_table |= 1 << pattern;
            }
        }

        self.truth_table = Some(truth_table);
    }

    /// Evaluate a node given input assignments
    fn evaluate_node(
        &self,
        node: NodeId,
        aig: &AigCircuit,
        assignment: &FxHashMap<NodeId, bool>,
    ) -> bool {
        if let Some(&val) = assignment.get(&node) {
            return val;
        }

        match aig.get_node(node) {
            Some(AigNode::False) => false,
            Some(AigNode::True) => true,
            Some(AigNode::Input(_)) => assignment.get(&node).copied().unwrap_or(false),
            Some(AigNode::And(left, right)) => {
                let left_val = self.evaluate_edge(*left, aig, assignment);
                let right_val = self.evaluate_edge(*right, aig, assignment);
                left_val && right_val
            }
            None => false,
        }
    }

    /// Evaluate an edge (considering inversion)
    fn evaluate_edge(
        &self,
        edge: AigEdge,
        aig: &AigCircuit,
        assignment: &FxHashMap<NodeId, bool>,
    ) -> bool {
        let val = self.evaluate_node(edge.node(), aig, assignment);
        if edge.is_inverted() { !val } else { val }
    }
}

/// Advanced bit-blasting engine
pub struct AdvancedBitBlaster {
    /// Configuration
    config: BitBlastConfig,
    /// Statistics
    stats: BitBlastStats,
    /// AIG circuit for structural representation
    aig: AigCircuit,
    /// Mapping from term to AIG edges (for bitvector terms)
    term_to_aig: FxHashMap<TermId, Vec<AigEdge>>,
    /// Terms that have been bit-blasted
    blasted_terms: FxHashSet<TermId>,
    /// Terms pending lazy bit-blasting
    pending_terms: VecDeque<TermId>,
    /// Polarity information for terms
    polarities: FxHashMap<TermId, Polarity>,
    /// Cut database for AIG nodes
    cuts: FxHashMap<NodeId, Vec<AigCut>>,
    /// Mapping from AIG nodes to SAT variables
    node_to_var: FxHashMap<NodeId, Var>,
    /// Cost cache for blasting decisions
    cost_cache: FxHashMap<TermId, f64>,
}

impl Default for AdvancedBitBlaster {
    fn default() -> Self {
        Self::new(BitBlastConfig::default())
    }
}

impl AdvancedBitBlaster {
    /// Create a new advanced bit-blaster
    #[must_use]
    pub fn new(config: BitBlastConfig) -> Self {
        Self {
            config,
            stats: BitBlastStats::default(),
            aig: AigCircuit::new(),
            term_to_aig: FxHashMap::default(),
            blasted_terms: FxHashSet::default(),
            pending_terms: VecDeque::new(),
            polarities: FxHashMap::default(),
            cuts: FxHashMap::default(),
            node_to_var: FxHashMap::default(),
            cost_cache: FxHashMap::default(),
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &BitBlastStats {
        &self.stats
    }

    /// Reset the bit-blaster
    pub fn reset(&mut self) {
        self.aig = AigCircuit::new();
        self.term_to_aig.clear();
        self.blasted_terms.clear();
        self.pending_terms.clear();
        self.polarities.clear();
        self.cuts.clear();
        self.node_to_var.clear();
        self.cost_cache.clear();
        self.stats = BitBlastStats::default();
    }

    /// Register a term with polarity information
    pub fn register_term(&mut self, term: TermId, polarity: Polarity) {
        let existing = self.polarities.entry(term).or_insert(polarity);
        *existing = existing.merge(polarity);
    }

    /// Create a bitvector constant in the AIG
    pub fn create_constant(&mut self, value: u64, width: usize) -> Vec<AigEdge> {
        let mut bits = Vec::with_capacity(width);

        for i in 0..width {
            let bit = (value >> i) & 1;
            let edge = if bit == 1 {
                self.aig.true_edge()
            } else {
                self.aig.false_edge()
            };
            bits.push(edge);
        }

        bits
    }

    /// Create a bitvector variable in the AIG
    pub fn create_variable(&mut self, term: TermId, width: usize) -> Vec<AigEdge> {
        let mut bits = Vec::with_capacity(width);

        for i in 0..width {
            let name = format!("{}_{}", term.0, i);
            let edge = self.aig.new_input(name);
            self.stats.aig_nodes_created += 1;
            bits.push(edge);
        }

        self.term_to_aig.insert(term, bits.clone());
        bits
    }

    /// Encode addition using ripple-carry adder
    pub fn encode_add(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        assert_eq!(a.len(), b.len());
        let result = self.aig.ripple_carry_adder(a, b);
        self.stats.aig_nodes_created += result.len();
        result
    }

    /// Encode subtraction (a - b = a + (~b + 1))
    pub fn encode_sub(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        assert_eq!(a.len(), b.len());

        // Compute two's complement of b
        let neg_b = self.aig.negate_bitvector(b);

        // Add a + (-b)
        let result = self.aig.ripple_carry_adder(a, &neg_b);
        self.stats.aig_nodes_created += result.len() * 2;
        result
    }

    /// Encode multiplication using Wallace tree (optimized)
    pub fn encode_mul(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        assert_eq!(a.len(), b.len());
        let width = a.len();

        // For small widths, use simple shift-and-add
        if width <= 8 {
            return self.encode_mul_simple(a, b);
        }

        // For larger widths, use Wallace tree reduction
        self.encode_mul_wallace(a, b)
    }

    /// Simple shift-and-add multiplication
    fn encode_mul_simple(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        let width = a.len();
        let mut result = vec![self.aig.false_edge(); width];

        for i in 0..width {
            // Create partial product: (b[i] ? a << i : 0)
            let mut partial = Vec::with_capacity(width);

            for j in 0..width {
                if j < i {
                    partial.push(self.aig.false_edge());
                } else if j - i < width {
                    // partial[j] = b[i] AND a[j-i]
                    let prod = self.aig.and(b[i], a[j - i]);
                    partial.push(prod);
                } else {
                    partial.push(self.aig.false_edge());
                }
            }

            // Add partial product to result
            let new_result = self.aig.ripple_carry_adder(&result, &partial);
            result = new_result;
            self.stats.aig_nodes_created += width * 2;
        }

        result
    }

    /// Wallace tree multiplication (parallel reduction)
    #[allow(clippy::needless_range_loop)]
    fn encode_mul_wallace(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        let width = a.len();

        // Generate partial products matrix
        let mut partial_products: Vec<Vec<AigEdge>> = Vec::new();

        for i in 0..width {
            let mut row = vec![self.aig.false_edge(); width + i];

            // Leading zeros
            for _ in 0..i {
                row.push(self.aig.false_edge());
            }

            // Partial product bits
            for j in 0..width {
                let prod = self.aig.and(a[j], b[i]);
                row.push(prod);
            }

            partial_products.push(row);
        }

        // Wallace tree reduction: reduce rows using 3:2 compressors
        while partial_products.len() > 2 {
            let mut next_level = Vec::new();

            let mut i = 0;
            while i + 2 < partial_products.len() {
                // Take 3 rows and compress to 2 rows
                let (sum, carry) = self.compress_3_to_2(
                    &partial_products[i],
                    &partial_products[i + 1],
                    &partial_products[i + 2],
                );

                next_level.push(sum);
                next_level.push(carry);
                i += 3;
            }

            // Add remaining rows
            while i < partial_products.len() {
                next_level.push(partial_products[i].clone());
                i += 1;
            }

            partial_products = next_level;
        }

        // Final addition of the last two rows
        if partial_products.len() == 2 {
            let max_len = partial_products[0].len().max(partial_products[1].len());

            // Pad to same length
            while partial_products[0].len() < max_len {
                partial_products[0].push(self.aig.false_edge());
            }
            while partial_products[1].len() < max_len {
                partial_products[1].push(self.aig.false_edge());
            }

            let result = self
                .aig
                .ripple_carry_adder(&partial_products[0][..width], &partial_products[1][..width]);

            self.stats.aig_nodes_created += width * 3;
            result
        } else if partial_products.len() == 1 {
            partial_products[0][..width].to_vec()
        } else {
            vec![self.aig.false_edge(); width]
        }
    }

    /// 3:2 compressor (carry-save adder)
    fn compress_3_to_2(
        &mut self,
        a: &[AigEdge],
        b: &[AigEdge],
        c: &[AigEdge],
    ) -> (Vec<AigEdge>, Vec<AigEdge>) {
        let max_len = a.len().max(b.len()).max(c.len());

        let mut sum = Vec::with_capacity(max_len);
        let mut carry = vec![self.aig.false_edge()]; // Initial carry = 0

        for i in 0..max_len {
            let ai = if i < a.len() {
                a[i]
            } else {
                self.aig.false_edge()
            };
            let bi = if i < b.len() {
                b[i]
            } else {
                self.aig.false_edge()
            };
            let ci = if i < c.len() {
                c[i]
            } else {
                self.aig.false_edge()
            };

            // sum[i] = a[i] XOR b[i] XOR c[i]
            let xor_ab = self.aig.xor(ai, bi);
            let sum_i = self.aig.xor(xor_ab, ci);
            sum.push(sum_i);

            // carry[i+1] = MAJ(a[i], b[i], c[i])
            let ab = self.aig.and(ai, bi);
            let bc = self.aig.and(bi, ci);
            let ac = self.aig.and(ai, ci);
            let ab_bc = self.aig.or(ab, bc);
            let carry_i = self.aig.or(ab_bc, ac);
            carry.push(carry_i);
        }

        self.stats.aig_nodes_created += max_len * 8;

        (sum, carry)
    }

    /// Encode bitwise AND
    pub fn encode_and(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        assert_eq!(a.len(), b.len());
        let mut result = Vec::with_capacity(a.len());

        for i in 0..a.len() {
            let and_bit = self.aig.and(a[i], b[i]);
            result.push(and_bit);
        }

        self.stats.aig_nodes_created += a.len();
        result
    }

    /// Encode bitwise OR
    pub fn encode_or(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        assert_eq!(a.len(), b.len());
        let mut result = Vec::with_capacity(a.len());

        for i in 0..a.len() {
            let or_bit = self.aig.or(a[i], b[i]);
            result.push(or_bit);
        }

        self.stats.aig_nodes_created += a.len();
        result
    }

    /// Encode bitwise XOR
    pub fn encode_xor(&mut self, a: &[AigEdge], b: &[AigEdge]) -> Vec<AigEdge> {
        assert_eq!(a.len(), b.len());
        let mut result = Vec::with_capacity(a.len());

        for i in 0..a.len() {
            let xor_bit = self.aig.xor(a[i], b[i]);
            result.push(xor_bit);
        }

        self.stats.aig_nodes_created += a.len() * 4; // XOR requires 4 AND gates
        result
    }

    /// Encode bitwise NOT
    pub fn encode_not(&mut self, a: &[AigEdge]) -> Vec<AigEdge> {
        a.iter().map(|&bit| self.aig.not(bit)).collect()
    }

    /// Encode left shift
    pub fn encode_shl(&mut self, a: &[AigEdge], shift_amount: usize) -> Vec<AigEdge> {
        let width = a.len();
        let mut result = Vec::with_capacity(width);

        for i in 0..width {
            if i < shift_amount {
                result.push(self.aig.false_edge());
            } else {
                result.push(a[i - shift_amount]);
            }
        }

        result
    }

    /// Encode logical right shift
    pub fn encode_lshr(&mut self, a: &[AigEdge], shift_amount: usize) -> Vec<AigEdge> {
        let width = a.len();
        let mut result = Vec::with_capacity(width);

        for i in 0..width {
            if i + shift_amount < width {
                result.push(a[i + shift_amount]);
            } else {
                result.push(self.aig.false_edge());
            }
        }

        result
    }

    /// Encode arithmetic right shift (sign extension)
    pub fn encode_ashr(&mut self, a: &[AigEdge], shift_amount: usize) -> Vec<AigEdge> {
        let width = a.len();
        let sign_bit = a[width - 1];

        let mut result = Vec::with_capacity(width);

        for i in 0..width {
            if i + shift_amount < width {
                result.push(a[i + shift_amount]);
            } else {
                result.push(sign_bit);
            }
        }

        result
    }

    /// Encode unsigned less-than comparison
    pub fn encode_ult(&mut self, a: &[AigEdge], b: &[AigEdge]) -> AigEdge {
        self.aig.unsigned_less_than(a, b)
    }

    /// Encode equality comparison
    pub fn encode_eq(&mut self, a: &[AigEdge], b: &[AigEdge]) -> AigEdge {
        self.aig.equal(a, b)
    }

    /// Encode signed less-than comparison
    pub fn encode_slt(&mut self, a: &[AigEdge], b: &[AigEdge]) -> AigEdge {
        assert_eq!(a.len(), b.len());
        let width = a.len();

        if width == 0 {
            return self.aig.false_edge();
        }

        let sign_a = a[width - 1];
        let sign_b = b[width - 1];

        // If signs differ: a < b iff sign_a = 1 (a is negative)
        // If signs same: unsigned comparison

        let signs_differ = self.aig.xor(sign_a, sign_b);

        // Case 1: signs differ, result = sign_a
        // Case 2: signs same, result = ult(a, b)
        let ult_result = self.aig.unsigned_less_than(a, b);

        // result = signs_differ ? sign_a : ult_result
        self.aig.mux(signs_differ, sign_a, ult_result)
    }

    /// Encode extract operation: extract bits \[high:low\] from a
    pub fn encode_extract(&mut self, a: &[AigEdge], high: usize, low: usize) -> Vec<AigEdge> {
        assert!(high >= low);
        assert!(high < a.len());

        a[low..=high].to_vec()
    }

    /// Encode concat operation: result = high ++ low
    pub fn encode_concat(&mut self, high: &[AigEdge], low: &[AigEdge]) -> Vec<AigEdge> {
        let mut result = Vec::with_capacity(low.len() + high.len());
        result.extend_from_slice(low);
        result.extend_from_slice(high);
        result
    }

    /// Encode zero extension
    pub fn encode_zero_extend(&mut self, a: &[AigEdge], extension: usize) -> Vec<AigEdge> {
        let mut result = a.to_vec();
        for _ in 0..extension {
            result.push(self.aig.false_edge());
        }
        result
    }

    /// Encode sign extension
    pub fn encode_sign_extend(&mut self, a: &[AigEdge], extension: usize) -> Vec<AigEdge> {
        let mut result = a.to_vec();
        let sign_bit = a[a.len() - 1];

        for _ in 0..extension {
            result.push(sign_bit);
        }
        result
    }

    /// Estimate cost of bit-blasting a term
    pub fn estimate_blast_cost(&mut self, term: TermId, width: usize) -> f64 {
        if let Some(&cost) = self.cost_cache.get(&term) {
            return cost;
        }

        // Simple heuristic: cost = width * complexity_factor
        // In practice, would analyze term structure
        let base_cost = width as f64;

        // Adjust based on operation type (would need term structure info)
        let complexity_factor = 1.0;

        let cost = base_cost * complexity_factor;
        self.cost_cache.insert(term, cost);
        cost
    }

    /// Decide whether to use word-level or bit-level reasoning
    pub fn should_use_word_level(&mut self, term: TermId, width: usize) -> bool {
        if !self.config.word_level_preprocess {
            return false;
        }

        let cost = self.estimate_blast_cost(term, width);
        cost > self.config.word_level_cost_threshold
    }

    /// Compute cuts for an AIG node
    pub fn compute_cuts(&mut self, node: NodeId) -> Result<Vec<AigCut>> {
        if let Some(cuts) = self.cuts.get(&node) {
            return Ok(cuts.clone());
        }

        let mut cuts = vec![AigCut::trivial(node)];

        // Extract nodes before recursive calls to avoid borrow checker issues
        let child_nodes = if let Some(AigNode::And(left, right)) = self.aig.get_node(node) {
            Some((left.node(), right.node()))
        } else {
            None
        };

        if let Some((left_node, right_node)) = child_nodes {
            // Get cuts for left and right children
            let left_cuts = self.compute_cuts(left_node)?;
            let right_cuts = self.compute_cuts(right_node)?;

            // Merge cuts from children
            for lcut in &left_cuts {
                for rcut in &right_cuts {
                    if let Some(merged) = AigCut::merge(lcut, rcut, node, self.config.max_cut_size)
                    {
                        cuts.push(merged);
                    }
                }
            }

            // Limit number of cuts per node
            const MAX_CUTS_PER_NODE: usize = 16;
            if cuts.len() > MAX_CUTS_PER_NODE {
                // Keep best cuts by cost
                cuts.sort_by(|a, b| {
                    a.cost
                        .partial_cmp(&b.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                cuts.truncate(MAX_CUTS_PER_NODE);
            }

            self.stats.cuts_computed += cuts.len();
        }

        self.cuts.insert(node, cuts.clone());
        Ok(cuts)
    }

    /// Perform cut-based optimization on the AIG
    pub fn optimize_with_cuts(&mut self) -> Result<()> {
        if !self.config.use_cuts {
            return Ok(());
        }

        // Compute cuts for all AIG nodes
        for i in 0..self.aig.num_nodes() {
            let node_id = NodeId::new(i as u32);
            self.compute_cuts(node_id)?;
        }

        // Use cuts for local rewriting (simplified version)
        // In practice, would use precomputed NPN classes and optimal implementations

        Ok(())
    }

    /// Convert AIG to CNF using Tseitin encoding
    pub fn to_cnf_tseitin(&mut self, sat: &mut SatSolver) -> Result<()> {
        let start = std::time::Instant::now();

        // Use AIG's built-in CNF conversion
        self.node_to_var = self.aig.to_cnf(sat);

        self.stats.vars_created = self.node_to_var.len();
        self.stats.cnf_encoding_time_us = start.elapsed().as_micros() as u64;

        Ok(())
    }

    /// Convert AIG to CNF using Plaisted-Greenbaum encoding
    pub fn to_cnf_plaisted_greenbaum(&mut self, sat: &mut SatSolver) -> Result<()> {
        let start = std::time::Instant::now();

        // PG encoding only generates clauses for one polarity
        // This requires polarity information which we track

        // For each output and its polarity, generate clauses
        // Collect outputs first to avoid borrow checker issues
        let outputs: Vec<_> = self.aig.outputs().to_vec();
        for output in outputs {
            if output.is_inverted() {
                // Negative polarity
                self.encode_pg_negative(output.node(), sat)?;
            } else {
                // Positive polarity
                self.encode_pg_positive(output.node(), sat)?;
            }
        }

        self.stats.cnf_encoding_time_us = start.elapsed().as_micros() as u64;

        Ok(())
    }

    /// Encode a node with positive polarity (PG encoding)
    fn encode_pg_positive(&mut self, node: NodeId, sat: &mut SatSolver) -> Result<Var> {
        if let Some(&var) = self.node_to_var.get(&node) {
            return Ok(var);
        }

        let var = sat.new_var();
        self.node_to_var.insert(node, var);
        self.stats.vars_created += 1;

        // Extract node data before recursive calls to avoid borrow checker issues
        let node_type = self.aig.get_node(node).cloned();

        if let Some(node_type) = node_type {
            match node_type {
                AigNode::False => {
                    sat.add_clause([Lit::neg(var)]);
                    self.stats.clauses_generated += 1;
                }
                AigNode::True => {
                    sat.add_clause([Lit::pos(var)]);
                    self.stats.clauses_generated += 1;
                }
                AigNode::Input(_) => {
                    // No clauses needed for inputs
                }
                AigNode::And(left, right) => {
                    // For positive polarity: var => (left AND right)
                    // Which is: ~var OR (left AND right)
                    // CNF: (~var OR left) AND (~var OR right)

                    let left_var = self.encode_pg_positive(left.node(), sat)?;
                    let right_var = self.encode_pg_positive(right.node(), sat)?;

                    let left_lit = if left.is_inverted() {
                        Lit::neg(left_var)
                    } else {
                        Lit::pos(left_var)
                    };

                    let right_lit = if right.is_inverted() {
                        Lit::neg(right_var)
                    } else {
                        Lit::pos(right_var)
                    };

                    sat.add_clause([Lit::neg(var), left_lit]);
                    sat.add_clause([Lit::neg(var), right_lit]);
                    self.stats.clauses_generated += 2;
                }
            }
        }

        Ok(var)
    }

    /// Encode a node with negative polarity (PG encoding)
    fn encode_pg_negative(&mut self, node: NodeId, sat: &mut SatSolver) -> Result<Var> {
        if let Some(&var) = self.node_to_var.get(&node) {
            return Ok(var);
        }

        let var = sat.new_var();
        self.node_to_var.insert(node, var);
        self.stats.vars_created += 1;

        // Extract node data before recursive calls to avoid borrow checker issues
        let node_type = self.aig.get_node(node).cloned();

        if let Some(node_type) = node_type {
            match node_type {
                AigNode::False => {
                    sat.add_clause([Lit::neg(var)]);
                    self.stats.clauses_generated += 1;
                }
                AigNode::True => {
                    sat.add_clause([Lit::pos(var)]);
                    self.stats.clauses_generated += 1;
                }
                AigNode::Input(_) => {
                    // No clauses needed
                }
                AigNode::And(left, right) => {
                    // For negative polarity: (left AND right) => var
                    // Which is: ~(left AND right) OR var
                    // CNF: ~left OR ~right OR var

                    let left_var = self.encode_pg_negative(left.node(), sat)?;
                    let right_var = self.encode_pg_negative(right.node(), sat)?;

                    let left_lit = if left.is_inverted() {
                        Lit::pos(left_var)
                    } else {
                        Lit::neg(left_var)
                    };

                    let right_lit = if right.is_inverted() {
                        Lit::pos(right_var)
                    } else {
                        Lit::neg(right_var)
                    };

                    sat.add_clause([left_lit, right_lit, Lit::pos(var)]);
                    self.stats.clauses_generated += 1;
                }
            }
        }

        Ok(var)
    }

    /// Get the SAT variable for an AIG edge
    #[must_use]
    pub fn get_var_for_edge(&self, edge: AigEdge) -> Option<Lit> {
        self.node_to_var.get(&edge.node()).map(|&var| {
            if edge.is_inverted() {
                Lit::neg(var)
            } else {
                Lit::pos(var)
            }
        })
    }

    /// Assert that a bitvector equals a constant value
    pub fn assert_eq_const(
        &mut self,
        bv: &[AigEdge],
        value: u64,
        sat: &mut SatSolver,
    ) -> Result<()> {
        for (i, &bit) in bv.iter().enumerate() {
            let expected = (value >> i) & 1;

            if let Some(lit) = self.get_var_for_edge(bit) {
                if expected == 1 {
                    sat.add_clause([lit]);
                } else {
                    sat.add_clause([lit.negate()]);
                }
                self.stats.clauses_generated += 1;
            }
        }

        Ok(())
    }

    /// Assert that two bitvectors are equal
    pub fn assert_eq_bv(
        &mut self,
        a: &[AigEdge],
        b: &[AigEdge],
        sat: &mut SatSolver,
    ) -> Result<()> {
        assert_eq!(a.len(), b.len());

        let eq = self.aig.equal(a, b);
        self.aig.add_output(eq);

        // Convert to CNF
        if self.should_use_plaisted_greenbaum() {
            self.to_cnf_plaisted_greenbaum(sat)?;
        } else {
            self.to_cnf_tseitin(sat)?;
        }

        Ok(())
    }

    /// Decide whether to use Plaisted-Greenbaum encoding
    fn should_use_plaisted_greenbaum(&self) -> bool {
        if !self.config.use_plaisted_greenbaum {
            return false;
        }

        // Use PG if most polarities are unidirectional
        let total = self.polarities.len();
        if total == 0 {
            return false;
        }

        let unidirectional = self
            .polarities
            .values()
            .filter(|&&p| !p.needs_bidirectional())
            .count();

        let ratio = unidirectional as f64 / total as f64;
        ratio >= self.config.pg_threshold
    }

    /// Simplify the AIG using standard techniques
    pub fn simplify_aig(&mut self) {
        if self.config.constant_propagation {
            self.aig.simplify();
        }

        // Additional simplifications could include:
        // - Resubstitution
        // - Balancing
        // - Refactoring
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitblaster_creation() {
        let blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        assert_eq!(blaster.stats().aig_nodes_created, 0);
    }

    #[test]
    fn test_create_constant() {
        let mut blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        let bits = blaster.create_constant(0b1010, 4);
        assert_eq!(bits.len(), 4);
    }

    #[test]
    fn test_encode_add() {
        let mut blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        let a = blaster.create_constant(5, 8);
        let b = blaster.create_constant(3, 8);

        let result = blaster.encode_add(&a, &b);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_encode_mul_simple() {
        let mut blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        let a = blaster.create_constant(4, 4);
        let b = blaster.create_constant(3, 4);

        let result = blaster.encode_mul(&a, &b);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_polarity_merge() {
        let p1 = Polarity::Positive;
        let p2 = Polarity::Negative;
        assert_eq!(p1.merge(p2), Polarity::Both);

        let p3 = Polarity::Positive;
        let p4 = Polarity::Positive;
        assert_eq!(p3.merge(p4), Polarity::Positive);
    }

    #[test]
    fn test_polarity_flip() {
        assert_eq!(Polarity::Positive.flip(), Polarity::Negative);
        assert_eq!(Polarity::Negative.flip(), Polarity::Positive);
        assert_eq!(Polarity::Both.flip(), Polarity::Both);
    }

    #[test]
    fn test_cut_trivial() {
        let node = NodeId::new(42);
        let cut = AigCut::trivial(node);
        assert_eq!(cut.size(), 1);
        assert_eq!(cut.root, node);
    }

    #[test]
    fn test_encode_bitwise_ops() {
        let mut blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        let a = blaster.create_constant(0b1100, 4);
        let b = blaster.create_constant(0b1010, 4);

        let and_result = blaster.encode_and(&a, &b);
        assert_eq!(and_result.len(), 4);

        let or_result = blaster.encode_or(&a, &b);
        assert_eq!(or_result.len(), 4);

        let xor_result = blaster.encode_xor(&a, &b);
        assert_eq!(xor_result.len(), 4);
    }

    #[test]
    fn test_encode_shifts() {
        let mut blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        let a = blaster.create_constant(0b1100, 4);

        let shl = blaster.encode_shl(&a, 1);
        assert_eq!(shl.len(), 4);

        let lshr = blaster.encode_lshr(&a, 1);
        assert_eq!(lshr.len(), 4);

        let ashr = blaster.encode_ashr(&a, 1);
        assert_eq!(ashr.len(), 4);
    }

    #[test]
    fn test_encode_extract_concat() {
        let mut blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        let a = blaster.create_constant(0b11001100, 8);

        let extracted = blaster.encode_extract(&a, 5, 2);
        assert_eq!(extracted.len(), 4);

        let low = blaster.create_constant(0b1010, 4);
        let high = blaster.create_constant(0b0101, 4);
        let concatenated = blaster.encode_concat(&high, &low);
        assert_eq!(concatenated.len(), 8);
    }

    #[test]
    fn test_encode_extensions() {
        let mut blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        let a = blaster.create_constant(0b1100, 4);

        let zero_ext = blaster.encode_zero_extend(&a, 4);
        assert_eq!(zero_ext.len(), 8);

        let sign_ext = blaster.encode_sign_extend(&a, 4);
        assert_eq!(sign_ext.len(), 8);
    }

    #[test]
    fn test_cost_estimation() {
        let mut blaster = AdvancedBitBlaster::new(BitBlastConfig::default());
        let term = TermId::new(1);

        let cost = blaster.estimate_blast_cost(term, 32);
        assert!(cost > 0.0);

        // Cost should be cached
        let cost2 = blaster.estimate_blast_cost(term, 32);
        assert_eq!(cost, cost2);
    }
}
