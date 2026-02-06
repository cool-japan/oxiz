//! BitVector Theory Solver

use crate::config::BvConfig;
use crate::theory::{Theory, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use oxiz_sat::{Lit, Solver as SatSolver, SolverResult, Var};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// A bit vector variable (sequence of SAT variables)
#[derive(Debug, Clone)]
pub struct BvVar {
    /// SAT variables for each bit (LSB first)
    bits: SmallVec<[Var; 32]>,
    /// Width in bits
    width: u32,
}

/// Comparison tracking for conflict detection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ComparisonKey {
    a: TermId,
    b: TermId,
}

/// BitVector Theory Solver using bit-blasting
#[derive(Debug)]
pub struct BvSolver {
    /// Embedded SAT solver
    sat: SatSolver,
    /// Term to BV variable mapping
    term_to_bv: FxHashMap<TermId, BvVar>,
    /// Pending assertions
    assertions: Vec<(TermId, bool)>,
    /// Context stack
    context_stack: Vec<usize>,
    /// Configuration
    config: BvConfig,
    /// Track unsigned less-than comparisons for conflict detection
    /// Maps (a, b) -> SAT variable representing a < b
    ult_cache: FxHashMap<ComparisonKey, Var>,
}

impl Default for BvSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BvSolver {
    /// Create a new BitVector solver
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(BvConfig::default())
    }

    /// Create a new BitVector solver with custom configuration
    #[must_use]
    pub fn with_config(config: BvConfig) -> Self {
        Self {
            sat: SatSolver::new(),
            term_to_bv: FxHashMap::default(),
            assertions: Vec::new(),
            context_stack: Vec::new(),
            config,
            ult_cache: FxHashMap::default(),
        }
    }

    /// Create a new bit vector variable
    pub fn new_bv(&mut self, term: TermId, width: u32) -> &BvVar {
        if !self.term_to_bv.contains_key(&term) {
            let bits: SmallVec<[Var; 32]> = (0..width).map(|_| self.sat.new_var()).collect();
            self.term_to_bv.insert(term, BvVar { bits, width });
        }
        self.term_to_bv
            .get(&term)
            .expect("BvVar should exist after insertion")
    }

    /// Get the BV variable for a term
    #[must_use]
    pub fn get_bv(&self, term: TermId) -> Option<&BvVar> {
        self.term_to_bv.get(&term)
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &BvConfig {
        &self.config
    }

    /// Assert equality: a = b
    pub fn assert_eq(&mut self, a: TermId, b: TermId) {
        let bv_a = self.term_to_bv.get(&a).cloned();
        let bv_b = self.term_to_bv.get(&b).cloned();

        if let (Some(va), Some(vb)) = (bv_a, bv_b) {
            assert_eq!(va.width, vb.width);

            for i in 0..va.width as usize {
                // a[i] <=> b[i]
                // (a[i] => b[i]) and (b[i] => a[i])
                // (~a[i] or b[i]) and (~b[i] or a[i])
                self.sat
                    .add_clause([Lit::neg(va.bits[i]), Lit::pos(vb.bits[i])]);
                self.sat
                    .add_clause([Lit::neg(vb.bits[i]), Lit::pos(va.bits[i])]);
            }
        }
    }

    /// Assert disequality: a != b
    pub fn assert_neq(&mut self, a: TermId, b: TermId) {
        let bv_a = self.term_to_bv.get(&a).cloned();
        let bv_b = self.term_to_bv.get(&b).cloned();

        if let (Some(va), Some(vb)) = (bv_a, bv_b) {
            assert_eq!(va.width, vb.width);

            // At least one bit must differ
            // Introduce auxiliary variables for XOR of each bit pair
            let mut diff_lits: SmallVec<[Lit; 32]> = SmallVec::new();

            for i in 0..va.width as usize {
                // diff[i] = a[i] XOR b[i]
                let diff = self.sat.new_var();
                diff_lits.push(Lit::pos(diff));

                let ai = va.bits[i];
                let bi = vb.bits[i];

                // diff <=> (a XOR b)
                // diff => (a or b) and (~a or ~b)
                // ~diff => (~a or b) and (a or ~b)
                self.sat
                    .add_clause([Lit::neg(diff), Lit::pos(ai), Lit::pos(bi)]);
                self.sat
                    .add_clause([Lit::neg(diff), Lit::neg(ai), Lit::neg(bi)]);
                self.sat
                    .add_clause([Lit::pos(diff), Lit::neg(ai), Lit::pos(bi)]);
                self.sat
                    .add_clause([Lit::pos(diff), Lit::pos(ai), Lit::neg(bi)]);
            }

            // At least one diff bit must be true
            self.sat.add_clause(diff_lits);
        }
    }

    /// Assert unsigned less than: a < b
    pub fn assert_ult(&mut self, a: TermId, b: TermId) {
        let bv_a = self.term_to_bv.get(&a).cloned();
        let bv_b = self.term_to_bv.get(&b).cloned();

        if let (Some(va), Some(vb)) = (bv_a, bv_b) {
            assert_eq!(va.width, vb.width);

            // Get or create comparison result variable for a < b
            let key_ab = ComparisonKey { a, b };
            let ult_ab = if let Some(&var) = self.ult_cache.get(&key_ab) {
                var
            } else {
                let var = self.sat.new_var();
                self.encode_ult_result(&va.bits, &vb.bits, var);
                self.ult_cache.insert(key_ab.clone(), var);
                var
            };

            // Assert that a < b is true
            self.sat.add_clause([Lit::pos(ult_ab)]);

            // Check for conflict with b < a
            let key_ba = ComparisonKey { a: b, b: a };
            if let Some(&ult_ba) = self.ult_cache.get(&key_ba) {
                // If both a < b and b < a are asserted, we have a conflict
                // Add clause: NOT(a < b) OR NOT(b < a)
                // Since we already asserted a < b, this will make b < a false
                self.sat.add_clause([Lit::neg(ult_ab), Lit::neg(ult_ba)]);
            }

            // Also check for conflict with a <= b and b <= a
            // If a < b, then NOT(a = b), so we ensure anti-symmetry
        }
    }

    /// Assert a constant value for a bit vector
    pub fn assert_const(&mut self, term: TermId, value: u64, width: u32) {
        let bv = self.new_bv(term, width).clone();

        for i in 0..width as usize {
            let bit = (value >> i) & 1;
            if bit == 1 {
                self.sat.add_clause([Lit::pos(bv.bits[i])]);
            } else {
                self.sat.add_clause([Lit::neg(bv.bits[i])]);
            }
        }
    }

    /// Concatenate two bit vectors: result = high ++ low
    /// result[0..low.width-1] = low, result[low.width..low.width+high.width-1] = high
    pub fn concat(&mut self, result: TermId, high: TermId, low: TermId) {
        if let (Some(h), Some(l)) = (
            self.term_to_bv.get(&high).cloned(),
            self.term_to_bv.get(&low).cloned(),
        ) {
            let result_width = h.width + l.width;
            let r = self.new_bv(result, result_width).clone();

            // Copy low bits
            for i in 0..l.width as usize {
                self.encode_bit_eq(r.bits[i], l.bits[i]);
            }

            // Copy high bits
            for i in 0..h.width as usize {
                self.encode_bit_eq(r.bits[l.width as usize + i], h.bits[i]);
            }
        }
    }

    /// Extract a bit range from a bit vector: result = bv\[high:low\]
    /// Extract bits from position `low` to `high` (inclusive)
    pub fn extract(&mut self, result: TermId, bv: TermId, high: u32, low: u32) {
        if let Some(v) = self.term_to_bv.get(&bv).cloned() {
            assert!(high >= low);
            assert!(high < v.width);

            let result_width = high - low + 1;
            let r = self.new_bv(result, result_width).clone();

            for i in 0..result_width {
                let src_idx = (low + i) as usize;
                self.encode_bit_eq(r.bits[i as usize], v.bits[src_idx]);
            }
        }
    }

    /// Bitwise NOT: result = ~a
    pub fn bv_not(&mut self, result: TermId, a: TermId) {
        if let Some(va) = self.term_to_bv.get(&a).cloned() {
            let r = self.new_bv(result, va.width).clone();

            for i in 0..va.width as usize {
                // r[i] = ~a[i]
                self.encode_not(r.bits[i], va.bits[i]);
            }
        }
    }

    /// Bitwise AND: result = a & b
    pub fn bv_and(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let r = self.new_bv(result, va.width).clone();

            for i in 0..va.width as usize {
                self.encode_and(r.bits[i], va.bits[i], vb.bits[i]);
            }
        }
    }

    /// Bitwise OR: result = a | b
    pub fn bv_or(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let r = self.new_bv(result, va.width).clone();

            for i in 0..va.width as usize {
                self.encode_or(r.bits[i], va.bits[i], vb.bits[i]);
            }
        }
    }

    /// Bitwise XOR: result = a ^ b
    pub fn bv_xor(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let r = self.new_bv(result, va.width).clone();

            for i in 0..va.width as usize {
                self.encode_xor(r.bits[i], va.bits[i], vb.bits[i]);
            }
        }
    }

    /// Negation (two's complement): result = -a = ~a + 1
    pub fn bv_neg(&mut self, result: TermId, a: TermId) {
        if let Some(va) = self.term_to_bv.get(&a).cloned() {
            let r = self.new_bv(result, va.width).clone();

            // First compute ~a
            let mut not_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for &bit in &va.bits {
                let not_bit = self.sat.new_var();
                self.encode_not(not_bit, bit);
                not_bits.push(not_bit);
            }

            // Then add 1 using a ripple-carry adder
            self.encode_add_const(&r.bits, &not_bits, 1);
        }
    }

    /// Addition: result = a + b
    pub fn bv_add(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let r = self.new_bv(result, va.width).clone();

            self.encode_adder(&r.bits, &va.bits, &vb.bits);
        }
    }

    /// Subtraction: result = a - b = a + (-b)
    pub fn bv_sub(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let r = self.new_bv(result, va.width).clone();

            // Compute -b (two's complement)
            let mut neg_b: SmallVec<[Var; 32]> = SmallVec::new();
            for &bit in &vb.bits {
                let not_bit = self.sat.new_var();
                self.encode_not(not_bit, bit);
                neg_b.push(not_bit);
            }

            // Create temp variables for -b
            let mut neg_b_with_one: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..va.width {
                neg_b_with_one.push(self.sat.new_var());
            }
            self.encode_add_const(&neg_b_with_one, &neg_b, 1);

            // Add a + (-b)
            self.encode_adder(&r.bits, &va.bits, &neg_b_with_one);
        }
    }

    /// Multiplication: result = a * b (using shift-and-add)
    pub fn bv_mul(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let r = self.new_bv(result, va.width).clone();
            self.encode_mul(&r.bits, &va.bits, &vb.bits);
        }
    }

    /// Left shift: result = a << b
    pub fn bv_shl(&mut self, result: TermId, a: TermId, shift_amount: TermId) {
        if let (Some(va), Some(shift)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&shift_amount).cloned(),
        ) {
            assert_eq!(va.width, shift.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Build a barrel shifter
            let mut current = va.bits.clone();

            for s in 0..width.ilog2() + 1 {
                let shift_by = 1 << s;
                let mut next: SmallVec<[Var; 32]> = SmallVec::new();

                for i in 0..width {
                    let next_bit = self.sat.new_var();

                    if i >= shift_by {
                        // next_bit = shift[s] ? current[i - shift_by] : current[i]
                        self.encode_mux(
                            next_bit,
                            shift.bits[s as usize],
                            current[i - shift_by],
                            current[i],
                        );
                    } else {
                        // next_bit = shift[s] ? 0 : current[i]
                        let zero = self.sat.new_var();
                        self.sat.add_clause([Lit::neg(zero)]);
                        self.encode_mux(next_bit, shift.bits[s as usize], zero, current[i]);
                    }

                    next.push(next_bit);
                }

                current = next;
            }

            for i in 0..width {
                self.encode_bit_eq(r.bits[i], current[i]);
            }
        }
    }

    /// Logical right shift: result = a >> b (unsigned)
    pub fn bv_lshr(&mut self, result: TermId, a: TermId, shift_amount: TermId) {
        if let (Some(va), Some(shift)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&shift_amount).cloned(),
        ) {
            assert_eq!(va.width, shift.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Build a barrel shifter (right)
            let mut current = va.bits.clone();

            for s in 0..width.ilog2() + 1 {
                let shift_by = 1 << s;
                let mut next: SmallVec<[Var; 32]> = SmallVec::new();

                for i in 0..width {
                    let next_bit = self.sat.new_var();

                    if i + shift_by < width {
                        // next_bit = shift[s] ? current[i + shift_by] : current[i]
                        self.encode_mux(
                            next_bit,
                            shift.bits[s as usize],
                            current[i + shift_by],
                            current[i],
                        );
                    } else {
                        // next_bit = shift[s] ? 0 : current[i]
                        let zero = self.sat.new_var();
                        self.sat.add_clause([Lit::neg(zero)]);
                        self.encode_mux(next_bit, shift.bits[s as usize], zero, current[i]);
                    }

                    next.push(next_bit);
                }

                current = next;
            }

            for i in 0..width {
                self.encode_bit_eq(r.bits[i], current[i]);
            }
        }
    }

    /// Arithmetic right shift: result = a >> b (signed, sign-extends)
    pub fn bv_ashr(&mut self, result: TermId, a: TermId, shift_amount: TermId) {
        if let (Some(va), Some(shift)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&shift_amount).cloned(),
        ) {
            assert_eq!(va.width, shift.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Sign bit
            let sign = va.bits[width - 1];

            // Build a barrel shifter (right) with sign extension
            let mut current = va.bits.clone();

            for s in 0..width.ilog2() + 1 {
                let shift_by = 1 << s;
                let mut next: SmallVec<[Var; 32]> = SmallVec::new();

                for i in 0..width {
                    let next_bit = self.sat.new_var();

                    if i + shift_by < width {
                        // next_bit = shift[s] ? current[i + shift_by] : current[i]
                        self.encode_mux(
                            next_bit,
                            shift.bits[s as usize],
                            current[i + shift_by],
                            current[i],
                        );
                    } else {
                        // next_bit = shift[s] ? sign : current[i]
                        self.encode_mux(next_bit, shift.bits[s as usize], sign, current[i]);
                    }

                    next.push(next_bit);
                }

                current = next;
            }

            for i in 0..width {
                self.encode_bit_eq(r.bits[i], current[i]);
            }
        }
    }

    /// Signed less than: a < b (two's complement)
    pub fn assert_slt(&mut self, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let width = va.width as usize;

            // For signed comparison:
            // If sign bits differ: a < b iff a is negative (a[n-1] = 1)
            // If sign bits same: compare as unsigned

            let sign_a = va.bits[width - 1];
            let sign_b = vb.bits[width - 1];

            // diff_sign = sign_a XOR sign_b
            let diff_sign = self.sat.new_var();
            self.encode_xor(diff_sign, sign_a, sign_b);

            // If signs differ, result = sign_a
            // If signs same, result = unsigned comparison of remaining bits

            // Create result variable
            let result = self.sat.new_var();

            // Case 1: diff_sign => result = sign_a
            // diff_sign => (sign_a <=> result)
            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::neg(sign_a), Lit::pos(result)]);
            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::pos(sign_a), Lit::neg(result)]);

            // Case 2: ~diff_sign => result = ult(a, b)
            // We need to compute unsigned less than and assert it when signs are equal
            let ult_result = self.sat.new_var();
            self.encode_ult_result(&va.bits, &vb.bits, ult_result);

            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::neg(ult_result), Lit::pos(result)]);
            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::pos(ult_result), Lit::neg(result)]);

            // Assert that result is true
            self.sat.add_clause([Lit::pos(result)]);
        }
    }

    /// Signed less than or equal: a <= b
    pub fn assert_sle(&mut self, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);

            // a <= b is equivalent to NOT(b < a)
            // Create temporary variables for checking b < a
            let slt_ba = self.sat.new_var();

            // Encode b < a into slt_ba
            let width = va.width as usize;
            let sign_a = va.bits[width - 1];
            let sign_b = vb.bits[width - 1];

            let diff_sign = self.sat.new_var();
            self.encode_xor(diff_sign, sign_b, sign_a);

            // If signs differ, b < a iff sign_b = 1
            // If signs same, b < a iff ult(b, a)
            let ult_result = self.sat.new_var();
            self.encode_ult_result(&vb.bits, &va.bits, ult_result);

            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::neg(sign_b), Lit::pos(slt_ba)]);
            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::pos(sign_b), Lit::neg(slt_ba)]);
            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::neg(ult_result), Lit::pos(slt_ba)]);
            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::pos(ult_result), Lit::neg(slt_ba)]);

            // Assert NOT(slt_ba) which means a <= b
            self.sat.add_clause([Lit::neg(slt_ba)]);
        }
    }

    // ===== Helper encoding functions =====

    /// Encode bit equality: a <=> b
    fn encode_bit_eq(&mut self, a: Var, b: Var) {
        self.sat.add_clause([Lit::neg(a), Lit::pos(b)]);
        self.sat.add_clause([Lit::pos(a), Lit::neg(b)]);
    }

    /// Encode NOT gate: out = ~in
    fn encode_not(&mut self, out: Var, input: Var) {
        self.sat.add_clause([Lit::pos(out), Lit::pos(input)]);
        self.sat.add_clause([Lit::neg(out), Lit::neg(input)]);
    }

    /// Encode AND gate: out = a & b
    fn encode_and(&mut self, out: Var, a: Var, b: Var) {
        // out <=> (a AND b)
        // out => a, out => b, (a AND b) => out
        self.sat.add_clause([Lit::neg(out), Lit::pos(a)]);
        self.sat.add_clause([Lit::neg(out), Lit::pos(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::neg(a), Lit::neg(b)]);
    }

    /// Encode OR gate: out = a | b
    fn encode_or(&mut self, out: Var, a: Var, b: Var) {
        // out <=> (a OR b)
        self.sat
            .add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
        self.sat.add_clause([Lit::pos(out), Lit::neg(a)]);
        self.sat.add_clause([Lit::pos(out), Lit::neg(b)]);
    }

    /// Encode XOR gate: out = a ^ b
    fn encode_xor(&mut self, out: Var, a: Var, b: Var) {
        // out <=> (a XOR b)
        self.sat
            .add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
        self.sat
            .add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
    }

    /// Encode multiplexer: out = sel ? if_true : if_false
    fn encode_mux(&mut self, out: Var, sel: Var, if_true: Var, if_false: Var) {
        // out = (sel AND if_true) OR (~sel AND if_false)
        self.sat
            .add_clause([Lit::neg(sel), Lit::neg(if_true), Lit::pos(out)]);
        self.sat
            .add_clause([Lit::neg(sel), Lit::pos(if_true), Lit::neg(out)]);
        self.sat
            .add_clause([Lit::pos(sel), Lit::neg(if_false), Lit::pos(out)]);
        self.sat
            .add_clause([Lit::pos(sel), Lit::pos(if_false), Lit::neg(out)]);
    }

    /// Encode full adder: (sum, carry_out) = a + b + carry_in
    fn encode_full_adder(&mut self, sum: Var, carry_out: Var, a: Var, b: Var, carry_in: Var) {
        // sum = a XOR b XOR carry_in
        let xor_ab = self.sat.new_var();
        self.encode_xor(xor_ab, a, b);
        self.encode_xor(sum, xor_ab, carry_in);

        // carry_out = (a AND b) OR (carry_in AND (a XOR b))
        let and_ab = self.sat.new_var();
        self.encode_and(and_ab, a, b);

        let and_cin_xor = self.sat.new_var();
        self.encode_and(and_cin_xor, carry_in, xor_ab);

        self.encode_or(carry_out, and_ab, and_cin_xor);
    }

    /// Encode ripple-carry adder: result = a + b
    fn encode_adder(&mut self, result: &[Var], a: &[Var], b: &[Var]) {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());

        let width = result.len();
        let mut carry = self.sat.new_var();
        self.sat.add_clause([Lit::neg(carry)]); // Initial carry = 0

        for i in 0..width {
            let next_carry = self.sat.new_var(); // Overflow carry ignored for last iteration

            self.encode_full_adder(result[i], next_carry, a[i], b[i], carry);
            carry = next_carry;
        }
    }

    /// Encode addition with constant: result = a + const
    fn encode_add_const(&mut self, result: &[Var], a: &[Var], constant: u64) {
        assert_eq!(result.len(), a.len());

        let width = result.len();
        let mut carry = self.sat.new_var();
        self.sat.add_clause([Lit::neg(carry)]); // Initial carry = 0

        for i in 0..width {
            let const_bit = ((constant >> i) & 1) == 1;
            let next_carry = self.sat.new_var(); // Overflow carry ignored for last iteration

            if const_bit {
                // Half adder with constant 1
                let one = self.sat.new_var();
                self.sat.add_clause([Lit::pos(one)]);
                self.encode_full_adder(result[i], next_carry, a[i], one, carry);
            } else {
                // Half adder with constant 0
                let zero = self.sat.new_var();
                self.sat.add_clause([Lit::neg(zero)]);
                self.encode_full_adder(result[i], next_carry, a[i], zero, carry);
            }

            carry = next_carry;
        }
    }

    /// Encode unsigned less than and store result in a variable
    /// Encode unsigned less-than: result ⇔ (a < b)
    /// Uses LSB-to-MSB comparison: higher bits override lower bits.
    fn encode_ult_result(&mut self, a_bits: &[Var], b_bits: &[Var], result: Var) {
        let width = a_bits.len();
        if width == 0 {
            // Empty bitvectors: 0 < 0 is false
            self.sat.add_clause([Lit::neg(result)]);
            return;
        }

        // Compare from LSB to MSB
        // lt_i represents "a < b considering only bits 0..i"
        // Higher indexed bits (more significant) override lower bits
        // Recurrence: lt_next = (~a[i] & b[i]) | ((a[i] = b[i]) & lt_prev)
        //
        // Meaning:
        // - If a[i] < b[i], then a < b (current bit overrides lower bits)
        // - If a[i] > b[i], then a > b (current bit overrides lower bits)
        // - If a[i] = b[i], result depends on lower bits (lt_prev)

        // Start with LSB (bit 0)
        // lt_0 = ~a[0] & b[0]
        let mut lt_prev = self.sat.new_var();
        self.encode_and_not_a(lt_prev, a_bits[0], b_bits[0]);

        // Process bits from 1 to MSB
        for i in 1..width {
            let ai = a_bits[i];
            let bi = b_bits[i];

            // lt_at_i = ~ai & bi (a < b at this specific bit)
            let lt_at_i = self.sat.new_var();
            self.encode_and_not_a(lt_at_i, ai, bi);

            // eq_i = (ai ⇔ bi) (bits are equal)
            let eq_i = self.sat.new_var();
            self.encode_xnor(eq_i, ai, bi);

            // carry_prev = eq_i & lt_prev (propagate from lower bits)
            let carry_prev = self.sat.new_var();
            self.encode_and(carry_prev, eq_i, lt_prev);

            // lt_next = lt_at_i | carry_prev
            let lt_next = self.sat.new_var();
            self.encode_or(lt_next, lt_at_i, carry_prev);

            lt_prev = lt_next;
        }

        self.encode_bit_eq(result, lt_prev);
    }

    /// Encode out = ~a & b (AND with first input negated)
    fn encode_and_not_a(&mut self, out: Var, a: Var, b: Var) {
        // out ⇔ (~a & b)
        // out → ~a: ~out | ~a
        self.sat.add_clause([Lit::neg(out), Lit::neg(a)]);
        // out → b: ~out | b
        self.sat.add_clause([Lit::neg(out), Lit::pos(b)]);
        // (~a & b) → out: a | ~b | out
        self.sat
            .add_clause([Lit::pos(a), Lit::neg(b), Lit::pos(out)]);
    }

    /// Encode out = (a ⇔ b) (XNOR gate)
    fn encode_xnor(&mut self, out: Var, a: Var, b: Var) {
        // out ⇔ (a ⇔ b)
        // out is true when a = b
        // Clauses:
        // ~out | ~a | b    (out & a → b)
        // ~out | a | ~b    (out & ~a → ~b)
        // out | ~a | ~b    (~out → a ≠ b, i.e., ~a & ~b → out, or a | b → ~out)
        // out | a | b      (~out → a ≠ b, i.e., a & b → out, or ~a | ~b → ~out)
        self.sat
            .add_clause([Lit::neg(out), Lit::neg(a), Lit::pos(b)]);
        self.sat
            .add_clause([Lit::neg(out), Lit::pos(a), Lit::neg(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::neg(a), Lit::neg(b)]);
        self.sat
            .add_clause([Lit::pos(out), Lit::pos(a), Lit::pos(b)]);
    }

    /// Unsigned division: result = a / b (unsigned)
    /// If b = 0, result is all 1s (SMT-LIB semantics)
    pub fn bv_udiv(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Check if divisor is zero
            let b_is_zero = self.sat.new_var();
            let mut all_zero_lits: SmallVec<[Var; 32]> = SmallVec::new();
            for &bit in &vb.bits {
                all_zero_lits.push(bit);
            }
            self.encode_all_zero(b_is_zero, &all_zero_lits);

            // Create quotient and remainder variables
            let mut quot_bits: SmallVec<[Var; 32]> = SmallVec::new();
            let mut rem_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                quot_bits.push(self.sat.new_var());
                rem_bits.push(self.sat.new_var());
            }

            // Encode: full_prod = quot * b using double-width to detect overflow
            // full_prod[0..width-1] = low bits, full_prod[width..2*width-1] = high bits
            let mut full_prod_bits: SmallVec<[Var; 64]> = SmallVec::new();
            for _ in 0..(2 * width) {
                full_prod_bits.push(self.sat.new_var());
            }
            self.encode_mul_full(&full_prod_bits, &quot_bits, &vb.bits);

            // The low bits are our prod
            let prod_bits: SmallVec<[Var; 32]> = full_prod_bits[0..width].iter().copied().collect();

            // Enforce: high bits of product are zero (no overflow) when b != 0
            for i in width..(2 * width) {
                // ~b_is_zero => ~full_prod_bits[i]
                // b_is_zero | ~full_prod_bits[i]
                self.sat
                    .add_clause([Lit::pos(b_is_zero), Lit::neg(full_prod_bits[i])]);
            }

            // Encode: sum = prod + rem
            let mut sum_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                sum_bits.push(self.sat.new_var());
            }
            self.encode_adder(&sum_bits, &prod_bits, &rem_bits);

            // Enforce: a = sum (the division equation)
            for i in 0..width {
                self.encode_bit_eq(va.bits[i], sum_bits[i]);
            }

            // Enforce: rem < b (when b != 0)
            let rem_lt_b = self.sat.new_var();
            self.encode_ult_result(&rem_bits, &vb.bits, rem_lt_b);
            self.sat
                .add_clause([Lit::pos(b_is_zero), Lit::pos(rem_lt_b)]);

            // All 1s for division by zero result
            let mut all_ones: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                let one = self.sat.new_var();
                self.sat.add_clause([Lit::pos(one)]);
                all_ones.push(one);
            }

            // result = b_is_zero ? all_ones : quot_bits
            for i in 0..width {
                self.encode_mux(r.bits[i], b_is_zero, all_ones[i], quot_bits[i]);
            }
        }
    }

    /// Unsigned remainder: result = a % b (unsigned)
    /// If b = 0, result = a (SMT-LIB semantics)
    pub fn bv_urem(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Check if divisor is zero
            let b_is_zero = self.sat.new_var();
            let mut all_zero_lits: SmallVec<[Var; 32]> = SmallVec::new();
            for &bit in &vb.bits {
                all_zero_lits.push(bit);
            }
            self.encode_all_zero(b_is_zero, &all_zero_lits);

            // Create quotient bits (unconstrained - solver will find values)
            let mut quot_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                quot_bits.push(self.sat.new_var());
            }

            // Create remainder bits (unconstrained - solver will find values)
            let mut rem_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                rem_bits.push(self.sat.new_var());
            }

            // Encode: full_prod = quot * b using double-width to detect overflow
            let mut full_prod_bits: SmallVec<[Var; 64]> = SmallVec::new();
            for _ in 0..(2 * width) {
                full_prod_bits.push(self.sat.new_var());
            }
            self.encode_mul_full(&full_prod_bits, &quot_bits, &vb.bits);

            // The low bits are our prod
            let prod_bits: SmallVec<[Var; 32]> = full_prod_bits[0..width].iter().copied().collect();

            // Enforce: high bits of product are zero (no overflow) when b != 0
            for i in width..(2 * width) {
                self.sat
                    .add_clause([Lit::pos(b_is_zero), Lit::neg(full_prod_bits[i])]);
            }

            // Encode: sum = prod + rem
            let mut sum_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                sum_bits.push(self.sat.new_var());
            }
            self.encode_adder(&sum_bits, &prod_bits, &rem_bits);

            // Enforce: a = sum (the division equation a = q*b + r)
            for i in 0..width {
                self.encode_bit_eq(va.bits[i], sum_bits[i]);
            }

            // Encode: rem < b (remainder must be less than divisor)
            let rem_lt_b = self.sat.new_var();
            self.encode_ult_result(&rem_bits, &vb.bits, rem_lt_b);
            // This constraint only applies when b != 0
            self.sat
                .add_clause([Lit::pos(b_is_zero), Lit::pos(rem_lt_b)]);

            // result = b_is_zero ? a : rem_bits
            for i in 0..width {
                self.encode_mux(r.bits[i], b_is_zero, va.bits[i], rem_bits[i]);
            }
        }
    }

    /// Signed division: result = a / b (signed, two's complement)
    /// If b = 0, result = all 1s (SMT-LIB semantics)
    pub fn bv_sdiv(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Check if divisor is zero
            let b_is_zero = self.sat.new_var();
            let mut all_zero_lits: SmallVec<[Var; 32]> = SmallVec::new();
            for &bit in &vb.bits {
                all_zero_lits.push(bit);
            }
            self.encode_all_zero(b_is_zero, &all_zero_lits);

            // Get sign bits
            let sign_a = va.bits[width - 1];
            let sign_b = vb.bits[width - 1];

            // Compute absolute values using MUX
            let mut abs_a: SmallVec<[Var; 32]> = SmallVec::new();
            let mut abs_b: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                abs_a.push(self.sat.new_var());
                abs_b.push(self.sat.new_var());
            }

            // neg_a = -a (two's complement)
            let mut neg_a: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_a.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_a, &va.bits);

            // abs_a = sign_a ? neg_a : a
            for i in 0..width {
                self.encode_mux(abs_a[i], sign_a, neg_a[i], va.bits[i]);
            }

            // neg_b = -b (two's complement)
            let mut neg_b: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_b.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_b, &vb.bits);

            // abs_b = sign_b ? neg_b : b
            for i in 0..width {
                self.encode_mux(abs_b[i], sign_b, neg_b[i], vb.bits[i]);
            }

            // Create quot_abs and rem_abs for unsigned division
            let mut quot_abs: SmallVec<[Var; 32]> = SmallVec::new();
            let mut rem_abs: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                quot_abs.push(self.sat.new_var());
                rem_abs.push(self.sat.new_var());
            }

            // Encode division constraint: abs_a = quot_abs * abs_b + rem_abs
            // Use double-width multiplication to detect overflow
            let mut full_prod: SmallVec<[Var; 64]> = SmallVec::new();
            for _ in 0..(2 * width) {
                full_prod.push(self.sat.new_var());
            }
            self.encode_mul_full(&full_prod, &quot_abs, &abs_b);

            // The low bits are our prod
            let prod: SmallVec<[Var; 32]> = full_prod[0..width].iter().copied().collect();

            // Enforce: high bits of product are zero (no overflow) when b != 0
            for i in width..(2 * width) {
                self.sat
                    .add_clause([Lit::pos(b_is_zero), Lit::neg(full_prod[i])]);
            }

            let mut sum: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                sum.push(self.sat.new_var());
            }
            self.encode_adder(&sum, &prod, &rem_abs);

            // Enforce abs_a = sum (unconditionally - division equation always holds)
            for i in 0..width {
                self.encode_bit_eq(abs_a[i], sum[i]);
            }

            // Enforce rem_abs < abs_b (unconditionally for well-formed division)
            let rem_lt_b = self.sat.new_var();
            self.encode_ult_result(&rem_abs, &abs_b, rem_lt_b);
            // Only enforce when b != 0
            self.sat
                .add_clause([Lit::pos(b_is_zero), Lit::pos(rem_lt_b)]);

            // Result sign: sign_a XOR sign_b
            let result_sign = self.sat.new_var();
            self.encode_xor(result_sign, sign_a, sign_b);

            // neg_quot = -quot_abs
            let mut neg_quot: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_quot.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_quot, &quot_abs);

            // signed_quot = result_sign ? neg_quot : quot_abs
            let mut signed_quot: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                signed_quot.push(self.sat.new_var());
            }
            for i in 0..width {
                self.encode_mux(signed_quot[i], result_sign, neg_quot[i], quot_abs[i]);
            }

            // All 1s for division by zero result
            let mut all_ones: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                let one = self.sat.new_var();
                self.sat.add_clause([Lit::pos(one)]); // Force to 1
                all_ones.push(one);
            }

            // result = b_is_zero ? all_ones : signed_quot
            for i in 0..width {
                self.encode_mux(r.bits[i], b_is_zero, all_ones[i], signed_quot[i]);
            }
        }
    }

    /// Signed remainder: result = a % b (signed)
    /// Sign of result matches sign of dividend a
    /// If b = 0, result = a (SMT-LIB semantics)
    pub fn bv_srem(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Check if divisor is zero
            let b_is_zero = self.sat.new_var();
            let mut all_zero_lits: SmallVec<[Var; 32]> = SmallVec::new();
            for &bit in &vb.bits {
                all_zero_lits.push(bit);
            }
            self.encode_all_zero(b_is_zero, &all_zero_lits);

            // Get sign bits
            let sign_a = va.bits[width - 1];
            let sign_b = vb.bits[width - 1];

            // Compute absolute values using MUX
            let mut abs_a: SmallVec<[Var; 32]> = SmallVec::new();
            let mut abs_b: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                abs_a.push(self.sat.new_var());
                abs_b.push(self.sat.new_var());
            }

            // neg_a = -a (two's complement)
            let mut neg_a: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_a.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_a, &va.bits);

            // abs_a = sign_a ? neg_a : a
            for i in 0..width {
                self.encode_mux(abs_a[i], sign_a, neg_a[i], va.bits[i]);
            }

            // neg_b = -b (two's complement)
            let mut neg_b: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_b.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_b, &vb.bits);

            // abs_b = sign_b ? neg_b : b
            for i in 0..width {
                self.encode_mux(abs_b[i], sign_b, neg_b[i], vb.bits[i]);
            }

            // Create quot_abs and rem_abs for unsigned division
            let mut quot_abs: SmallVec<[Var; 32]> = SmallVec::new();
            let mut rem_abs: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                quot_abs.push(self.sat.new_var());
                rem_abs.push(self.sat.new_var());
            }

            // Encode division constraint: abs_a = quot_abs * abs_b + rem_abs
            // Use double-width multiplication to detect overflow
            let mut full_prod: SmallVec<[Var; 64]> = SmallVec::new();
            for _ in 0..(2 * width) {
                full_prod.push(self.sat.new_var());
            }
            self.encode_mul_full(&full_prod, &quot_abs, &abs_b);

            // The low bits are our prod
            let prod: SmallVec<[Var; 32]> = full_prod[0..width].iter().copied().collect();

            // Enforce: high bits of product are zero (no overflow) when b != 0
            for i in width..(2 * width) {
                self.sat
                    .add_clause([Lit::pos(b_is_zero), Lit::neg(full_prod[i])]);
            }

            let mut sum: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                sum.push(self.sat.new_var());
            }
            self.encode_adder(&sum, &prod, &rem_abs);

            // Enforce abs_a = sum (unconditionally - division equation always holds)
            for i in 0..width {
                self.encode_bit_eq(abs_a[i], sum[i]);
            }

            // Enforce rem_abs < abs_b (only when b != 0)
            let rem_lt_b = self.sat.new_var();
            self.encode_ult_result(&rem_abs, &abs_b, rem_lt_b);
            self.sat
                .add_clause([Lit::pos(b_is_zero), Lit::pos(rem_lt_b)]);

            // neg_rem = -rem_abs (for negative dividend case)
            let mut neg_rem: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_rem.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_rem, &rem_abs);

            // signed_rem = sign_a ? neg_rem : rem_abs
            // (sign of result matches sign of dividend)
            let mut signed_rem: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                signed_rem.push(self.sat.new_var());
            }
            for i in 0..width {
                self.encode_mux(signed_rem[i], sign_a, neg_rem[i], rem_abs[i]);
            }

            // result = b_is_zero ? a : signed_rem
            for i in 0..width {
                self.encode_mux(r.bits[i], b_is_zero, va.bits[i], signed_rem[i]);
            }
        }
    }

    // ===== Additional helper encoding functions =====

    /// Encode: out = 1 iff all bits in the list are 0
    fn encode_all_zero(&mut self, out: Var, bits: &[Var]) {
        if bits.is_empty() {
            self.sat.add_clause([Lit::pos(out)]);
            return;
        }

        // out = AND(~bits[i] for all i)
        // out => ~bits[i] for all i
        for &bit in bits {
            self.sat.add_clause([Lit::neg(out), Lit::neg(bit)]);
        }

        // (~bits[0] AND ... AND ~bits[n-1]) => out
        let mut clause: SmallVec<[Lit; 32]> = SmallVec::new();
        clause.push(Lit::pos(out));
        for &bit in bits {
            clause.push(Lit::pos(bit));
        }
        self.sat.add_clause(clause);
    }

    /// Encode two's complement negation: result = -a
    fn encode_two_complement(&mut self, result: &[Var], a: &[Var]) {
        assert_eq!(result.len(), a.len());

        // ~a
        let mut not_a: SmallVec<[Var; 32]> = SmallVec::new();
        for &bit in a {
            let not_bit = self.sat.new_var();
            self.encode_not(not_bit, bit);
            not_a.push(not_bit);
        }

        // ~a + 1
        self.encode_add_const(result, &not_a, 1);
    }

    /// Encode multiplication using symmetric schoolbook method: result = a * b
    /// This encoding is symmetric with respect to a and b, allowing solving for either operand.
    /// Uses Wallace tree-style carry propagation with proper column tracking.
    fn encode_mul(&mut self, result: &[Var], a: &[Var], b: &[Var]) {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());

        let width = result.len();

        // Create partial products: columns[k] contains all bits that contribute to result[k]
        // Initially, columns[k] = { a[i] AND b[j] | i + j = k }
        let mut columns: Vec<Vec<Var>> = vec![Vec::new(); width];

        for (i, &a_bit) in a.iter().enumerate().take(width) {
            for (j, &b_bit) in b.iter().enumerate().take(width) {
                let sum_pos = i + j;
                if sum_pos < width {
                    let pp = self.sat.new_var();
                    self.encode_and(pp, a_bit, b_bit);
                    columns[sum_pos].push(pp);
                }
            }
        }

        // Use carry-save reduction to reduce each column to at most 2 bits
        // Then do a final ripple-carry addition
        self.reduce_columns_and_add(result, &mut columns);
    }

    /// Reduce columns using 3:2 compressors until each column has at most 2 bits,
    /// then use a final ripple-carry adder to produce the result.
    fn reduce_columns_and_add(&mut self, result: &[Var], columns: &mut Vec<Vec<Var>>) {
        let width = columns.len();

        // Repeatedly reduce columns using 3:2 compressors
        // Each full adder takes 3 bits from column k and produces:
        //   - 1 sum bit in column k
        //   - 1 carry bit in column k+1
        loop {
            let max_height = columns.iter().map(|c| c.len()).max().unwrap_or(0);
            if max_height <= 2 {
                break;
            }

            let mut new_columns: Vec<Vec<Var>> = vec![Vec::new(); width];

            for k in 0..width {
                let bits = &columns[k];
                let mut i = 0;

                while i + 2 < bits.len() {
                    // Full adder: sum stays in column k, carry goes to column k+1
                    let sum = self.sat.new_var();
                    let carry = self.sat.new_var();
                    self.encode_full_adder_bit(sum, carry, bits[i], bits[i + 1], bits[i + 2]);
                    new_columns[k].push(sum);
                    if k + 1 < width {
                        new_columns[k + 1].push(carry);
                    }
                    i += 3;
                }

                // Pass through remaining bits (0, 1, or 2)
                for &bit in &bits[i..] {
                    new_columns[k].push(bit);
                }
            }

            *columns = new_columns;
        }

        // Now each column has at most 2 bits
        // Create two operands for final addition
        let mut operand_a: SmallVec<[Var; 32]> = SmallVec::new();
        let mut operand_b: SmallVec<[Var; 32]> = SmallVec::new();

        for column in columns.iter().take(width) {
            match column.len() {
                0 => {
                    let zero = self.sat.new_var();
                    self.sat.add_clause([Lit::neg(zero)]);
                    operand_a.push(zero);
                    let zero2 = self.sat.new_var();
                    self.sat.add_clause([Lit::neg(zero2)]);
                    operand_b.push(zero2);
                }
                1 => {
                    operand_a.push(column[0]);
                    let zero = self.sat.new_var();
                    self.sat.add_clause([Lit::neg(zero)]);
                    operand_b.push(zero);
                }
                2 => {
                    operand_a.push(column[0]);
                    operand_b.push(column[1]);
                }
                _ => unreachable!("Column should have at most 2 bits after reduction"),
            }
        }

        // Final ripple-carry addition
        self.encode_adder(result, &operand_a, &operand_b);
    }

    /// Full adder for single bits: sum = a XOR b XOR cin, cout = (a AND b) OR (cin AND (a XOR b))
    fn encode_full_adder_bit(&mut self, sum: Var, cout: Var, a: Var, b: Var, cin: Var) {
        // a XOR b
        let a_xor_b = self.sat.new_var();
        self.encode_xor(a_xor_b, a, b);

        // sum = a_xor_b XOR cin
        self.encode_xor(sum, a_xor_b, cin);

        // a AND b
        let a_and_b = self.sat.new_var();
        self.encode_and(a_and_b, a, b);

        // cin AND (a XOR b)
        let cin_and_axorb = self.sat.new_var();
        self.encode_and(cin_and_axorb, cin, a_xor_b);

        // cout = (a AND b) OR (cin AND (a XOR b))
        self.encode_or(cout, a_and_b, cin_and_axorb);
    }

    /// Encode full multiplication: result = a * b with double-width result
    /// result has length 2*width, a and b have length width
    /// result[0..width-1] = low bits, result[width..2*width-1] = high bits
    /// Uses Wallace tree-style carry propagation with proper column tracking.
    fn encode_mul_full(&mut self, result: &[Var], a: &[Var], b: &[Var]) {
        let width = a.len();
        assert_eq!(b.len(), width);
        assert_eq!(result.len(), 2 * width);

        let double_width = 2 * width;

        // Create partial products: columns[k] contains all bits that contribute to result[k]
        let mut columns: Vec<Vec<Var>> = vec![Vec::new(); double_width];

        for (i, &a_bit) in a.iter().enumerate().take(width) {
            for (j, &b_bit) in b.iter().enumerate().take(width) {
                let sum_pos = i + j;
                let pp = self.sat.new_var();
                self.encode_and(pp, a_bit, b_bit);
                columns[sum_pos].push(pp);
            }
        }

        // Use carry-save reduction and final addition
        self.reduce_columns_and_add(result, &mut columns);
    }

    /// Get the value of a bit vector from the model
    #[must_use]
    pub fn get_value(&self, term: TermId) -> Option<u64> {
        let bv = self.term_to_bv.get(&term)?;
        let model = self.sat.model();

        let mut value = 0u64;
        for (i, &var) in bv.bits.iter().enumerate() {
            if model[var.index()].is_true() {
                value |= 1 << i;
            }
        }
        Some(value)
    }
}

impl Theory for BvSolver {
    fn id(&self) -> TheoryId {
        TheoryId::BV
    }

    fn name(&self) -> &str {
        "BV"
    }

    fn can_handle(&self, _term: TermId) -> bool {
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        self.assertions.push((term, true));
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        self.assertions.push((term, false));
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        match self.sat.solve() {
            SolverResult::Sat => {
                // Backtrack to root level so that new clauses can be added
                // without conflicting with the current satisfying assignment
                self.sat.backtrack_to_root();
                Ok(TheoryResult::Sat)
            }
            SolverResult::Unsat => Ok(TheoryResult::Unsat(Vec::new())),
            SolverResult::Unknown => Ok(TheoryResult::Unknown),
        }
    }

    fn push(&mut self) {
        self.context_stack.push(self.assertions.len());
        self.sat.push();
    }

    fn pop(&mut self) {
        if let Some(len) = self.context_stack.pop() {
            self.assertions.truncate(len);
            self.sat.pop();
        }
    }

    fn reset(&mut self) {
        self.sat.reset();
        self.term_to_bv.clear();
        self.assertions.clear();
        self.context_stack.clear();
        self.ult_cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bv_eq() {
        let mut solver = BvSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_bv(a, 8);
        solver.new_bv(b, 8);

        // a = 42
        solver.assert_const(a, 42, 8);

        // a = b
        solver.assert_eq(a, b);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        // b should be 42
        assert_eq!(solver.get_value(b), Some(42));
    }

    #[test]
    fn test_bv_neq() {
        let mut solver = BvSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_bv(a, 4);
        solver.new_bv(b, 4);

        // a = 5
        solver.assert_const(a, 5, 4);
        // b = 5
        solver.assert_const(b, 5, 4);
        // a != b (contradiction)
        solver.assert_neq(a, b);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }
}
