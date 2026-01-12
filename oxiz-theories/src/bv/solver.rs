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

            // Ripple comparison: a < b means ~a[n-1] & b[n-1] or (a[n-1] = b[n-1] & a[n-2..0] < b[n-2..0])
            // We use auxiliary variables for the comparison result
            self.encode_ult(&va.bits, &vb.bits);
        }
    }

    /// Encode unsigned less than comparison
    fn encode_ult(&mut self, a_bits: &[Var], b_bits: &[Var]) {
        let width = a_bits.len();
        if width == 0 {
            // Empty comparison is always false
            self.sat.add_clause(std::iter::empty());
            return;
        }

        // Create carry variables for ripple comparison
        // lt[i] = a[i-1..0] < b[i-1..0]
        // lt[i] = (a[i-1] < b[i-1]) or (a[i-1] = b[i-1] and lt[i-1])
        //       = (~a[i-1] and b[i-1]) or ((a[i-1] <=> b[i-1]) and lt[i-1])

        let mut lt_vars: Vec<Var> = Vec::with_capacity(width + 1);

        // lt[0] = false (no bits compared yet)
        let lt0 = self.sat.new_var();
        self.sat.add_clause([Lit::neg(lt0)]); // lt0 = false
        lt_vars.push(lt0);

        for i in 0..width {
            let lt_next = self.sat.new_var();
            let ai = a_bits[i];
            let bi = b_bits[i];
            let lt_prev = lt_vars[i];

            // lt_next = (~ai and bi) or (ai = bi and lt_prev)
            // Simplified encoding:
            // lt_next => (~ai and bi) or (ai = bi and lt_prev)
            // ~lt_next or (~ai and bi) or (ai = bi and lt_prev)

            // Break into CNF:
            // lt_next => bi or (ai and lt_prev)
            self.sat
                .add_clause([Lit::neg(lt_next), Lit::pos(bi), Lit::pos(ai)]);
            self.sat
                .add_clause([Lit::neg(lt_next), Lit::pos(bi), Lit::pos(lt_prev)]);

            // lt_next => ~ai or lt_prev
            self.sat
                .add_clause([Lit::neg(lt_next), Lit::neg(ai), Lit::pos(lt_prev)]);

            // (~ai and bi) => lt_next
            self.sat
                .add_clause([Lit::pos(ai), Lit::neg(bi), Lit::pos(lt_next)]);

            // (ai = bi and lt_prev) => lt_next
            // Split by case: ai and bi and lt_prev => lt_next
            self.sat.add_clause([
                Lit::neg(ai),
                Lit::neg(bi),
                Lit::neg(lt_prev),
                Lit::pos(lt_next),
            ]);
            // ~ai and ~bi and lt_prev => lt_next
            self.sat.add_clause([
                Lit::pos(ai),
                Lit::pos(bi),
                Lit::neg(lt_prev),
                Lit::pos(lt_next),
            ]);

            lt_vars.push(lt_next);
        }

        // Assert that the final comparison is true
        let final_lt = *lt_vars
            .last()
            .expect("lt_vars should not be empty after loop");
        self.sat.add_clause([Lit::pos(final_lt)]);
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
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Initialize accumulator to 0
            let mut acc: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                let zero = self.sat.new_var();
                self.sat.add_clause([Lit::neg(zero)]);
                acc.push(zero);
            }

            // Shift-and-add multiplication
            for i in 0..width {
                // If b[i] is 1, add (a << i) to accumulator
                let mut shifted_a: SmallVec<[Var; 32]> = SmallVec::new();

                // Shift a left by i positions
                for j in 0..width {
                    if j < i {
                        let zero = self.sat.new_var();
                        self.sat.add_clause([Lit::neg(zero)]);
                        shifted_a.push(zero);
                    } else if j - i < width {
                        shifted_a.push(va.bits[j - i]);
                    } else {
                        let zero = self.sat.new_var();
                        self.sat.add_clause([Lit::neg(zero)]);
                        shifted_a.push(zero);
                    }
                }

                // Conditional add: if b[i] then acc += shifted_a
                let mut new_acc: SmallVec<[Var; 32]> = SmallVec::new();
                for _ in 0..width {
                    new_acc.push(self.sat.new_var());
                }

                let mut sum: SmallVec<[Var; 32]> = SmallVec::new();
                for _ in 0..width {
                    sum.push(self.sat.new_var());
                }
                self.encode_adder(&sum, &acc, &shifted_a);

                // new_acc = b[i] ? sum : acc
                for j in 0..width {
                    self.encode_mux(new_acc[j], vb.bits[i], sum[j], acc[j]);
                }

                acc = new_acc;
            }

            // Copy accumulator to result
            for i in 0..width {
                self.encode_bit_eq(r.bits[i], acc[i]);
            }
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
    fn encode_ult_result(&mut self, a_bits: &[Var], b_bits: &[Var], result: Var) {
        let width = a_bits.len();
        if width == 0 {
            self.sat.add_clause([Lit::neg(result)]);
            return;
        }

        let mut lt_vars: Vec<Var> = Vec::with_capacity(width + 1);

        let lt0 = self.sat.new_var();
        self.sat.add_clause([Lit::neg(lt0)]);
        lt_vars.push(lt0);

        for i in 0..width {
            let lt_next = self.sat.new_var();
            let ai = a_bits[i];
            let bi = b_bits[i];
            let lt_prev = lt_vars[i];

            self.sat
                .add_clause([Lit::neg(lt_next), Lit::pos(bi), Lit::pos(ai)]);
            self.sat
                .add_clause([Lit::neg(lt_next), Lit::pos(bi), Lit::pos(lt_prev)]);
            self.sat
                .add_clause([Lit::neg(lt_next), Lit::neg(ai), Lit::pos(lt_prev)]);
            self.sat
                .add_clause([Lit::pos(ai), Lit::neg(bi), Lit::pos(lt_next)]);
            self.sat.add_clause([
                Lit::neg(ai),
                Lit::neg(bi),
                Lit::neg(lt_prev),
                Lit::pos(lt_next),
            ]);
            self.sat.add_clause([
                Lit::pos(ai),
                Lit::pos(bi),
                Lit::neg(lt_prev),
                Lit::pos(lt_next),
            ]);

            lt_vars.push(lt_next);
        }

        let final_lt = *lt_vars.last().expect("lt_vars should not be empty");
        self.encode_bit_eq(result, final_lt);
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

            // If b = 0, result = all 1s
            for i in 0..width {
                let ones = self.sat.new_var();
                self.sat.add_clause([Lit::neg(b_is_zero), Lit::pos(ones)]);
                self.sat
                    .add_clause([Lit::pos(b_is_zero), Lit::pos(r.bits[i]), Lit::neg(ones)]);
                self.sat
                    .add_clause([Lit::pos(b_is_zero), Lit::neg(r.bits[i]), Lit::pos(ones)]);
            }

            // For non-zero divisor, use bit-by-bit division
            // This is a simplified encoding - real implementation would be more complex
            // For now, we encode the constraint: a = result * b + remainder, remainder < b

            // Create product and remainder variables
            let mut prod_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                prod_bits.push(self.sat.new_var());
            }

            let mut rem_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                rem_bits.push(self.sat.new_var());
            }

            // Encode: prod = result * b (when b != 0)
            self.encode_conditional_mul(&prod_bits, &r.bits, &vb.bits, b_is_zero);

            // Encode: a = prod + rem (when b != 0)
            let mut sum_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                sum_bits.push(self.sat.new_var());
            }
            self.encode_adder(&sum_bits, &prod_bits, &rem_bits);

            // Assert: a = sum (when b != 0)
            for i in 0..width {
                self.sat.add_clause([
                    Lit::pos(b_is_zero),
                    Lit::neg(va.bits[i]),
                    Lit::pos(sum_bits[i]),
                ]);
                self.sat.add_clause([
                    Lit::pos(b_is_zero),
                    Lit::pos(va.bits[i]),
                    Lit::neg(sum_bits[i]),
                ]);
            }

            // Assert: rem < b (when b != 0)
            let rem_lt_b = self.sat.new_var();
            self.encode_ult_result(&rem_bits, &vb.bits, rem_lt_b);
            self.sat
                .add_clause([Lit::pos(b_is_zero), Lit::pos(rem_lt_b)]);
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

            // If b = 0, result = a
            for i in 0..width {
                self.sat.add_clause([
                    Lit::pos(b_is_zero),
                    Lit::neg(va.bits[i]),
                    Lit::pos(r.bits[i]),
                ]);
                self.sat.add_clause([
                    Lit::pos(b_is_zero),
                    Lit::pos(va.bits[i]),
                    Lit::neg(r.bits[i]),
                ]);
            }

            // For non-zero divisor: a = q * b + result, result < b
            let mut quot_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                quot_bits.push(self.sat.new_var());
            }

            let mut prod_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                prod_bits.push(self.sat.new_var());
            }

            // prod = quot * b
            self.encode_conditional_mul(&prod_bits, &quot_bits, &vb.bits, b_is_zero);

            // a = prod + result
            let mut sum_bits: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                sum_bits.push(self.sat.new_var());
            }
            self.encode_adder(&sum_bits, &prod_bits, &r.bits);

            for i in 0..width {
                self.sat.add_clause([
                    Lit::pos(b_is_zero),
                    Lit::neg(va.bits[i]),
                    Lit::pos(sum_bits[i]),
                ]);
                self.sat.add_clause([
                    Lit::pos(b_is_zero),
                    Lit::pos(va.bits[i]),
                    Lit::neg(sum_bits[i]),
                ]);
            }

            // result < b
            let rem_lt_b = self.sat.new_var();
            self.encode_ult_result(&r.bits, &vb.bits, rem_lt_b);
            self.sat
                .add_clause([Lit::pos(b_is_zero), Lit::pos(rem_lt_b)]);
        }
    }

    /// Signed division: result = a / b (signed, two's complement)
    /// If b = 0, result = all 1s. If a = INT_MIN and b = -1, result = INT_MIN
    pub fn bv_sdiv(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            // Get sign bits
            let sign_a = va.bits[width - 1];
            let sign_b = vb.bits[width - 1];

            // Compute absolute values
            let mut abs_a: SmallVec<[Var; 32]> = SmallVec::new();
            let mut abs_b: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                abs_a.push(self.sat.new_var());
                abs_b.push(self.sat.new_var());
            }

            // abs_a = sign_a ? -a : a
            let mut neg_a: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_a.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_a, &va.bits);
            for i in 0..width {
                self.encode_mux(abs_a[i], sign_a, neg_a[i], va.bits[i]);
            }

            // abs_b = sign_b ? -b : b
            let mut neg_b: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_b.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_b, &vb.bits);
            for i in 0..width {
                self.encode_mux(abs_b[i], sign_b, neg_b[i], vb.bits[i]);
            }

            // Unsigned division on absolute values
            let mut quot_abs: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                quot_abs.push(self.sat.new_var());
            }

            // Simplified: encode quot_abs * abs_b + rem = abs_a
            // For now, just a placeholder constraint
            let mut prod: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                prod.push(self.sat.new_var());
            }

            // Result sign: sign_a XOR sign_b
            let result_sign = self.sat.new_var();
            self.encode_xor(result_sign, sign_a, sign_b);

            // result = result_sign ? -quot_abs : quot_abs
            let mut neg_quot: SmallVec<[Var; 32]> = SmallVec::new();
            for _ in 0..width {
                neg_quot.push(self.sat.new_var());
            }
            self.encode_two_complement(&neg_quot, &quot_abs);

            for i in 0..width {
                self.encode_mux(r.bits[i], result_sign, neg_quot[i], quot_abs[i]);
            }
        }
    }

    /// Signed remainder: result = a % b (signed)
    /// Sign of result matches sign of dividend a
    pub fn bv_srem(&mut self, result: TermId, a: TermId, b: TermId) {
        if let (Some(va), Some(vb)) = (
            self.term_to_bv.get(&a).cloned(),
            self.term_to_bv.get(&b).cloned(),
        ) {
            assert_eq!(va.width, vb.width);
            let width = va.width as usize;
            let r = self.new_bv(result, va.width).clone();

            let sign_a = va.bits[width - 1];

            // Compute remainder with sign matching dividend
            // This is a simplified encoding

            // For now, just ensure result has same sign as a
            let result_sign = r.bits[width - 1];
            self.encode_bit_eq(result_sign, sign_a);
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

    /// Encode conditional multiplication: result = (cond ? 0 : a * b)
    fn encode_conditional_mul(&mut self, result: &[Var], a: &[Var], b: &[Var], cond: Var) {
        let width = result.len();
        assert_eq!(a.len(), width);
        assert_eq!(b.len(), width);

        // Compute a * b
        let mut prod: SmallVec<[Var; 32]> = SmallVec::new();
        for _ in 0..width {
            prod.push(self.sat.new_var());
        }

        // Simplified multiplication (shift-and-add would be full implementation)
        // For now, just use a basic constraint

        // result = cond ? 0 : prod
        for i in 0..width {
            let zero = self.sat.new_var();
            self.sat.add_clause([Lit::neg(zero)]);
            self.encode_mux(result[i], cond, zero, prod[i]);
        }
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
            SolverResult::Sat => Ok(TheoryResult::Sat),
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
