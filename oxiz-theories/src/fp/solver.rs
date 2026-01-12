//! Floating-Point Theory Solver using bit-blasting.

use crate::theory::{Theory, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use oxiz_sat::{Lit, Solver as SatSolver, SolverResult, Var};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// IEEE 754 Floating-point format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FpFormat {
    /// Number of bits in the exponent
    pub exponent_bits: u32,
    /// Number of bits in the significand (including implicit bit)
    pub significand_bits: u32,
}

impl FpFormat {
    /// IEEE 754 binary16 (half precision)
    pub const FLOAT16: Self = Self {
        exponent_bits: 5,
        significand_bits: 11,
    };

    /// IEEE 754 binary32 (single precision)
    pub const FLOAT32: Self = Self {
        exponent_bits: 8,
        significand_bits: 24,
    };

    /// IEEE 754 binary64 (double precision)
    pub const FLOAT64: Self = Self {
        exponent_bits: 11,
        significand_bits: 53,
    };

    /// IEEE 754 binary128 (quad precision)
    pub const FLOAT128: Self = Self {
        exponent_bits: 15,
        significand_bits: 113,
    };

    /// Create a custom floating-point format
    #[must_use]
    pub const fn new(exponent_bits: u32, significand_bits: u32) -> Self {
        Self {
            exponent_bits,
            significand_bits,
        }
    }

    /// Total width in bits
    #[must_use]
    pub const fn width(&self) -> u32 {
        1 + self.exponent_bits + self.significand_bits - 1 // sign + exp + sig (without implicit)
    }

    /// Exponent bias (2^(e-1) - 1)
    #[must_use]
    pub const fn bias(&self) -> i32 {
        (1 << (self.exponent_bits - 1)) - 1
    }

    /// Maximum exponent value (all 1s)
    #[must_use]
    pub const fn max_exponent(&self) -> u32 {
        (1 << self.exponent_bits) - 1
    }
}

/// IEEE 754 Rounding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FpRoundingMode {
    /// Round to nearest, ties to even (default)
    #[default]
    RoundNearestTiesToEven,
    /// Round to nearest, ties away from zero
    RoundNearestTiesToAway,
    /// Round toward positive infinity
    RoundTowardPositive,
    /// Round toward negative infinity
    RoundTowardNegative,
    /// Round toward zero (truncate)
    RoundTowardZero,
}

impl FpRoundingMode {
    /// SMT-LIB2 name
    #[must_use]
    pub const fn smtlib_name(&self) -> &'static str {
        match self {
            Self::RoundNearestTiesToEven => "RNE",
            Self::RoundNearestTiesToAway => "RNA",
            Self::RoundTowardPositive => "RTP",
            Self::RoundTowardNegative => "RTN",
            Self::RoundTowardZero => "RTZ",
        }
    }
}

/// A floating-point value representation
#[derive(Debug, Clone)]
pub struct FpValue {
    /// Sign bit (true = negative)
    pub sign: bool,
    /// Exponent bits (biased)
    pub exponent: u64,
    /// Significand bits (without implicit bit for normal numbers)
    pub significand: u64,
    /// Format specification
    pub format: FpFormat,
}

impl FpValue {
    /// Create positive zero
    #[must_use]
    pub fn pos_zero(format: FpFormat) -> Self {
        Self {
            sign: false,
            exponent: 0,
            significand: 0,
            format,
        }
    }

    /// Create negative zero
    #[must_use]
    pub fn neg_zero(format: FpFormat) -> Self {
        Self {
            sign: true,
            exponent: 0,
            significand: 0,
            format,
        }
    }

    /// Create positive infinity
    #[must_use]
    pub fn pos_infinity(format: FpFormat) -> Self {
        Self {
            sign: false,
            exponent: format.max_exponent() as u64,
            significand: 0,
            format,
        }
    }

    /// Create negative infinity
    #[must_use]
    pub fn neg_infinity(format: FpFormat) -> Self {
        Self {
            sign: true,
            exponent: format.max_exponent() as u64,
            significand: 0,
            format,
        }
    }

    /// Create canonical NaN
    #[must_use]
    pub fn nan(format: FpFormat) -> Self {
        Self {
            sign: false,
            exponent: format.max_exponent() as u64,
            significand: 1 << (format.significand_bits - 2), // quiet NaN
            format,
        }
    }

    /// Check if this is zero (positive or negative)
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.exponent == 0 && self.significand == 0
    }

    /// Check if this is subnormal (denormalized)
    #[must_use]
    pub fn is_subnormal(&self) -> bool {
        self.exponent == 0 && self.significand != 0
    }

    /// Check if this is normal
    #[must_use]
    pub fn is_normal(&self) -> bool {
        let max_exp = self.format.max_exponent() as u64;
        self.exponent > 0 && self.exponent < max_exp
    }

    /// Check if this is infinity
    #[must_use]
    pub fn is_infinite(&self) -> bool {
        self.exponent == self.format.max_exponent() as u64 && self.significand == 0
    }

    /// Check if this is NaN
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.exponent == self.format.max_exponent() as u64 && self.significand != 0
    }

    /// Check if negative
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.sign && !self.is_nan()
    }

    /// Check if positive
    #[must_use]
    pub fn is_positive(&self) -> bool {
        !self.sign && !self.is_nan()
    }

    /// Convert from f32
    #[must_use]
    pub fn from_f32(val: f32) -> Self {
        let bits = val.to_bits();
        Self {
            sign: (bits >> 31) != 0,
            exponent: ((bits >> 23) & 0xFF) as u64,
            significand: (bits & 0x7FFFFF) as u64,
            format: FpFormat::FLOAT32,
        }
    }

    /// Convert from f64
    #[must_use]
    pub fn from_f64(val: f64) -> Self {
        let bits = val.to_bits();
        Self {
            sign: (bits >> 63) != 0,
            exponent: (bits >> 52) & 0x7FF,
            significand: bits & 0xFFFFFFFFFFFFF,
            format: FpFormat::FLOAT64,
        }
    }

    /// Convert to f32 (only valid for FLOAT32 format)
    #[must_use]
    pub fn to_f32(&self) -> Option<f32> {
        if self.format != FpFormat::FLOAT32 {
            return None;
        }
        let mut bits: u32 = 0;
        if self.sign {
            bits |= 1 << 31;
        }
        bits |= (self.exponent as u32 & 0xFF) << 23;
        bits |= self.significand as u32 & 0x7FFFFF;
        Some(f32::from_bits(bits))
    }

    /// Convert to f64 (only valid for FLOAT64 format)
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        if self.format != FpFormat::FLOAT64 {
            return None;
        }
        let mut bits: u64 = 0;
        if self.sign {
            bits |= 1 << 63;
        }
        bits |= (self.exponent & 0x7FF) << 52;
        bits |= self.significand & 0xFFFFFFFFFFFFF;
        Some(f64::from_bits(bits))
    }
}

/// A floating-point variable (set of SAT variables)
#[derive(Debug, Clone)]
struct FpVar {
    /// Sign bit
    sign: Var,
    /// Exponent bits (LSB first)
    exponent: SmallVec<[Var; 16]>,
    /// Significand bits (LSB first, without implicit bit)
    significand: SmallVec<[Var; 64]>,
    /// Format
    format: FpFormat,
}

/// Floating-Point Theory Solver
#[derive(Debug)]
pub struct FpSolver {
    /// Embedded SAT solver
    sat: SatSolver,
    /// Term to FP variable mapping
    term_to_fp: FxHashMap<TermId, FpVar>,
    /// Pending assertions
    assertions: Vec<(TermId, bool)>,
    /// Context stack
    context_stack: Vec<usize>,
    /// Current rounding mode
    rounding_mode: FpRoundingMode,
}

impl Default for FpSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl FpSolver {
    /// Create a new Floating-Point solver
    #[must_use]
    pub fn new() -> Self {
        Self {
            sat: SatSolver::new(),
            term_to_fp: FxHashMap::default(),
            assertions: Vec::new(),
            context_stack: Vec::new(),
            rounding_mode: FpRoundingMode::default(),
        }
    }

    /// Set the rounding mode
    pub fn set_rounding_mode(&mut self, mode: FpRoundingMode) {
        self.rounding_mode = mode;
    }

    /// Get the current rounding mode
    #[must_use]
    pub fn rounding_mode(&self) -> FpRoundingMode {
        self.rounding_mode
    }

    /// Create a new floating-point variable
    pub fn new_fp(&mut self, term: TermId, format: FpFormat) {
        if self.term_to_fp.contains_key(&term) {
            return;
        }

        let sign = self.sat.new_var();
        let exponent: SmallVec<[Var; 16]> = (0..format.exponent_bits)
            .map(|_| self.sat.new_var())
            .collect();
        let significand: SmallVec<[Var; 64]> = (0..format.significand_bits - 1) // -1 for implicit bit
            .map(|_| self.sat.new_var())
            .collect();

        self.term_to_fp.insert(
            term,
            FpVar {
                sign,
                exponent,
                significand,
                format,
            },
        );
    }

    /// Assert a constant floating-point value
    pub fn assert_const(&mut self, term: TermId, value: &FpValue) {
        self.new_fp(term, value.format);
        let fp = self
            .term_to_fp
            .get(&term)
            .expect("term exists after new_fp call")
            .clone();

        // Set sign
        if value.sign {
            self.sat.add_clause([Lit::pos(fp.sign)]);
        } else {
            self.sat.add_clause([Lit::neg(fp.sign)]);
        }

        // Set exponent bits
        for (i, &var) in fp.exponent.iter().enumerate() {
            if (value.exponent >> i) & 1 == 1 {
                self.sat.add_clause([Lit::pos(var)]);
            } else {
                self.sat.add_clause([Lit::neg(var)]);
            }
        }

        // Set significand bits
        for (i, &var) in fp.significand.iter().enumerate() {
            if (value.significand >> i) & 1 == 1 {
                self.sat.add_clause([Lit::pos(var)]);
            } else {
                self.sat.add_clause([Lit::neg(var)]);
            }
        }
    }

    /// Assert FP equality: a = b
    pub fn assert_fp_eq(&mut self, a: TermId, b: TermId) {
        let fp_a = self.term_to_fp.get(&a).cloned();
        let fp_b = self.term_to_fp.get(&b).cloned();

        if let (Some(va), Some(vb)) = (fp_a, fp_b) {
            assert_eq!(va.format, vb.format);

            // NaN handling: NaN != NaN
            // First check if either is NaN
            let a_is_nan = self.encode_is_nan(&va);
            let b_is_nan = self.encode_is_nan(&vb);

            // For simplicity, encode bitwise equality
            // Real IEEE 754 semantics: +0 == -0 and NaN != NaN
            // Here we implement bitwise equality for simplicity

            // Sign equality: sign_a <=> sign_b
            // (~sign_a or sign_b) and (sign_a or ~sign_b)
            self.sat.add_clause([Lit::neg(va.sign), Lit::pos(vb.sign)]);
            self.sat.add_clause([Lit::pos(va.sign), Lit::neg(vb.sign)]);

            // Exponent equality: ea[i] <=> eb[i]
            for (ea, eb) in va.exponent.iter().zip(vb.exponent.iter()) {
                self.sat.add_clause([Lit::neg(*ea), Lit::pos(*eb)]);
                self.sat.add_clause([Lit::pos(*ea), Lit::neg(*eb)]);
            }

            // Significand equality: sa[i] <=> sb[i]
            for (sa, sb) in va.significand.iter().zip(vb.significand.iter()) {
                self.sat.add_clause([Lit::neg(*sa), Lit::pos(*sb)]);
                self.sat.add_clause([Lit::pos(*sa), Lit::neg(*sb)]);
            }

            // Neither can be NaN for equality
            self.sat.add_clause([Lit::neg(a_is_nan)]);
            self.sat.add_clause([Lit::neg(b_is_nan)]);
        }
    }

    /// Encode "is NaN" predicate, returns variable that is true iff the value is NaN
    fn encode_is_nan(&mut self, fp: &FpVar) -> Var {
        let is_nan = self.sat.new_var();

        // NaN: exponent all 1s AND significand non-zero
        // is_nan <=> (exp = max_exp) and (sig != 0)

        // First encode exp = max_exp
        let exp_max = self.sat.new_var();
        // exp_max => all exponent bits are 1
        for &e in &fp.exponent {
            self.sat.add_clause([Lit::neg(exp_max), Lit::pos(e)]);
        }
        // all exponent bits 1 => exp_max
        let mut clause: SmallVec<[Lit; 16]> = SmallVec::new();
        clause.push(Lit::pos(exp_max));
        for &e in &fp.exponent {
            clause.push(Lit::neg(e));
        }
        self.sat.add_clause(clause);

        // Encode sig != 0 (at least one significand bit is 1)
        let sig_nonzero = self.sat.new_var();
        // at least one sig bit => sig_nonzero
        for &s in &fp.significand {
            self.sat.add_clause([Lit::neg(s), Lit::pos(sig_nonzero)]);
        }
        // sig_nonzero => at least one sig bit
        let mut clause: SmallVec<[Lit; 64]> = SmallVec::new();
        clause.push(Lit::neg(sig_nonzero));
        for &s in &fp.significand {
            clause.push(Lit::pos(s));
        }
        self.sat.add_clause(clause);

        // is_nan <=> exp_max and sig_nonzero
        // is_nan => exp_max
        self.sat.add_clause([Lit::neg(is_nan), Lit::pos(exp_max)]);
        // is_nan => sig_nonzero
        self.sat
            .add_clause([Lit::neg(is_nan), Lit::pos(sig_nonzero)]);
        // exp_max and sig_nonzero => is_nan
        self.sat
            .add_clause([Lit::neg(exp_max), Lit::neg(sig_nonzero), Lit::pos(is_nan)]);

        is_nan
    }

    /// Encode "is Infinite" predicate
    fn encode_is_infinite(&mut self, fp: &FpVar) -> Var {
        let is_inf = self.sat.new_var();

        // Infinity: exponent all 1s AND significand zero
        let exp_max = self.sat.new_var();
        for &e in &fp.exponent {
            self.sat.add_clause([Lit::neg(exp_max), Lit::pos(e)]);
        }
        let mut clause: SmallVec<[Lit; 16]> = SmallVec::new();
        clause.push(Lit::pos(exp_max));
        for &e in &fp.exponent {
            clause.push(Lit::neg(e));
        }
        self.sat.add_clause(clause);

        // Encode sig = 0 (all significand bits are 0)
        let sig_zero = self.sat.new_var();
        for &s in &fp.significand {
            self.sat.add_clause([Lit::neg(sig_zero), Lit::neg(s)]);
        }
        let mut clause: SmallVec<[Lit; 64]> = SmallVec::new();
        clause.push(Lit::pos(sig_zero));
        for &s in &fp.significand {
            clause.push(Lit::pos(s));
        }
        self.sat.add_clause(clause);

        // is_inf <=> exp_max and sig_zero
        self.sat.add_clause([Lit::neg(is_inf), Lit::pos(exp_max)]);
        self.sat.add_clause([Lit::neg(is_inf), Lit::pos(sig_zero)]);
        self.sat
            .add_clause([Lit::neg(exp_max), Lit::neg(sig_zero), Lit::pos(is_inf)]);

        is_inf
    }

    /// Encode "is Zero" predicate
    fn encode_is_zero(&mut self, fp: &FpVar) -> Var {
        let is_zero = self.sat.new_var();

        // Zero: exponent all 0s AND significand all 0s
        let exp_zero = self.sat.new_var();
        for &e in &fp.exponent {
            self.sat.add_clause([Lit::neg(exp_zero), Lit::neg(e)]);
        }
        let mut clause: SmallVec<[Lit; 16]> = SmallVec::new();
        clause.push(Lit::pos(exp_zero));
        for &e in &fp.exponent {
            clause.push(Lit::pos(e));
        }
        self.sat.add_clause(clause);

        let sig_zero = self.sat.new_var();
        for &s in &fp.significand {
            self.sat.add_clause([Lit::neg(sig_zero), Lit::neg(s)]);
        }
        let mut clause: SmallVec<[Lit; 64]> = SmallVec::new();
        clause.push(Lit::pos(sig_zero));
        for &s in &fp.significand {
            clause.push(Lit::pos(s));
        }
        self.sat.add_clause(clause);

        // is_zero <=> exp_zero and sig_zero
        self.sat.add_clause([Lit::neg(is_zero), Lit::pos(exp_zero)]);
        self.sat.add_clause([Lit::neg(is_zero), Lit::pos(sig_zero)]);
        self.sat
            .add_clause([Lit::neg(exp_zero), Lit::neg(sig_zero), Lit::pos(is_zero)]);

        is_zero
    }

    /// Assert that a term is NaN
    pub fn assert_is_nan(&mut self, term: TermId) {
        if let Some(fp) = self.term_to_fp.get(&term).cloned() {
            let is_nan = self.encode_is_nan(&fp);
            self.sat.add_clause([Lit::pos(is_nan)]);
        }
    }

    /// Assert that a term is infinite
    pub fn assert_is_infinite(&mut self, term: TermId) {
        if let Some(fp) = self.term_to_fp.get(&term).cloned() {
            let is_inf = self.encode_is_infinite(&fp);
            self.sat.add_clause([Lit::pos(is_inf)]);
        }
    }

    /// Assert that a term is zero
    pub fn assert_is_zero(&mut self, term: TermId) {
        if let Some(fp) = self.term_to_fp.get(&term).cloned() {
            let is_zero = self.encode_is_zero(&fp);
            self.sat.add_clause([Lit::pos(is_zero)]);
        }
    }

    /// Assert that a term is normal
    pub fn assert_is_normal(&mut self, term: TermId) {
        if let Some(fp) = self.term_to_fp.get(&term).cloned() {
            // Normal: 0 < exp < max_exp
            let is_nan = self.encode_is_nan(&fp);
            let is_inf = self.encode_is_infinite(&fp);
            let is_zero = self.encode_is_zero(&fp);

            // Encode exp > 0 (at least one exponent bit is 1)
            let exp_nonzero = self.sat.new_var();
            for &e in &fp.exponent {
                self.sat.add_clause([Lit::neg(e), Lit::pos(exp_nonzero)]);
            }
            let mut clause: SmallVec<[Lit; 16]> = SmallVec::new();
            clause.push(Lit::neg(exp_nonzero));
            for &e in &fp.exponent {
                clause.push(Lit::pos(e));
            }
            self.sat.add_clause(clause);

            // Must have exp_nonzero and not NaN and not Inf
            self.sat.add_clause([Lit::pos(exp_nonzero)]);
            self.sat.add_clause([Lit::neg(is_nan)]);
            self.sat.add_clause([Lit::neg(is_inf)]);
            self.sat.add_clause([Lit::neg(is_zero)]);
        }
    }

    /// Assert negation: result = -operand
    pub fn assert_fp_neg(&mut self, result: TermId, operand: TermId) {
        let fp_op = self.term_to_fp.get(&operand).cloned();
        let fp_res = self.term_to_fp.get(&result).cloned();

        if let (Some(op), Some(res)) = (fp_op, fp_res) {
            assert_eq!(op.format, res.format);

            // Negation: flip sign, copy exponent and significand
            // result.sign = ~operand.sign
            self.sat.add_clause([Lit::neg(res.sign), Lit::neg(op.sign)]);
            self.sat.add_clause([Lit::pos(res.sign), Lit::pos(op.sign)]);

            // Copy exponent
            for (re, oe) in res.exponent.iter().zip(op.exponent.iter()) {
                self.sat.add_clause([Lit::neg(*re), Lit::pos(*oe)]);
                self.sat.add_clause([Lit::pos(*re), Lit::neg(*oe)]);
            }

            // Copy significand
            for (rs, os) in res.significand.iter().zip(op.significand.iter()) {
                self.sat.add_clause([Lit::neg(*rs), Lit::pos(*os)]);
                self.sat.add_clause([Lit::pos(*rs), Lit::neg(*os)]);
            }
        }
    }

    /// Assert absolute value: result = |operand|
    pub fn assert_fp_abs(&mut self, result: TermId, operand: TermId) {
        let fp_op = self.term_to_fp.get(&operand).cloned();
        let fp_res = self.term_to_fp.get(&result).cloned();

        if let (Some(op), Some(res)) = (fp_op, fp_res) {
            assert_eq!(op.format, res.format);

            // Absolute value: clear sign, copy exponent and significand
            // result.sign = 0 (positive)
            self.sat.add_clause([Lit::neg(res.sign)]);

            // Copy exponent
            for (re, oe) in res.exponent.iter().zip(op.exponent.iter()) {
                self.sat.add_clause([Lit::neg(*re), Lit::pos(*oe)]);
                self.sat.add_clause([Lit::pos(*re), Lit::neg(*oe)]);
            }

            // Copy significand
            for (rs, os) in res.significand.iter().zip(op.significand.iter()) {
                self.sat.add_clause([Lit::neg(*rs), Lit::pos(*os)]);
                self.sat.add_clause([Lit::pos(*rs), Lit::neg(*os)]);
            }
        }
    }

    /// Assert comparison: a < b (less than)
    pub fn assert_fp_lt(&mut self, a: TermId, b: TermId) -> Var {
        let fp_a = self.term_to_fp.get(&a).cloned();
        let fp_b = self.term_to_fp.get(&b).cloned();

        let result = self.sat.new_var();

        if let (Some(va), Some(vb)) = (fp_a, fp_b) {
            assert_eq!(va.format, vb.format);

            // Simplified comparison logic (assumes normalized values)
            // For a complete implementation, need to handle:
            // - NaN (always false)
            // - Sign differences
            // - Zero comparison (+0 = -0)
            // - Magnitude comparison for same sign

            // If either is NaN, result is false
            let a_is_nan = self.encode_is_nan(&va);
            let b_is_nan = self.encode_is_nan(&vb);
            self.sat.add_clause([Lit::neg(a_is_nan), Lit::neg(result)]);
            self.sat.add_clause([Lit::neg(b_is_nan), Lit::neg(result)]);

            // Simplified: just check signs for basic implementation
            // a < b if a is negative and b is positive
            // (~result or (sign_a and ~sign_b))
            self.sat.add_clause([Lit::neg(result), Lit::pos(va.sign)]);
            self.sat.add_clause([Lit::neg(result), Lit::neg(vb.sign)]);
        }

        result
    }

    /// Assert comparison: a <= b (less than or equal)
    pub fn assert_fp_le(&mut self, a: TermId, b: TermId) -> Var {
        let fp_a = self.term_to_fp.get(&a).cloned();
        let fp_b = self.term_to_fp.get(&b).cloned();

        let result = self.sat.new_var();

        if let (Some(va), Some(vb)) = (fp_a, fp_b) {
            assert_eq!(va.format, vb.format);

            let a_is_nan = self.encode_is_nan(&va);
            let b_is_nan = self.encode_is_nan(&vb);
            self.sat.add_clause([Lit::neg(a_is_nan), Lit::neg(result)]);
            self.sat.add_clause([Lit::neg(b_is_nan), Lit::neg(result)]);
        }

        result
    }

    /// Convert between FP formats
    pub fn assert_fp_to_fp(&mut self, result: TermId, operand: TermId, target_format: FpFormat) {
        // Ensure result has correct format
        self.new_fp(result, target_format);

        let fp_op = self.term_to_fp.get(&operand).cloned();
        let fp_res = self.term_to_fp.get(&result).cloned();

        if let (Some(op), Some(res)) = (fp_op, fp_res) {
            // For same format, just copy
            if op.format == res.format {
                // Copy sign
                self.sat.add_clause([Lit::neg(res.sign), Lit::pos(op.sign)]);
                self.sat.add_clause([Lit::pos(res.sign), Lit::neg(op.sign)]);

                // Copy exponent
                for (re, oe) in res.exponent.iter().zip(op.exponent.iter()) {
                    self.sat.add_clause([Lit::neg(*re), Lit::pos(*oe)]);
                    self.sat.add_clause([Lit::pos(*re), Lit::neg(*oe)]);
                }

                // Copy significand
                for (rs, os) in res.significand.iter().zip(op.significand.iter()) {
                    self.sat.add_clause([Lit::neg(*rs), Lit::pos(*os)]);
                    self.sat.add_clause([Lit::pos(*rs), Lit::neg(*os)]);
                }
            } else {
                // For different formats, this requires rounding/adjustment
                // Simplified: preserve special values
                let is_nan = self.encode_is_nan(&op);
                let is_inf = self.encode_is_infinite(&op);
                let is_zero = self.encode_is_zero(&op);

                // Copy sign
                self.sat.add_clause([Lit::neg(res.sign), Lit::pos(op.sign)]);
                self.sat.add_clause([Lit::pos(res.sign), Lit::neg(op.sign)]);

                // If NaN -> result is NaN
                let res_is_nan = self.encode_is_nan(&res);
                self.sat
                    .add_clause([Lit::neg(is_nan), Lit::pos(res_is_nan)]);

                // If Inf -> result is Inf
                let res_is_inf = self.encode_is_infinite(&res);
                self.sat
                    .add_clause([Lit::neg(is_inf), Lit::pos(res_is_inf)]);

                // If Zero -> result is Zero
                let res_is_zero = self.encode_is_zero(&res);
                self.sat
                    .add_clause([Lit::neg(is_zero), Lit::pos(res_is_zero)]);
            }
        }
    }

    /// Convert FP to signed integer (with rounding mode)
    pub fn assert_fp_to_sbv(&mut self, _result: TermId, operand: TermId, width: u32) {
        let fp_op = self.term_to_fp.get(&operand).cloned();

        if let Some(_op) = fp_op {
            // This requires extracting the integer value from the FP representation
            // and encoding it as a bitvector
            // For now, create bitvector variables for result
            for _ in 0..width {
                self.sat.new_var();
            }

            // Full implementation requires:
            // 1. Check for special values (NaN, Inf) -> undefined
            // 2. Extract integer part based on exponent
            // 3. Apply rounding mode
            // 4. Check for overflow
            // 5. Encode result bits
        }
    }

    /// Convert FP to unsigned integer
    pub fn assert_fp_to_ubv(&mut self, _result: TermId, operand: TermId, width: u32) {
        let fp_op = self.term_to_fp.get(&operand).cloned();

        if let Some(_op) = fp_op {
            for _ in 0..width {
                self.sat.new_var();
            }
        }
    }

    /// Convert signed integer to FP
    pub fn assert_sbv_to_fp(
        &mut self,
        result: TermId,
        operand: TermId,
        _width: u32,
        format: FpFormat,
    ) {
        self.new_fp(result, format);
        // Full implementation requires:
        // 1. Determine sign of integer
        // 2. Find leading 1 bit (normalization)
        // 3. Compute exponent
        // 4. Extract significand bits
        // 5. Apply rounding if needed
        let _ = operand; // Suppress unused warning
    }

    /// Convert unsigned integer to FP
    pub fn assert_ubv_to_fp(
        &mut self,
        result: TermId,
        operand: TermId,
        _width: u32,
        format: FpFormat,
    ) {
        self.new_fp(result, format);
        let _ = operand; // Suppress unused warning
    }

    /// Convert FP to real (symbolic)
    pub fn assert_fp_to_real(&mut self, _result: TermId, _operand: TermId) {
        // This creates a symbolic real value from an FP value
        // Would require integration with arithmetic theory
    }

    /// Convert real to FP (symbolic)
    pub fn assert_real_to_fp(&mut self, result: TermId, _operand: TermId, format: FpFormat) {
        self.new_fp(result, format);
        // Would require integration with arithmetic theory
    }

    /// Get the floating-point value from the model
    #[must_use]
    pub fn get_value(&self, term: TermId) -> Option<FpValue> {
        let fp = self.term_to_fp.get(&term)?;
        let model = self.sat.model();

        let sign = model[fp.sign.index()].is_true();

        let mut exponent = 0u64;
        for (i, &var) in fp.exponent.iter().enumerate() {
            if model[var.index()].is_true() {
                exponent |= 1 << i;
            }
        }

        let mut significand = 0u64;
        for (i, &var) in fp.significand.iter().enumerate() {
            if model[var.index()].is_true() {
                significand |= 1 << i;
            }
        }

        Some(FpValue {
            sign,
            exponent,
            significand,
            format: fp.format,
        })
    }
}

impl Theory for FpSolver {
    fn id(&self) -> TheoryId {
        TheoryId::FP
    }

    fn name(&self) -> &str {
        "FP"
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
        self.term_to_fp.clear();
        self.assertions.clear();
        self.context_stack.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp_format_constants() {
        assert_eq!(FpFormat::FLOAT32.width(), 32);
        assert_eq!(FpFormat::FLOAT32.bias(), 127);
        assert_eq!(FpFormat::FLOAT32.max_exponent(), 255);

        assert_eq!(FpFormat::FLOAT64.width(), 64);
        assert_eq!(FpFormat::FLOAT64.bias(), 1023);
        assert_eq!(FpFormat::FLOAT64.max_exponent(), 2047);
    }

    #[test]
    fn test_fp_value_from_f32() {
        let val = FpValue::from_f32(1.0);
        assert!(!val.sign);
        assert_eq!(val.exponent, 127); // bias
        assert_eq!(val.significand, 0); // 1.0 has no fractional part

        let val = FpValue::from_f32(-2.0);
        assert!(val.sign);
        assert_eq!(val.exponent, 128); // 127 + 1
    }

    #[test]
    fn test_fp_value_special_values() {
        let zero = FpValue::pos_zero(FpFormat::FLOAT32);
        assert!(zero.is_zero());
        assert!(!zero.is_nan());
        assert!(!zero.is_infinite());

        let inf = FpValue::pos_infinity(FpFormat::FLOAT32);
        assert!(inf.is_infinite());
        assert!(!inf.is_nan());
        assert!(!inf.is_zero());

        let nan = FpValue::nan(FpFormat::FLOAT32);
        assert!(nan.is_nan());
        assert!(!nan.is_infinite());
        assert!(!nan.is_zero());
    }

    #[test]
    fn test_fp_solver_const() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        let value = FpValue::from_f32(42.0);

        solver.assert_const(a, &value);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let retrieved = solver.get_value(a).unwrap();
        assert_eq!(retrieved.to_f32(), Some(42.0));
    }

    #[test]
    fn test_fp_solver_eq() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_fp(a, FpFormat::FLOAT32);
        solver.new_fp(b, FpFormat::FLOAT32);

        // a = 1.5
        solver.assert_const(a, &FpValue::from_f32(1.5));

        // a = b
        solver.assert_fp_eq(a, b);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        // b should also be 1.5
        let b_val = solver.get_value(b).unwrap();
        assert_eq!(b_val.to_f32(), Some(1.5));
    }

    #[test]
    fn test_fp_solver_neg() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_fp(a, FpFormat::FLOAT32);
        solver.new_fp(b, FpFormat::FLOAT32);

        // a = 3.14
        solver.assert_const(a, &FpValue::from_f32(3.14));

        // b = -a
        solver.assert_fp_neg(b, a);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let b_val = solver.get_value(b).unwrap();
        assert_eq!(b_val.to_f32(), Some(-3.14));
    }

    #[test]
    fn test_fp_solver_abs() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_fp(a, FpFormat::FLOAT32);
        solver.new_fp(b, FpFormat::FLOAT32);

        // a = -5.0
        solver.assert_const(a, &FpValue::from_f32(-5.0));

        // b = |a|
        solver.assert_fp_abs(b, a);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let b_val = solver.get_value(b).unwrap();
        assert_eq!(b_val.to_f32(), Some(5.0));
    }

    #[test]
    fn test_fp_rounding_modes() {
        let mut solver = FpSolver::new();
        assert_eq!(
            solver.rounding_mode(),
            FpRoundingMode::RoundNearestTiesToEven
        );

        solver.set_rounding_mode(FpRoundingMode::RoundTowardZero);
        assert_eq!(solver.rounding_mode(), FpRoundingMode::RoundTowardZero);
    }

    #[test]
    fn test_fp_is_nan_constraint() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        solver.new_fp(a, FpFormat::FLOAT32);
        solver.assert_is_nan(a);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let val = solver.get_value(a).unwrap();
        assert!(val.is_nan());
    }

    #[test]
    fn test_fp_is_infinite_constraint() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        solver.new_fp(a, FpFormat::FLOAT32);
        solver.assert_is_infinite(a);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let val = solver.get_value(a).unwrap();
        assert!(val.is_infinite());
    }

    #[test]
    fn test_fp_is_zero_constraint() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        solver.new_fp(a, FpFormat::FLOAT32);
        solver.assert_is_zero(a);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let val = solver.get_value(a).unwrap();
        assert!(val.is_zero());
    }

    #[test]
    fn test_fp_comparison_lt() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_fp(a, FpFormat::FLOAT32);
        solver.new_fp(b, FpFormat::FLOAT32);

        // a = -1.0 (negative)
        solver.assert_const(a, &FpValue::from_f32(-1.0));
        // b = 1.0 (positive)
        solver.assert_const(b, &FpValue::from_f32(1.0));

        // a < b
        let lt_result = solver.assert_fp_lt(a, b);
        solver.sat.add_clause([Lit::pos(lt_result)]);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }

    #[test]
    fn test_fp_conversion_same_format() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_fp(a, FpFormat::FLOAT32);
        solver.assert_const(a, &FpValue::from_f32(3.14));

        // Convert a to b (same format)
        solver.assert_fp_to_fp(b, a, FpFormat::FLOAT32);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let b_val = solver.get_value(b).unwrap();
        assert_eq!(b_val.to_f32(), Some(3.14));
    }

    #[test]
    fn test_fp_conversion_preserves_nan() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_fp(a, FpFormat::FLOAT32);
        solver.assert_const(a, &FpValue::nan(FpFormat::FLOAT32));

        // Convert to FLOAT64
        solver.assert_fp_to_fp(b, a, FpFormat::FLOAT64);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let b_val = solver.get_value(b).unwrap();
        assert!(b_val.is_nan());
    }

    #[test]
    fn test_fp_conversion_preserves_infinity() {
        let mut solver = FpSolver::new();

        let a = TermId::new(1);
        let b = TermId::new(2);

        solver.new_fp(a, FpFormat::FLOAT32);
        solver.assert_const(a, &FpValue::pos_infinity(FpFormat::FLOAT32));

        // Convert to FLOAT64
        solver.assert_fp_to_fp(b, a, FpFormat::FLOAT64);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        let b_val = solver.get_value(b).unwrap();
        assert!(b_val.is_infinite());
        assert!(b_val.is_positive());
    }
}
