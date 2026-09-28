//! Barrel-shifter bit-blasting for `bvshl`, `bvlshr`, and `bvashr`.
//!
//! Split out of `solver.rs` to keep that file under the 2000-line policy.
//! All three encodings share the same structure: a logarithmic chain of mux
//! stages consumes the low `num_stages` bits of the shift amount, and any
//! *high* shift bit (index >= `num_stages`) forces an over-shift, whose result
//! is the SMT-LIB fill value (0 for `bvshl`/`bvlshr`, the sign bit for
//! `bvashr`).
//!
//! The high-bit handling is the soundness-critical part: without it a shift
//! amount whose only set bits are above `num_stages` (e.g. `#x10` for width 8)
//! would be silently treated as a shift by 0, encoding `bvshl x #x10` as `x`
//! instead of `0`.

use super::BvSolver;
use oxiz_core::ast::TermId;
use oxiz_sat::{Lit, Var};
use smallvec::SmallVec;

impl BvSolver {
    /// Number of mux stages a barrel shifter needs for a given width.
    ///
    /// Stage `s` (for `s` in `0..num_stages`) shifts by `2^s`, so the stages
    /// cover shift-amount bits `0..num_stages`. `num_stages = ilog2(width) + 1`
    /// guarantees the covered bits can express every in-range shift and that
    /// the smallest *uncovered* bit already has value `2^num_stages > width`,
    /// i.e. any high bit means "shift >= width".
    fn barrel_stages(width: usize) -> u32 {
        width.ilog2() + 1
    }

    /// OR together a slice of SAT variables, returning a variable that is true
    /// iff any input is true. Returns `None` for an empty slice (no over-shift
    /// bits exist for this width).
    fn encode_or_bits(&mut self, bits: &[Var]) -> Option<Var> {
        let mut acc: Option<Var> = None;
        for &b in bits {
            acc = Some(match acc {
                None => b,
                Some(prev) => {
                    let v = self.sat.new_var();
                    self.encode_or(v, prev, b);
                    v
                }
            });
        }
        acc
    }

    /// A fresh SAT variable forced to constant 0.
    fn fresh_zero(&mut self) -> Var {
        let zero = self.sat.new_var();
        self.define([Lit::neg(zero)]);
        zero
    }

    /// Wire `result` bits from `current`, but force the SMT-LIB `fill` value
    /// when `overshift` is set (shift amount >= width). When `overshift` is
    /// `None` (no high bits exist) the result is copied through directly.
    fn commit_shift_result(
        &mut self,
        result: &[Var],
        current: &[Var],
        overshift: Option<Var>,
        fill: Var,
    ) {
        for i in 0..result.len() {
            match overshift {
                Some(ov) => {
                    let out = self.sat.new_var();
                    // out = overshift ? fill : current[i]
                    self.encode_mux(out, ov, fill, current[i]);
                    self.encode_bit_eq(result[i], out);
                }
                None => self.encode_bit_eq(result[i], current[i]),
            }
        }
    }

    /// Left shift: result = a << b (SMT-LIB `bvshl`).
    ///
    /// Shift amounts >= width yield all-zero, per SMT-LIB semantics. The high
    /// bits of the shift amount participate via the over-shift detector.
    pub fn bv_shl(&mut self, result: TermId, a: TermId, shift_amount: TermId) -> bool {
        if let Some((va, shift)) = self.binop_bits(a, shift_amount) {
            let width = va.width as usize;
            if width == 0 {
                return false;
            }
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };
            let num_stages = Self::barrel_stages(width);

            let mut current = va.bits.clone();
            for s in 0..num_stages {
                let shift_by = 1usize << s;
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
                        let zero = self.fresh_zero();
                        self.encode_mux(next_bit, shift.bits[s as usize], zero, current[i]);
                    }
                    next.push(next_bit);
                }
                current = next;
            }

            // Any shift bit at or above num_stages means shift >= width -> 0.
            let overshift = self.encode_or_bits(&shift.bits[num_stages as usize..]);
            let fill = self.fresh_zero();
            self.commit_shift_result(&r.bits, &current, overshift, fill);
            true
        } else {
            false
        }
    }

    /// Logical right shift: result = a >> b (unsigned, SMT-LIB `bvlshr`).
    ///
    /// Shift amounts >= width yield all-zero. High shift bits participate via
    /// the over-shift detector.
    pub fn bv_lshr(&mut self, result: TermId, a: TermId, shift_amount: TermId) -> bool {
        if let Some((va, shift)) = self.binop_bits(a, shift_amount) {
            let width = va.width as usize;
            if width == 0 {
                return false;
            }
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };
            let num_stages = Self::barrel_stages(width);

            let mut current = va.bits.clone();
            for s in 0..num_stages {
                let shift_by = 1usize << s;
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
                        let zero = self.fresh_zero();
                        self.encode_mux(next_bit, shift.bits[s as usize], zero, current[i]);
                    }
                    next.push(next_bit);
                }
                current = next;
            }

            let overshift = self.encode_or_bits(&shift.bits[num_stages as usize..]);
            let fill = self.fresh_zero();
            self.commit_shift_result(&r.bits, &current, overshift, fill);
            true
        } else {
            false
        }
    }

    /// Arithmetic right shift: result = a >> b (signed, sign-extends;
    /// SMT-LIB `bvashr`).
    ///
    /// Shift amounts >= width yield an all-`sign` result (0 for non-negative
    /// `a`, all-ones for negative `a`). High shift bits participate via the
    /// over-shift detector, with the sign bit as the fill value.
    pub fn bv_ashr(&mut self, result: TermId, a: TermId, shift_amount: TermId) -> bool {
        if let Some((va, shift)) = self.binop_bits(a, shift_amount) {
            let width = va.width as usize;
            if width == 0 {
                return false;
            }
            let Some(r) = self.result_bits(result, va.width) else {
                return false;
            };
            let num_stages = Self::barrel_stages(width);

            // Sign bit is the fill for both in-range and over-shift cases.
            let sign = va.bits[width - 1];

            let mut current = va.bits.clone();
            for s in 0..num_stages {
                let shift_by = 1usize << s;
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

            let overshift = self.encode_or_bits(&shift.bits[num_stages as usize..]);
            self.commit_shift_result(&r.bits, &current, overshift, sign);
            true
        } else {
            false
        }
    }
}
