//! Indexed-identifier term construction for the SMT-LIB2 parser.
//!
//! SMT-LIB indexed identifiers — `(_ f i ...)` — are theory constructs, never
//! user declarations. This module builds the ones whose lowering is purely
//! syntactic (the bit-vector extend/rotate/repeat family, `divisible`, the
//! regular-expression repetition operators) and the floating-point conversions
//! that take a leading rounding-mode symbol. Split out of `terms.rs` to keep
//! that file under the 2000-line limit.

use super::super::lexer::TokenKind;
use super::Parser;
use super::commands::{RmOperand, is_rounding_mode_literal};
use crate::ast::{RoundingMode, TermId};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortKind;
use num_bigint::BigInt;

impl Parser<'_> {
    /// Attempt to build an indexed operator whose name/indices/arguments have
    /// already been parsed.
    ///
    /// Handles the indexed bit-vector operators that have no dedicated
    /// [`crate::ast::TermKind`] (`zero_extend`, `sign_extend`, `rotate_left`,
    /// `rotate_right`, `repeat`) by lowering them to existing primitives
    /// (`concat`, `extract`), and the arithmetic `divisible` predicate.
    ///
    /// Returns `Ok(Some(term))` when the operator was recognized and built,
    /// `Ok(None)` when it is not one of these operators (so the caller can fall
    /// back to a generic application), or `Err(..)` on a malformed use.
    pub(super) fn build_indexed_op(
        &mut self,
        name: &str,
        index_parts: &[String],
        args: &[TermId],
    ) -> Result<Option<TermId>> {
        // Parse the leading numeric index shared by every operator here.
        let single_index = |parts: &[String]| -> Result<u32> {
            if parts.len() != 1 {
                return Err(OxizError::ParseError {
                    position: 0,
                    message: format!(
                        "(_ {name} ...) requires exactly 1 index, got {}",
                        parts.len()
                    ),
                });
            }
            parts[0].parse::<u32>().map_err(|_| OxizError::ParseError {
                position: 0,
                message: format!("invalid index for (_ {name} ...): {}", parts[0]),
            })
        };
        let single_arg = |args: &[TermId]| -> Result<TermId> {
            if args.len() != 1 {
                return Err(OxizError::ParseError {
                    position: 0,
                    message: format!(
                        "(_ {name} ...) requires exactly 1 argument, got {}",
                        args.len()
                    ),
                });
            }
            Ok(args[0])
        };

        match name {
            // `((_ extract i j) x)` — the standard SMT-LIB spelling, where the
            // indexed identifier is its own S-expression.  Without this arm it
            // fell through to the `Bool`-sorted uninterpreted-application
            // fallback below, so `(= ((_ extract 3 0) #xab) #xc)` answered
            // `sat` and `(concat #x0 ((_ extract 3 0) x))` could not even
            // determine an operand width.  (The flattened `(_ extract i j x)`
            // spelling is handled separately by the caller.)
            "extract" => {
                if index_parts.len() != 2 {
                    return Err(OxizError::ParseError {
                        position: 0,
                        message: format!(
                            "(_ extract ...) requires exactly 2 indices, got {}",
                            index_parts.len()
                        ),
                    });
                }
                let parse_index = |raw: &String| -> Result<u32> {
                    raw.parse::<u32>().map_err(|_| OxizError::ParseError {
                        position: 0,
                        message: format!("invalid index for (_ extract ...): {raw}"),
                    })
                };
                let high = parse_index(&index_parts[0])?;
                let low = parse_index(&index_parts[1])?;
                let arg = single_arg(args)?;
                if low > high {
                    return Err(OxizError::ParseError {
                        position: 0,
                        message: format!("(_ extract {high} {low}) requires {high} >= {low}"),
                    });
                }
                let width = self.bv_width(arg).ok_or_else(|| OxizError::ParseError {
                    position: 0,
                    message: "extract requires a bit-vector argument".to_string(),
                })?;
                if high >= width {
                    return Err(OxizError::ParseError {
                        position: 0,
                        message: format!(
                            "(_ extract {high} {low}) is out of range for a {width}-bit operand"
                        ),
                    });
                }
                Ok(Some(self.manager.mk_bv_extract(high, low, arg)))
            }
            "zero_extend" => {
                let n = single_index(index_parts)?;
                let arg = single_arg(args)?;
                if n == 0 {
                    return Ok(Some(arg));
                }
                // Prepend `n` zero bits: concat(0:n, arg).
                let zeros = self.manager.mk_bitvec(0, n);
                Ok(Some(self.manager.try_mk_bv_concat(zeros, arg)?))
            }
            "sign_extend" => {
                let n = single_index(index_parts)?;
                let arg = single_arg(args)?;
                if n == 0 {
                    return Ok(Some(arg));
                }
                let width = self.bv_width(arg).ok_or_else(|| OxizError::ParseError {
                    position: 0,
                    message: "sign_extend requires a bit-vector argument".to_string(),
                })?;
                // Replicate the sign bit `n` times, then concat with the arg.
                let sign_bit = self.manager.mk_bv_extract(width - 1, width - 1, arg);
                let mut ext = sign_bit;
                for _ in 1..n {
                    ext = self.manager.try_mk_bv_concat(ext, sign_bit)?;
                }
                Ok(Some(self.manager.try_mk_bv_concat(ext, arg)?))
            }
            "rotate_left" | "rotate_right" => {
                let raw = single_index(index_parts)?;
                let arg = single_arg(args)?;
                let width = self.bv_width(arg).ok_or_else(|| OxizError::ParseError {
                    position: 0,
                    message: format!("{name} requires a bit-vector argument"),
                })?;
                if width == 0 {
                    return Ok(Some(arg));
                }
                // Effective left-rotation amount in 0..width.
                let amount = if name == "rotate_left" {
                    raw % width
                } else {
                    (width - (raw % width)) % width
                };
                if amount == 0 {
                    return Ok(Some(arg));
                }
                // rol(x, a) = concat(x[width-1-a : 0], x[width-1 : width-a]).
                let low = self.manager.mk_bv_extract(width - 1 - amount, 0, arg);
                let high = self.manager.mk_bv_extract(width - 1, width - amount, arg);
                Ok(Some(self.manager.try_mk_bv_concat(low, high)?))
            }
            "repeat" => {
                let n = single_index(index_parts)?;
                let arg = single_arg(args)?;
                if n == 0 {
                    return Err(OxizError::ParseError {
                        position: 0,
                        message: "(_ repeat 0 ...) is not a valid bit-vector".to_string(),
                    });
                }
                let mut result = arg;
                for _ in 1..n {
                    result = self.manager.try_mk_bv_concat(result, arg)?;
                }
                Ok(Some(result))
            }
            "divisible" => {
                // ((_ divisible n) x) <=> (= (mod x n) 0).
                if index_parts.len() != 1 {
                    return Err(OxizError::ParseError {
                        position: 0,
                        message: format!(
                            "(_ divisible ...) requires exactly 1 index, got {}",
                            index_parts.len()
                        ),
                    });
                }
                let arg = single_arg(args)?;
                let n: BigInt = index_parts[0].parse().map_err(|_| OxizError::ParseError {
                    position: 0,
                    message: format!("invalid divisor for divisible: {}", index_parts[0]),
                })?;
                let divisor = self.manager.mk_int(n);
                let modulo = self.manager.mk_mod(arg, divisor);
                let zero = self.manager.mk_int(0);
                Ok(Some(self.manager.mk_eq(modulo, zero)))
            }
            // ((_ re.^ n) R): R repeated exactly n times.
            "re.^" => {
                let n = single_index(index_parts)?;
                let re = single_arg(args)?;
                Ok(Some(self.manager.mk_re_power(n, re)))
            }
            // ((_ re.loop lo hi) R): R repeated between lo and hi times.
            "re.loop" => {
                if index_parts.len() != 2 {
                    return Err(OxizError::ParseError {
                        position: 0,
                        message: format!(
                            "(_ re.loop ...) requires exactly 2 indices, got {}",
                            index_parts.len()
                        ),
                    });
                }
                let lo = index_parts[0]
                    .parse::<u32>()
                    .map_err(|_| OxizError::ParseError {
                        position: 0,
                        message: format!("invalid lower bound for re.loop: {}", index_parts[0]),
                    })?;
                let hi = index_parts[1]
                    .parse::<u32>()
                    .map_err(|_| OxizError::ParseError {
                        position: 0,
                        message: format!("invalid upper bound for re.loop: {}", index_parts[1]),
                    })?;
                let re = single_arg(args)?;
                Ok(Some(self.manager.mk_re_loop(lo, hi, re)))
            }
            _ => Ok(None),
        }
    }

    /// The `(eb, sb)` format indices of `(_ to_fp eb sb)` /
    /// `(_ to_fp_unsigned eb sb)`.
    fn fp_conv_format(&self, name: &str, index_parts: &[String]) -> Result<(u32, u32)> {
        let [eb_raw, sb_raw] = index_parts else {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!(
                    "(_ {name} eb sb) requires exactly 2 indices, got {}",
                    index_parts.len()
                ),
            });
        };
        let eb: u32 = eb_raw.parse().map_err(|_| OxizError::ParseError {
            position: self.lexer.position(),
            message: format!("invalid exponent-bits index for {name}: {eb_raw}"),
        })?;
        let sb: u32 = sb_raw.parse().map_err(|_| OxizError::ParseError {
            position: self.lexer.position(),
            message: format!("invalid significand-bits index for {name}: {sb_raw}"),
        })?;
        Ok((eb, sb))
    }

    /// The single width index of `(_ fp.to_sbv m)` / `(_ fp.to_ubv m)`.
    fn fp_conv_width(&self, name: &str, index_parts: &[String]) -> Result<u32> {
        let [raw] = index_parts else {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!(
                    "(_ {name} m) requires exactly 1 index, got {}",
                    index_parts.len()
                ),
            });
        };
        raw.parse::<u32>().map_err(|_| OxizError::ParseError {
            position: self.lexer.position(),
            message: format!("invalid width index for {name}: {raw}"),
        })
    }

    /// Read the head of one of the indexed floating-point conversion
    /// operators: `((_ to_fp eb sb) RM x)`, `((_ to_fp_unsigned eb sb) RM x)`,
    /// `((_ fp.to_sbv m) RM x)`, `((_ fp.to_ubv m) RM x)`.
    ///
    /// These all take a rounding-mode *symbol* (`RNE`/`RNA`/`RTP`/`RTN`/`RTZ`)
    /// as their first argument. A bare rounding-mode symbol is not itself a
    /// term — it is not a declared constant, so the strict unknown-symbol check
    /// in [`Parser::parse_symbol`] would reject it if it were parsed as an
    /// operand — so it is consumed here, while the head is being read, with
    /// [`Parser::parse_rounding_mode`].
    ///
    /// Returns `Ok(Some(rm))` when `name` is one of these operators and the
    /// rounding mode was read (the caller then parses exactly one operand and
    /// calls [`Parser::build_indexed_fp_conv`]), `Ok(None)` when `name` is not
    /// one of them, or `Err(..)` on a malformed use.
    ///
    /// Note: the alternate no-rounding-mode bit-pattern form
    /// `((_ to_fp eb sb) bv)` (a single bit-vector operand, reinterpreted as
    /// an IEEE-754 bit pattern rather than rounded) has no builder yet and is
    /// intentionally reported as `Ok(None)`; it falls through to the generic
    /// uninterpreted-application fallback exactly as before.
    pub(super) fn open_indexed_fp_conv(
        &mut self,
        name: &str,
        index_parts: &[String],
    ) -> Result<Option<RmOperand>> {
        match name {
            "to_fp" | "to_fp_unsigned" => {
                // Validate the format indices before deciding which form this
                // is, so a malformed `(_ to_fp x y)` is reported as such.
                let _ = self.fp_conv_format(name, index_parts)?;
                // Peek: is the next token a rounding-mode symbol? If not,
                // this is the (not-yet-supported) bit-pattern form; leave it
                // to the generic fallback rather than misparsing it.
                if !self.next_token_is_rounding_mode() {
                    return Ok(None);
                }
                Ok(Some(self.parse_rounding_mode_operand()?))
            }
            "fp.to_sbv" | "fp.to_ubv" => {
                let _ = self.fp_conv_width(name, index_parts)?;
                Ok(Some(self.parse_rounding_mode_operand()?))
            }
            _ => Ok(None),
        }
    }

    /// Build an indexed floating-point conversion whose rounding mode was read
    /// by [`Parser::open_indexed_fp_conv`] and whose single operand has now
    /// been parsed.
    pub(super) fn build_indexed_fp_conv(
        &mut self,
        name: &str,
        index_parts: &[String],
        rm: RmOperand,
        arg: TermId,
    ) -> Result<TermId> {
        match rm {
            RmOperand::Concrete(rm) => {
                self.build_indexed_fp_conv_concrete(name, index_parts, rm, arg)
            }
            RmOperand::Symbolic(mode) => {
                let name = name.to_string();
                let index_parts = index_parts.to_vec();
                self.expand_symbolic_rm(mode, |parser, rm| {
                    parser.build_indexed_fp_conv_concrete(&name, &index_parts, rm, arg)
                })
            }
        }
    }

    /// Build an indexed floating-point conversion at one *literal* rounding
    /// mode.
    fn build_indexed_fp_conv_concrete(
        &mut self,
        name: &str,
        index_parts: &[String],
        rm: RoundingMode,
        arg: TermId,
    ) -> Result<TermId> {
        match name {
            "to_fp" | "to_fp_unsigned" => {
                let (eb, sb) = self.fp_conv_format(name, index_parts)?;
                if name == "to_fp_unsigned" {
                    return Ok(self.manager.mk_ubv_to_fp(rm, arg, eb, sb));
                }
                // Plain `to_fp` dispatches on the operand's sort: Real,
                // another FloatingPoint format, or a signed bit-vector.
                let arg_kind = self
                    .manager
                    .get(arg)
                    .and_then(|t| self.manager.sorts.get(t.sort))
                    .map(|s| s.kind.clone());
                match arg_kind {
                    Some(SortKind::Real) => Ok(self.manager.mk_real_to_fp(rm, arg, eb, sb)),
                    Some(SortKind::FloatingPoint { .. }) => {
                        Ok(self.manager.mk_fp_to_fp(rm, arg, eb, sb))
                    }
                    Some(SortKind::BitVec(_)) => Ok(self.manager.mk_sbv_to_fp(rm, arg, eb, sb)),
                    _ => Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!(
                            "(_ to_fp {eb} {sb}): operand must have Real, FloatingPoint, or BitVec sort"
                        ),
                    }),
                }
            }
            "fp.to_sbv" | "fp.to_ubv" => {
                let width = self.fp_conv_width(name, index_parts)?;
                if name == "fp.to_sbv" {
                    Ok(self.manager.mk_fp_to_sbv(rm, arg, width))
                } else {
                    Ok(self.manager.mk_fp_to_ubv(rm, arg, width))
                }
            }
            _ => Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("internal: {name} is not an indexed FP conversion"),
            }),
        }
    }

    /// Peek at the next token and report whether it looks like a rounding
    /// mode symbol, without consuming it. Used by
    /// [`Parser::build_indexed_fp_conv`] to distinguish the RM-first
    /// `to_fp`/`to_fp_unsigned` form from the (unsupported) no-RM
    /// bit-pattern form.
    pub(super) fn next_token_is_rounding_mode(&self) -> bool {
        let Some(TokenKind::Symbol(s)) = self.lexer.peek().map(|t| t.kind) else {
            return false;
        };
        if is_rounding_mode_literal(&s) {
            return true;
        }
        // A *symbolic* mode occupies the same position as a literal one, and
        // so does the bit-pattern form's single bit-vector operand — both are
        // bare symbols, so the spelling alone cannot separate them. Resolve
        // the sort instead: only a `RoundingMode`-sorted symbol is a mode.
        let rm_sort = self.manager.sorts.rounding_mode_sort;
        if let Some(&term) = self.bindings.get(&s) {
            return self.manager.get(term).is_some_and(|t| t.sort == rm_sort);
        }
        self.constants.get(&s) == Some(&rm_sort)
    }
}
