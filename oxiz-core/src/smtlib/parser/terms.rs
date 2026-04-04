//! Term parsing for the SMT-LIB2 parser

use super::super::lexer::TokenKind;
use super::{Attribute, Parser, parse_decimal_to_rational};
use crate::ast::TermId;
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use smallvec::SmallVec;

impl<'a> Parser<'a> {
    /// Parse a term
    pub fn parse_term(&mut self) -> Result<TermId> {
        let token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "unexpected end of input".to_string(),
            })?;

        match token.kind {
            TokenKind::LParen => self.parse_compound_term(),
            TokenKind::Symbol(s) => self.parse_symbol(&s),
            TokenKind::Numeral(n) => {
                let value: BigInt = n.parse().map_err(|_| OxizError::ParseError {
                    position: token.start,
                    message: format!("invalid numeral: {n}"),
                })?;
                Ok(self.manager.mk_int(value))
            }
            TokenKind::Hexadecimal(h) => {
                let value =
                    BigInt::parse_bytes(h.as_bytes(), 16).ok_or_else(|| OxizError::ParseError {
                        position: token.start,
                        message: format!("invalid hexadecimal: {h}"),
                    })?;
                let width = (h.len() * 4) as u32;
                Ok(self.manager.mk_bitvec(value, width))
            }
            TokenKind::Binary(b) => {
                let value =
                    BigInt::parse_bytes(b.as_bytes(), 2).ok_or_else(|| OxizError::ParseError {
                        position: token.start,
                        message: format!("invalid binary: {b}"),
                    })?;
                let width = b.len() as u32;
                Ok(self.manager.mk_bitvec(value, width))
            }
            TokenKind::Decimal(d) => {
                // Parse decimal literal as Rational64
                let rational =
                    parse_decimal_to_rational(&d).map_err(|e| OxizError::ParseError {
                        position: token.start,
                        message: format!("invalid decimal: {d} - {e}"),
                    })?;
                Ok(self.manager.mk_real(rational))
            }
            TokenKind::StringLit(s) => {
                // Parse string literal
                Ok(self.manager.mk_string_lit(&s))
            }
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("unexpected token: {:?}", token.kind),
            }),
        }
    }

    pub(super) fn parse_symbol(&mut self, s: &str) -> Result<TermId> {
        match s {
            "true" => Ok(self.manager.mk_true()),
            "false" => Ok(self.manager.mk_false()),
            _ => {
                // Check bindings first
                if let Some(&term) = self.bindings.get(s) {
                    return Ok(term);
                }
                // Check if this is a datatype constructor (e.g., Monday, nil, cons, etc.)
                if let Some(&dt_sort) = self.dt_constructors.get(s) {
                    return Ok(self.manager.mk_dt_constructor(s, vec![], dt_sort));
                }
                // Check constants
                if let Some(&sort) = self.constants.get(s) {
                    return Ok(self.manager.mk_var(s, sort));
                }
                // Default to boolean variable
                let sort = self.manager.sorts.bool_sort;
                Ok(self.manager.mk_var(s, sort))
            }
        }
    }

    pub(super) fn parse_compound_term(&mut self) -> Result<TermId> {
        let op_token = self
            .lexer
            .next_token()
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "unexpected end of input".to_string(),
            })?;

        // Handle indexed identifiers that start with `(`: ((_ to_fp 8 24) RNE 1.5)
        // or qualified identifiers: ((as const (Array Int Int)) 0)
        if matches!(op_token.kind, TokenKind::LParen) {
            // Peek at the next symbol to determine what kind of compound operator this is
            let qualifier = self.expect_symbol()?;
            if qualifier == "as" {
                // SMT-LIB qualified identifier: (as <symbol> <sort>)
                // Consumes: symbol, sort, closing ')'
                let symbol = self.expect_symbol()?;
                let sort = self.parse_sort()?;
                self.expect_rparen()?; // Close the (as ...) form
                // Now parse arguments for the qualified function application
                let args = self.parse_term_list()?;
                // Build a qualified apply node with the annotated sort
                // For known forms like (as const (Array D R)), we represent this
                // as an Apply node with function name "(as const)" and the proper sort.
                let func_name = format!("(as {symbol})");
                return Ok(self.manager.mk_apply(&func_name, args, sort));
            }
            if qualifier != "_" {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!(
                        "expected '_' or 'as' in compound operator, found '{qualifier}'"
                    ),
                });
            }

            let name = self.expect_symbol()?;

            // Parse indices (can be numerals or symbols, depending on the operation)
            let mut index_parts = Vec::new();
            loop {
                if let Some(token) = self.lexer.peek() {
                    match &token.kind {
                        TokenKind::RParen => {
                            self.lexer.next_token(); // consume rparen
                            break;
                        }
                        TokenKind::Numeral(n) => {
                            let n = n.clone();
                            self.lexer.next_token();
                            index_parts.push(n);
                        }
                        TokenKind::Symbol(s) => {
                            // For datatype testers like (_ is nil), the constructor name is a symbol
                            let s = s.clone();
                            self.lexer.next_token();
                            index_parts.push(s);
                        }
                        _ => {
                            return Err(OxizError::ParseError {
                                position: token.start,
                                message: format!(
                                    "expected numeral, symbol, or ')' in indexed identifier, found {:?}",
                                    token.kind
                                ),
                            });
                        }
                    }
                } else {
                    break;
                }
            }

            // Handle special indexed operators
            if name == "is" {
                // Handle datatype tester: ((_ is constructor) arg)
                if index_parts.len() != 1 {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!(
                            "(_ is) requires exactly 1 constructor name, got {}",
                            index_parts.len()
                        ),
                    });
                }
                let constructor_name = &index_parts[0];
                let arg = self.parse_term()?;
                self.expect_rparen()?; // Close the outer application
                return Ok(self.manager.mk_dt_tester(constructor_name, arg));
            }

            // Now parse the arguments and closing paren
            // Build the indexed identifier name
            let indices_str = index_parts.join(" ");
            let func_name = if index_parts.is_empty() {
                format!("(_ {name})")
            } else {
                format!("(_ {name} {indices_str})")
            };

            // Parse arguments
            let args = self.parse_term_list()?;
            let sort = self.manager.sorts.bool_sort; // Default
            return Ok(self.manager.mk_apply(&func_name, args, sort));
        }

        // Handle indexed identifiers: (_ extract 7 4), (_ sign_extend 16), etc.
        if matches!(op_token.kind, TokenKind::Symbol(ref s) if s == "_") {
            let name = self.expect_symbol()?;

            // Parse indices (can be numerals or symbols)
            let mut indices = Vec::new();
            let mut index_parts = Vec::new();
            loop {
                if let Some(token) = self.lexer.peek() {
                    match &token.kind {
                        TokenKind::RParen => {
                            break;
                        }
                        TokenKind::Numeral(n) => {
                            let n = n.clone();
                            self.lexer.next_token();
                            let idx = n.parse::<u32>().map_err(|_| OxizError::ParseError {
                                position: token.start,
                                message: format!("invalid index: {n}"),
                            })?;
                            indices.push(idx);
                            index_parts.push(n);
                        }
                        TokenKind::Symbol(s) => {
                            // For datatype testers and similar constructs
                            let s = s.clone();
                            self.lexer.next_token();
                            index_parts.push(s);
                        }
                        _ => break,
                    }
                } else {
                    break;
                }
            }

            // Handle indexed operations
            // Check for bitvector literal: (_ bvN M) where N is the value and M is the width
            if let Some(bv_val_str) = name.strip_prefix("bv") {
                // (_ bvN M) is a bitvector literal with value N and width M
                if indices.len() != 1 {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!(
                            "bitvector literal (_ {name} ...) requires exactly 1 index (width), got {}",
                            indices.len()
                        ),
                    });
                }
                let value: i64 = bv_val_str.parse().map_err(|_| OxizError::ParseError {
                    position: self.lexer.position(),
                    message: format!("invalid bitvector literal value: {bv_val_str}"),
                })?;
                let width = indices[0];
                if width == 0 || width > 65536 {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: format!("invalid bitvector width: {width} (must be 1-65536)"),
                    });
                }
                // Consume the closing rparen of (_ bvN M)
                self.expect_rparen()?;
                return Ok(self.manager.mk_bitvec(value, width));
            }

            match name.as_str() {
                "extract" => {
                    if indices.len() != 2 {
                        return Err(OxizError::ParseError {
                            position: self.lexer.position(),
                            message: format!(
                                "extract requires exactly 2 indices, got {}",
                                indices.len()
                            ),
                        });
                    }
                    let arg = self.parse_term()?;
                    self.expect_rparen()?;
                    return Ok(self.manager.mk_bv_extract(indices[0], indices[1], arg));
                }
                "is" => {
                    // Handle datatype tester: (_ is constructor) arg
                    if index_parts.len() != 1 {
                        return Err(OxizError::ParseError {
                            position: self.lexer.position(),
                            message: format!(
                                "(_ is) requires exactly 1 constructor name, got {}",
                                index_parts.len()
                            ),
                        });
                    }
                    self.expect_rparen()?; // Close the (_ is X) part
                    let constructor_name = &index_parts[0];
                    let arg = self.parse_term()?;
                    self.expect_rparen()?; // Close the outer application
                    return Ok(self.manager.mk_dt_tester(constructor_name, arg));
                }
                _ => {
                    // For unrecognized indexed identifiers (like to_fp, is, etc.),
                    // treat them as function applications and parse arguments
                    // Expect closing paren for the indexed identifier
                    self.expect_rparen()?;

                    // Build the indexed identifier name
                    let indices_str = index_parts.join(" ");
                    let func_name = if index_parts.is_empty() {
                        format!("(_ {name})")
                    } else {
                        format!("(_ {name} {indices_str})")
                    };

                    // Parse arguments
                    let args = self.parse_term_list()?;
                    let sort = self.manager.sorts.bool_sort; // Default
                    return Ok(self.manager.mk_apply(&func_name, args, sort));
                }
            }
        }

        let op = match &op_token.kind {
            TokenKind::Symbol(s) => s.clone(),
            TokenKind::Keyword(k) => format!(":{k}"),
            _ => {
                return Err(OxizError::ParseError {
                    position: op_token.start,
                    message: format!("expected operator, found {:?}", op_token.kind),
                });
            }
        };

        let result = match op.as_str() {
            "!" => {
                // Annotation: (! term :attr1 val1 :attr2 val2 ...)
                let term = self.parse_term()?;
                let attrs = self.parse_attributes()?;
                self.expect_rparen()?;

                // Store annotations for this term
                if !attrs.is_empty() {
                    self.annotations.insert(term, attrs);
                }

                term
            }
            "not" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_not(arg)
            }
            "and" => {
                let args = self.parse_term_list()?;
                self.manager.mk_and(args)
            }
            "or" => {
                let args = self.parse_term_list()?;
                self.manager.mk_or(args)
            }
            "=>" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_implies(lhs, rhs)
            }
            "xor" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                // XOR = (a and not b) or (not a and b)
                let not_lhs = self.manager.mk_not(lhs);
                let not_rhs = self.manager.mk_not(rhs);
                let and1 = self.manager.mk_and([lhs, not_rhs]);
                let and2 = self.manager.mk_and([not_lhs, rhs]);
                self.manager.mk_or([and1, and2])
            }
            "ite" => {
                let cond = self.parse_term()?;
                let then_branch = self.parse_term()?;
                let else_branch = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_ite(cond, then_branch, else_branch)
            }
            "=" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_eq(lhs, rhs)
            }
            "distinct" => {
                let args = self.parse_term_list()?;
                self.manager.mk_distinct(args)
            }
            "+" => {
                let args = self.parse_term_list()?;
                self.manager.mk_add(args)
            }
            "-" => {
                let first = self.parse_term()?;
                if let Some(token) = self.lexer.peek()
                    && matches!(token.kind, TokenKind::RParen)
                {
                    self.lexer.next_token();
                    // Unary minus - use mk_neg for proper negation
                    return Ok(self.manager.mk_neg(first));
                }
                let second = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_sub(first, second)
            }
            "*" => {
                let args = self.parse_term_list()?;
                self.manager.mk_mul(args)
            }
            "div" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                // For now, treat div as subtraction placeholder
                self.manager.mk_sub(lhs, rhs)
            }
            "mod" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                // For now, treat mod as subtraction placeholder
                self.manager.mk_sub(lhs, rhs)
            }
            "<" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_lt(lhs, rhs)
            }
            "<=" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_le(lhs, rhs)
            }
            ">" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_gt(lhs, rhs)
            }
            ">=" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_ge(lhs, rhs)
            }
            "select" => {
                let array = self.parse_term()?;
                let index = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_select(array, index)
            }
            "store" => {
                let array = self.parse_term()?;
                let index = self.parse_term()?;
                let value = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_store(array, index, value)
            }
            "let" => self.parse_let()?,
            "forall" => self.parse_forall()?,
            "exists" => self.parse_exists()?,
            // BitVector operations
            "bvnot" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_not(arg)
            }
            "bvand" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_and(lhs, rhs)
            }
            "bvor" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_or(lhs, rhs)
            }
            "bvadd" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_add(lhs, rhs)
            }
            "bvsub" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_sub(lhs, rhs)
            }
            "bvmul" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_mul(lhs, rhs)
            }
            "bvult" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_ult(lhs, rhs)
            }
            "bvslt" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_slt(lhs, rhs)
            }
            "bvule" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_ule(lhs, rhs)
            }
            "bvsle" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_sle(lhs, rhs)
            }
            "bvugt" => {
                // bvugt(a, b) = bvult(b, a)
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_ult(rhs, lhs)
            }
            "bvsgt" => {
                // bvsgt(a, b) = bvslt(b, a)
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_slt(rhs, lhs)
            }
            "bvuge" => {
                // bvuge(a, b) = NOT bvult(a, b)
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                let ult = self.manager.mk_bv_ult(lhs, rhs);
                self.manager.mk_not(ult)
            }
            "bvsge" => {
                // bvsge(a, b) = NOT bvslt(a, b)
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                let slt = self.manager.mk_bv_slt(lhs, rhs);
                self.manager.mk_not(slt)
            }
            "bvxor" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_xor(lhs, rhs)
            }
            "bvudiv" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_udiv(lhs, rhs)
            }
            "bvsdiv" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_sdiv(lhs, rhs)
            }
            "bvurem" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_urem(lhs, rhs)
            }
            "bvsrem" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_srem(lhs, rhs)
            }
            "bvshl" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_shl(lhs, rhs)
            }
            "bvlshr" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_lshr(lhs, rhs)
            }
            "bvashr" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_ashr(lhs, rhs)
            }
            "concat" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_bv_concat(lhs, rhs)
            }
            // Floating-point arithmetic operations (take rounding mode as first argument)
            "fp.add" => {
                let rm = self.parse_rounding_mode()?;
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_add(rm, lhs, rhs)
            }
            "fp.sub" => {
                let rm = self.parse_rounding_mode()?;
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_sub(rm, lhs, rhs)
            }
            "fp.mul" => {
                let rm = self.parse_rounding_mode()?;
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_mul(rm, lhs, rhs)
            }
            "fp.div" => {
                let rm = self.parse_rounding_mode()?;
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_div(rm, lhs, rhs)
            }
            "fp.rem" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_rem(lhs, rhs)
            }
            "fp.sqrt" => {
                let rm = self.parse_rounding_mode()?;
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_sqrt(rm, arg)
            }
            "fp.fma" => {
                let rm = self.parse_rounding_mode()?;
                let x = self.parse_term()?;
                let y = self.parse_term()?;
                let z = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_fma(rm, x, y, z)
            }
            "fp.roundToIntegral" => {
                let rm = self.parse_rounding_mode()?;
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_round_to_integral(rm, arg)
            }
            // Floating-point comparisons
            "fp.eq" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_eq(lhs, rhs)
            }
            "fp.lt" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_lt(lhs, rhs)
            }
            "fp.gt" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_gt(lhs, rhs)
            }
            "fp.leq" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_leq(lhs, rhs)
            }
            "fp.geq" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_geq(lhs, rhs)
            }
            // Floating-point predicates
            "fp.isNormal" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_normal(arg)
            }
            "fp.isSubnormal" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_subnormal(arg)
            }
            "fp.isZero" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_zero(arg)
            }
            "fp.isInfinite" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_infinite(arg)
            }
            "fp.isNaN" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_nan(arg)
            }
            "fp.isNegative" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_negative(arg)
            }
            "fp.isPositive" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_is_positive(arg)
            }
            // Floating-point unary operations
            "fp.abs" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_abs(arg)
            }
            "fp.neg" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_neg(arg)
            }
            // Floating-point binary min/max
            "fp.min" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_min(lhs, rhs)
            }
            "fp.max" => {
                let lhs = self.parse_term()?;
                let rhs = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_fp_max(lhs, rhs)
            }
            // String operations
            "str.++" => {
                let mut result = self.parse_term()?;
                loop {
                    if let Some(token) = self.lexer.peek()
                        && matches!(token.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }
                    let next = self.parse_term()?;
                    result = self.manager.mk_str_concat(result, next);
                }
                result
            }
            "str.len" => {
                let arg = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_len(arg)
            }
            "str.substr" => {
                let s = self.parse_term()?;
                let start = self.parse_term()?;
                let len = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_substr(s, start, len)
            }
            "str.at" => {
                let s = self.parse_term()?;
                let i = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_at(s, i)
            }
            "str.contains" => {
                let s = self.parse_term()?;
                let sub = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_contains(s, sub)
            }
            "str.prefixof" => {
                let prefix = self.parse_term()?;
                let s = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_prefixof(prefix, s)
            }
            "str.suffixof" => {
                let suffix = self.parse_term()?;
                let s = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_suffixof(suffix, s)
            }
            "str.indexof" => {
                let s = self.parse_term()?;
                let sub = self.parse_term()?;
                let offset = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_indexof(s, sub, offset)
            }
            "str.replace" => {
                let s = self.parse_term()?;
                let pattern = self.parse_term()?;
                let replacement = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_replace(s, pattern, replacement)
            }
            "str.replace_all" => {
                let s = self.parse_term()?;
                let pattern = self.parse_term()?;
                let replacement = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_replace_all(s, pattern, replacement)
            }
            "str.to_int" | "str.to.int" => {
                let s = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_to_int(s)
            }
            "int.to_str" | "int.to.str" | "str.from_int" => {
                let n = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_int_to_str(n)
            }
            "str.in_re" | "str.in.re" => {
                let s = self.parse_term()?;
                let re = self.parse_term()?;
                self.expect_rparen()?;
                self.manager.mk_str_in_re(s, re)
            }
            _ => {
                // Check for defined function
                if let Some((params, body)) = self.function_defs.get(&op).cloned() {
                    // Parse arguments
                    let args = self.parse_term_list()?;

                    if args.len() != params.len() {
                        return Err(OxizError::ParseError {
                            position: 0,
                            message: format!(
                                "wrong number of arguments for {}: expected {}, got {}",
                                op,
                                params.len(),
                                args.len()
                            ),
                        });
                    }

                    // Substitute arguments into the body
                    let mut substitution = FxHashMap::default();
                    for ((param_name, _param_sort), &arg) in params.iter().zip(args.iter()) {
                        // Find the parameter variable in the body
                        let param_sort = self
                            .constants
                            .get(param_name)
                            .copied()
                            .unwrap_or(self.manager.sorts.bool_sort);
                        let param_var = self.manager.mk_var(param_name, param_sort);
                        substitution.insert(param_var, arg);
                    }

                    // Apply substitution to get the result
                    self.manager.substitute(body, &substitution)
                } else {
                    // Regular function application.
                    //
                    // Look up the declared return sort from the functions table so
                    // that the Apply node carries the correct sort (e.g. `Int` for
                    // `(declare-fun f (Int) Int)` applications).  This is essential
                    // for theory reasoning: without the correct sort, an expression
                    // like `(> (f k) 10)` would be created with `f(k)` having
                    // `Bool` sort, causing the arithmetic theory to ignore it.
                    let args = self.parse_term_list()?;
                    let sort = self
                        .functions
                        .get(&op)
                        .map(|(_, ret)| *ret)
                        .unwrap_or(self.manager.sorts.bool_sort);
                    self.manager.mk_apply(&op, args, sort)
                }
            }
        };

        Ok(result)
    }

    pub(super) fn parse_term_list(&mut self) -> Result<SmallVec<[TermId; 4]>> {
        let mut args = SmallVec::new();
        loop {
            if let Some(token) = self.lexer.peek()
                && matches!(token.kind, TokenKind::RParen)
            {
                self.lexer.next_token();
                break;
            }
            args.push(self.parse_term()?);
        }
        Ok(args)
    }

    /// Parse attributes in an annotation
    pub(super) fn parse_attributes(&mut self) -> Result<Vec<Attribute>> {
        let mut attrs = Vec::new();

        loop {
            // Check if we've reached the closing paren
            if let Some(token) = self.lexer.peek() {
                if matches!(token.kind, TokenKind::RParen) {
                    break;
                }

                // Attributes start with a keyword (e.g., :named, :pattern)
                if let TokenKind::Keyword(key) = &token.kind {
                    let key = key.clone();
                    self.lexer.next_token(); // consume the keyword

                    // Try to parse the attribute value
                    let value = if let Some(next_token) = self.lexer.peek() {
                        match &next_token.kind {
                            // If next is a keyword or rparen, this attribute has no value
                            TokenKind::Keyword(_) | TokenKind::RParen => None,
                            // Otherwise, parse the value
                            _ => Some(self.parse_attribute_value()?),
                        }
                    } else {
                        None
                    };

                    attrs.push(Attribute { key, value });
                } else {
                    return Err(OxizError::ParseError {
                        position: token.start,
                        message: format!("expected keyword in annotation, found {:?}", token.kind),
                    });
                }
            } else {
                return Err(OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "unexpected end of input in annotation".to_string(),
                });
            }
        }

        Ok(attrs)
    }

    /// Parse an attribute value
    pub(super) fn parse_attribute_value(&mut self) -> Result<super::AttributeValue> {
        let token = self.lexer.peek().ok_or_else(|| OxizError::ParseError {
            position: self.lexer.position(),
            message: "unexpected end of input in attribute value".to_string(),
        })?;

        match &token.kind {
            TokenKind::Symbol(s) => {
                let s = s.clone();
                self.lexer.next_token();
                Ok(super::AttributeValue::Symbol(s))
            }
            TokenKind::Numeral(n) => {
                let n = n.clone();
                self.lexer.next_token();
                Ok(super::AttributeValue::Numeral(n))
            }
            TokenKind::StringLit(s) => {
                let s = s.clone();
                self.lexer.next_token();
                Ok(super::AttributeValue::String(s))
            }
            TokenKind::LParen => {
                // Could be an S-expression or a term
                // For :pattern, this would be a term list
                self.lexer.next_token(); // consume lparen
                let mut values = Vec::new();

                loop {
                    if let Some(t) = self.lexer.peek()
                        && matches!(t.kind, TokenKind::RParen)
                    {
                        self.lexer.next_token();
                        break;
                    }

                    // Try to parse as term first
                    let term = self.parse_term()?;
                    values.push(super::AttributeValue::Term(term));
                }

                Ok(super::AttributeValue::SExpr(values))
            }
            _ => Err(OxizError::ParseError {
                position: token.start,
                message: format!("unexpected token in attribute value: {:?}", token.kind),
            }),
        }
    }
}
