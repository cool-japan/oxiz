//! Operand plans and term construction for the compound-term parser.
//!
//! The term parser in `terms.rs` runs an explicit frame stack rather than
//! recursive descent: it reads a compound term's head, asks [`operand_plan`]
//! how many operand terms follow, collects those operands *iteratively*, and
//! only then hands the finished operand list to one of the `build_*` methods
//! below. Splitting "how many operands does this head take" from "what term do
//! these operands build" is what lets the driver run in constant native stack
//! space no matter how deeply the input nests.
//!
//! Every built-in operator therefore appears exactly twice: once in
//! [`operand_plan`], which fixes its arity, and once in the `build_*` method
//! matching that arity. `tests::planned_operators_all_reach_a_builder` guards
//! the two against drifting apart.

use super::Parser;
use super::commands::RmOperand;
use super::terms::Plan;
use crate::ast::{RoundingMode, TermId, TermKind};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortKind;
use num_bigint::BigInt;
use num_rational::Rational64;

/// The operand plan of a built-in SMT-LIB operator, or `None` when `op` is not
/// a built-in (so the parser resolves it against the declaration tables
/// instead).
///
/// The operators whose head carries extra syntax — `!`, `let`, `forall`,
/// `exists`, and the floating-point operators that take a leading rounding
/// mode — are *not* listed here: their heads need work at open time, so
/// `Parser::open_compound` handles them before consulting this table.
pub(super) fn operand_plan(op: &str) -> Option<Plan> {
    let plan =
        match op {
            // ---- one operand ----
            "not" | "abs" | "to_real" | "to_int" | "is_int" | "bvnot" | "bvneg" | "fp.isNormal"
            | "fp.isSubnormal" | "fp.isZero" | "fp.isInfinite" | "fp.isNaN" | "fp.isNegative"
            | "fp.isPositive" | "fp.abs" | "fp.neg" | "fp.to_real" | "str.len" | "str.is_digit"
            | "str.to_code" | "str.from_code" | "str.to_int" | "str.to.int" | "int.to_str"
            | "int.to.str" | "str.from_int" | "str.to_re" | "str.to.re" | "re.*" | "re.+"
            | "re.opt" | "re.comp" => Plan::Fixed(1),

            // ---- two operands ----
            "mod" | "select" | "bvand" | "bvor" | "bvadd" | "bvsub" | "bvmul" | "bvult"
            | "bvslt" | "bvule" | "bvsle" | "bvugt" | "bvsgt" | "bvuge" | "bvsge" | "bvxor"
            | "bvnand" | "bvnor" | "bvxnor" | "bvcomp" | "bvsmod" | "bvudiv" | "bvsdiv"
            | "bvurem" | "bvsrem" | "bvshl" | "bvlshr" | "bvashr" | "concat" | "fp.rem"
            | "fp.eq" | "fp.lt" | "fp.gt" | "fp.leq" | "fp.geq" | "fp.min" | "fp.max"
            | "str.at" | "str.contains" | "str.prefixof" | "str.suffixof" | "str.in_re"
            | "str.in.re" | "re.diff" | "re.range" => Plan::Fixed(2),

            // ---- three operands ----
            "ite" | "store" | "fp" | "str.substr" | "str.indexof" | "str.replace"
            | "str.replace_all" | "str.replace_re" | "str.replace_re_all" => Plan::Fixed(3),

            // ---- operands until the closing paren ----
            "and" | "or" | "=>" | "xor" | "=" | "distinct" | "+" | "-" | "*" | "div" | "/"
            | "<" | "<=" | ">" | ">=" | "str.++" | "str.<" | "str.<=" | "re.++" | "re.union"
            | "re.inter" => Plan::Variadic,

            _ => return None,
        };
    Some(plan)
}

impl Parser<'_> {
    /// Returns `true` if the given term has Real sort.
    fn is_real_term(&self, term: TermId) -> bool {
        self.manager
            .get(term)
            .and_then(|t| self.manager.sorts.get(t.sort))
            .is_some_and(|s| matches!(s.kind, SortKind::Real))
    }

    /// Extract `(value, width)` from `term` if it is a bit-vector literal.
    fn bv_const_parts(&self, term: TermId) -> Option<(BigInt, u32)> {
        match self.manager.get(term).map(|t| &t.kind) {
            Some(TermKind::BitVecConst { value, width }) => Some((value.clone(), *width)),
            _ => None,
        }
    }

    /// Construct an honest arity error for a core operator that requires at
    /// least `min` operands but received `got`.
    pub(super) fn min_arity_err(&self, op: &str, min: usize, got: usize) -> OxizError {
        OxizError::ParseError {
            position: self.lexer.position(),
            message: format!("{op} requires at least {min} arguments, got {got}"),
        }
    }

    /// Reported when [`operand_plan`] and the `build_*` methods disagree about
    /// an operator — an internal inconsistency rather than a user error, but
    /// surfaced as a parse error instead of panicking.
    fn plan_mismatch(&self, op: &str) -> OxizError {
        OxizError::ParseError {
            position: self.lexer.position(),
            message: format!("internal: no builder for planned operator {op}"),
        }
    }

    /// The `(_ BitVec w)` width of `term`'s sort, or `None` when the sort is
    /// not a bit-vector sort.
    fn bv_sort_width(&self, term: TermId) -> Option<u32> {
        let sort = self.manager.get(term)?.sort;
        match self.manager.sorts.get(sort)?.kind {
            SortKind::BitVec(w) => Some(w),
            _ => None,
        }
    }

    /// Enforce the SMT-LIB sort rule shared by every binary bit-vector
    /// operator except `concat`: both operands must have one and the same
    /// `(_ BitVec w)` sort. `TermManager`'s `mk_bv_*` constructors have no
    /// error channel and take the left operand's width on a mismatch, so the
    /// parser is the layer that must reject it — exactly where Z3 does.
    fn check_bv_binary_widths(&self, op: &str, x: TermId, y: TermId) -> Result<()> {
        let (xw, yw) = (self.bv_sort_width(x), self.bv_sort_width(y));
        match (xw, yw) {
            (Some(xw), Some(yw)) if xw == yw => Ok(()),
            (Some(xw), Some(yw)) => Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!(
                    "operands of {op} must have the same bit-vector width, \
                     got (_ BitVec {xw}) and (_ BitVec {yw})"
                ),
            }),
            _ => Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("operands of {op} must have bit-vector sorts"),
            }),
        }
    }

    /// Build the conjunction of the boolean atoms produced by a chainable
    /// operator (`=`, `<`, `<=`, `>`, `>=`). SMT-LIB defines these operators as
    /// *chainable*: `(op a b c)` means `(and (op a b) (op b c))`. When there is
    /// a single atom (the binary case) it is returned directly so that ordinary
    /// binary uses keep their exact term kind (e.g. `Lt`, `Eq`) rather than
    /// being wrapped in a one-element `and`.
    fn chain_conjunction(&mut self, atoms: Vec<TermId>) -> TermId {
        if atoms.len() == 1 {
            atoms[0]
        } else {
            self.manager.mk_and(atoms)
        }
    }

    /// Two-operand XOR lowered to `and`/`or`/`not`, used to fold the
    /// left-associative n-ary `xor`.
    fn mk_xor2(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let not_lhs = self.manager.mk_not(lhs);
        let not_rhs = self.manager.mk_not(rhs);
        let and1 = self.manager.mk_and([lhs, not_rhs]);
        let and2 = self.manager.mk_and([not_lhs, rhs]);
        self.manager.mk_or([and1, and2])
    }

    /// Build the `(fp sign exponent significand)` bit-triple literal
    /// constructor. Per the SMT-LIB `FloatingPoint` theory, all three
    /// operands must be bit-vector *literals*: a 1-bit sign, an `eb`-bit
    /// biased exponent, and an `(sb-1)`-bit significand (the implicit
    /// leading bit is not stored). Symbolic (non-literal) operands are
    /// rejected with an honest parse error rather than silently degrading
    /// to an uninterpreted application.
    fn build_fp_lit(&mut self, sign: TermId, exp: TermId, sig: TermId) -> Result<TermId> {
        let (sign_val, sign_width) =
            self.bv_const_parts(sign)
                .ok_or_else(|| OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "fp: sign operand must be a bit-vector literal".to_string(),
                })?;
        let (exp_val, eb) = self
            .bv_const_parts(exp)
            .ok_or_else(|| OxizError::ParseError {
                position: self.lexer.position(),
                message: "fp: exponent operand must be a bit-vector literal".to_string(),
            })?;
        let (sig_val, sig_width) =
            self.bv_const_parts(sig)
                .ok_or_else(|| OxizError::ParseError {
                    position: self.lexer.position(),
                    message: "fp: significand operand must be a bit-vector literal".to_string(),
                })?;
        if sign_width != 1 {
            return Err(OxizError::ParseError {
                position: self.lexer.position(),
                message: format!("fp: sign operand must be exactly 1 bit wide, got {sign_width}"),
            });
        }
        let sb = sig_width + 1;
        let sign_bool = sign_val != BigInt::from(0);
        Ok(self.manager.mk_fp_lit(sign_bool, exp_val, sig_val, eb, sb))
    }

    /// Build a `Plan::Fixed(1)` built-in from its single operand.
    pub(super) fn build_unary(&mut self, op: &str, x: TermId) -> Result<TermId> {
        let term = match op {
            "not" => self.manager.mk_not(x),
            // (abs x) = (ite (>= x 0) x (- x)), with the zero literal typed to
            // match the operand sort so mixed Int/Real reasoning stays
            // consistent.
            "abs" => {
                let zero = if self.is_real_term(x) {
                    self.manager.mk_real(Rational64::from_integer(0))
                } else {
                    self.manager.mk_int(0)
                };
                let cond = self.manager.mk_ge(x, zero);
                let neg = self.manager.mk_neg(x);
                self.manager.mk_ite(cond, x, neg)
            }
            // Int -> Real injection. The arithmetic engine represents both
            // sorts as rationals and the value is preserved exactly, so the
            // injection is the identity on the term; the operand keeps its
            // (integer) constraints, which is exactly `to_real` semantics.
            "to_real" => x,
            // (to_int r) is the floor of a real. For a *constant* operand we
            // compute the exact Euclidean floor (`numer.div_euclid(denom)`,
            // denom always positive in a normalized rational); an
            // already-integer constant maps to itself. For a *symbolic* real
            // there is no floor operator in this term representation, so
            // rather than emit a silently wrong term we surface an honest
            // parse error.
            "to_int" => match self.manager.get(x).map(|t| t.kind.clone()) {
                Some(TermKind::IntConst(_)) => x,
                Some(TermKind::RealConst(r)) => {
                    let floor = r.numer().div_euclid(*r.denom());
                    self.manager.mk_int(floor)
                }
                _ => {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: "unsupported to_int on a symbolic real (no floor operator)"
                            .to_string(),
                    });
                }
            },
            // (is_int r) tests whether a real is integer-valued. Decidable
            // exactly for a constant operand; for a symbolic real we have no
            // integrality predicate to lower to, so we surface an honest parse
            // error instead of a wrong term.
            "is_int" => match self.manager.get(x).map(|t| t.kind.clone()) {
                Some(TermKind::IntConst(_)) => self.manager.mk_true(),
                Some(TermKind::RealConst(r)) => {
                    if r.is_integer() {
                        self.manager.mk_true()
                    } else {
                        self.manager.mk_false()
                    }
                }
                _ => {
                    return Err(OxizError::ParseError {
                        position: self.lexer.position(),
                        message: "unsupported is_int on a symbolic real (no integrality predicate)"
                            .to_string(),
                    });
                }
            },
            "bvnot" => self.manager.mk_bv_not(x),
            "bvneg" => self.manager.mk_bv_neg(x),
            "fp.isNormal" => self.manager.mk_fp_is_normal(x),
            "fp.isSubnormal" => self.manager.mk_fp_is_subnormal(x),
            "fp.isZero" => self.manager.mk_fp_is_zero(x),
            "fp.isInfinite" => self.manager.mk_fp_is_infinite(x),
            "fp.isNaN" => self.manager.mk_fp_is_nan(x),
            "fp.isNegative" => self.manager.mk_fp_is_negative(x),
            "fp.isPositive" => self.manager.mk_fp_is_positive(x),
            "fp.abs" => self.manager.mk_fp_abs(x),
            "fp.neg" => self.manager.mk_fp_neg(x),
            "fp.to_real" => self.manager.mk_fp_to_real(x),
            "str.len" => self.manager.mk_str_len(x),
            // `(str.is_digit s)` holds iff `s` is a single-character string
            // whose character is a decimal digit. That is exactly the language
            // of `(re.range "0" "9")`, so it lowers to a membership constraint
            // with no new term kind — and stays exact for symbolic operands.
            "str.is_digit" => {
                let lo = self.manager.mk_string_lit("0");
                let hi = self.manager.mk_string_lit("9");
                let digits = self.manager.mk_re_range(lo, hi);
                self.manager.mk_str_in_re(x, digits)
            }
            // `(str.to_code s)` is the code point of `s` when `s` is a
            // one-character string, and `-1` otherwise. `(str.from_code n)` is
            // the one-character string with code point `n` when `n` is in the
            // alphabet `[0, 0x2FFFF]`, and `""` otherwise. Constant operands
            // fold in the builders (`ast::str_fold`); symbolic ones become
            // `StrToCode` / `StrFromCode` terms.
            "str.to_code" => self.manager.mk_str_to_code(x),
            "str.from_code" => self.manager.mk_str_from_code(x),
            "str.to_int" | "str.to.int" => self.manager.mk_str_to_int(x),
            "int.to_str" | "int.to.str" | "str.from_int" => self.manager.mk_int_to_str(x),
            "str.to_re" | "str.to.re" => self.manager.mk_str_to_re(x),
            "re.*" => self.manager.mk_re_star(x),
            "re.+" => self.manager.mk_re_plus(x),
            "re.opt" => self.manager.mk_re_opt(x),
            "re.comp" => self.manager.mk_re_comp(x),
            _ => return Err(self.plan_mismatch(op)),
        };
        Ok(term)
    }

    /// Build a `Plan::Fixed(2)` built-in from its two operands.
    pub(super) fn build_binary(&mut self, op: &str, x: TermId, y: TermId) -> Result<TermId> {
        // SMT-LIB sort check: every binary bit-vector operator except
        // `concat` requires both operands to share one `(_ BitVec w)` sort.
        // Z3 rejects a mixed-width or non-bit-vector application at parse
        // time; accepting it here would intern a term no theory can encode
        // (the encoder now answers an honest Unknown for such a term, but
        // the standard-mandated answer is an error).
        if is_same_width_bv_op(op) {
            self.check_bv_binary_widths(op, x, y)?;
        }
        let term = match op {
            "mod" => self.manager.mk_mod(x, y),
            "select" => self.manager.mk_select(x, y),
            "bvand" => self.manager.mk_bv_and(x, y),
            "bvor" => self.manager.mk_bv_or(x, y),
            "bvadd" => self.manager.mk_bv_add(x, y),
            "bvsub" => self.manager.mk_bv_sub(x, y),
            "bvmul" => self.manager.mk_bv_mul(x, y),
            "bvult" => self.manager.mk_bv_ult(x, y),
            "bvslt" => self.manager.mk_bv_slt(x, y),
            "bvule" => self.manager.mk_bv_ule(x, y),
            "bvsle" => self.manager.mk_bv_sle(x, y),
            // bvugt(a, b) = bvult(b, a); bvsgt(a, b) = bvslt(b, a).
            "bvugt" => self.manager.mk_bv_ult(y, x),
            "bvsgt" => self.manager.mk_bv_slt(y, x),
            // bvuge(a, b) = NOT bvult(a, b); bvsge(a, b) = NOT bvslt(a, b).
            "bvuge" => {
                let ult = self.manager.mk_bv_ult(x, y);
                self.manager.mk_not(ult)
            }
            "bvsge" => {
                let slt = self.manager.mk_bv_slt(x, y);
                self.manager.mk_not(slt)
            }
            "bvxor" => self.manager.mk_bv_xor(x, y),
            "bvnand" => self.manager.mk_bv_nand(x, y),
            "bvnor" => self.manager.mk_bv_nor(x, y),
            "bvxnor" => self.manager.mk_bv_xnor(x, y),
            "bvcomp" => self.manager.mk_bv_comp(x, y),
            "bvsmod" => self.manager.mk_bv_smod(x, y),
            "bvudiv" => self.manager.mk_bv_udiv(x, y),
            "bvsdiv" => self.manager.mk_bv_sdiv(x, y),
            "bvurem" => self.manager.mk_bv_urem(x, y),
            "bvsrem" => self.manager.mk_bv_srem(x, y),
            "bvshl" => self.manager.mk_bv_shl(x, y),
            "bvlshr" => self.manager.mk_bv_lshr(x, y),
            "bvashr" => self.manager.mk_bv_ashr(x, y),
            "concat" => self.manager.try_mk_bv_concat(x, y)?,
            "fp.rem" => self.manager.mk_fp_rem(x, y),
            "fp.eq" => self.manager.mk_fp_eq(x, y),
            "fp.lt" => self.manager.mk_fp_lt(x, y),
            "fp.gt" => self.manager.mk_fp_gt(x, y),
            "fp.leq" => self.manager.mk_fp_leq(x, y),
            "fp.geq" => self.manager.mk_fp_geq(x, y),
            "fp.min" => self.manager.mk_fp_min(x, y),
            "fp.max" => self.manager.mk_fp_max(x, y),
            "str.at" => self.manager.mk_str_at(x, y),
            "str.contains" => self.manager.mk_str_contains(x, y),
            "str.prefixof" => self.manager.mk_str_prefixof(x, y),
            "str.suffixof" => self.manager.mk_str_suffixof(x, y),
            "str.in_re" | "str.in.re" => self.manager.mk_str_in_re(x, y),
            "re.diff" => self.manager.mk_re_diff(x, y),
            "re.range" => self.manager.mk_re_range(x, y),
            _ => return Err(self.plan_mismatch(op)),
        };
        Ok(term)
    }

    /// Build a `Plan::Fixed(3)` built-in from its three operands.
    pub(super) fn build_ternary(
        &mut self,
        op: &str,
        x: TermId,
        y: TermId,
        z: TermId,
    ) -> Result<TermId> {
        let term = match op {
            "ite" => self.manager.mk_ite(x, y, z),
            "store" => self.manager.mk_store(x, y, z),
            // Floating-point bit-triple literal constructor: (fp sign exp sig).
            "fp" => self.build_fp_lit(x, y, z)?,
            "str.substr" => self.manager.mk_str_substr(x, y, z),
            "str.indexof" => self.manager.mk_str_indexof(x, y, z),
            "str.replace" => self.manager.mk_str_replace(x, y, z),
            "str.replace_all" => self.manager.mk_str_replace_all(x, y, z),
            // `(str.replace_re s r t)` replaces the shortest *leftmost* match
            // of the regular language `r` in `s` by `t`; `str.replace_re_all`
            // replaces every shortest *non-empty* match, left to right. The
            // middle operand is a `RegLan` term, so evaluation needs the
            // theory's derivative engine and happens there.
            "str.replace_re" => self.manager.mk_str_replace_re(x, y, z),
            "str.replace_re_all" => self.manager.mk_str_replace_re_all(x, y, z),
            _ => return Err(self.plan_mismatch(op)),
        };
        Ok(term)
    }

    /// Build a floating-point operator whose rounding mode was consumed while
    /// its head was read (see `Parser::open_named_head`).
    ///
    /// A literal mode builds the `TermKind::Fp*` node directly; a symbolic one
    /// is resolved by [`Parser::expand_symbolic_rm`].
    pub(super) fn build_fp_rounded(
        &mut self,
        op: &str,
        rm: RmOperand,
        args: &[TermId],
    ) -> Result<TermId> {
        match rm {
            RmOperand::Concrete(rm) => self.build_fp_rounded_concrete(op, rm, args),
            RmOperand::Symbolic(mode) => {
                let args = args.to_vec();
                self.expand_symbolic_rm(mode, |parser, rm| {
                    parser.build_fp_rounded_concrete(op, rm, &args)
                })
            }
        }
    }

    /// Resolve a *symbolic* rounding mode into a five-way case split over the
    /// modes it can be:
    ///
    /// ```text
    /// (ite (= m RNE) op(RNE, ..)
    ///   (ite (= m RNA) op(RNA, ..)
    ///     (ite (= m RTP) op(RTP, ..)
    ///       (ite (= m RTN) op(RTN, ..)
    ///                      op(RTZ, ..)))))
    /// ```
    ///
    /// `build_fn` is called once per mode and produces that mode's concrete
    /// operator node, so every leaf of the split is an ordinary `TermKind::Fp*`
    /// term and the floating-point theory needs no notion of a symbolic mode
    /// at all.
    ///
    /// # The final `else` is not a default
    ///
    /// The last branch is `RTZ` *unguarded*, which is only correct because `m`
    /// is guaranteed to be one of the five: the solver asserts a closure axiom
    /// `(or (= m RNE) ... (= m RTZ))` for every declared rounding-mode
    /// constant, and the parser relativizes every `RoundingMode` quantifier
    /// binder the same way. Drop either guarantee and a sixth element of the
    /// sort would silently be *treated as* `RTZ` here while comparing distinct
    /// from all five — the exact shape that makes `(distinct m1 .. m6)` answer
    /// `sat` when it must be `unsat`.
    pub(super) fn expand_symbolic_rm(
        &mut self,
        rm_term: TermId,
        mut build_fn: impl FnMut(&mut Self, RoundingMode) -> Result<TermId>,
    ) -> Result<TermId> {
        let Some((&last, head)) = RoundingMode::ALL.split_last() else {
            return Err(self.plan_mismatch("fp rounding mode"));
        };
        // Built innermost-first, so the unguarded `else` is the last mode.
        let mut result = build_fn(self, last)?;
        for &mode in head.iter().rev() {
            let branch = build_fn(self, mode)?;
            let mode_term = self.manager.mk_rounding_mode(mode);
            let cond = self.manager.mk_eq(rm_term, mode_term);
            result = self.manager.mk_ite(cond, branch, result);
        }
        Ok(result)
    }

    /// Build a floating-point operator at one *literal* rounding mode.
    fn build_fp_rounded_concrete(
        &mut self,
        op: &str,
        rm: RoundingMode,
        args: &[TermId],
    ) -> Result<TermId> {
        let term = match (op, args) {
            ("fp.add", [x, y]) => self.manager.mk_fp_add(rm, *x, *y),
            ("fp.sub", [x, y]) => self.manager.mk_fp_sub(rm, *x, *y),
            ("fp.mul", [x, y]) => self.manager.mk_fp_mul(rm, *x, *y),
            ("fp.div", [x, y]) => self.manager.mk_fp_div(rm, *x, *y),
            ("fp.sqrt", [x]) => self.manager.mk_fp_sqrt(rm, *x),
            ("fp.roundToIntegral", [x]) => self.manager.mk_fp_round_to_integral(rm, *x),
            ("fp.fma", [x, y, z]) => self.manager.mk_fp_fma(rm, *x, *y, *z),
            _ => return Err(self.plan_mismatch(op)),
        };
        Ok(term)
    }

    /// Build a `Plan::Variadic` built-in from every operand that appeared
    /// before the closing parenthesis.
    ///
    /// The operators that have no n-ary term representation are folded into
    /// binary chains here. A fold turns `n` operands into a chain that is
    /// *deeper* than the one node the driver counted for this frame, so each
    /// such arm first charges the chain against the term-depth budget via
    /// [`Parser::charge_fold_depth`](super::terms) — otherwise a flat
    /// `(str.++ x1 … x100000)` of syntactic depth 2 would build a
    /// 100 000-deep term while `MAX_PARSE_DEPTH` reported it as depth 2.
    pub(super) fn build_variadic(&mut self, op: &str, args: &[TermId]) -> Result<TermId> {
        let term = match op {
            "and" => self.manager.mk_and(args.iter().copied()),
            "or" => self.manager.mk_or(args.iter().copied()),
            "distinct" => self.manager.mk_distinct(args.iter().copied()),
            "+" => self.manager.mk_add(args.iter().copied()),
            "*" => self.manager.mk_mul(args.iter().copied()),
            "re.++" => self.manager.mk_re_concat(args.iter().copied()),
            "re.union" => self.manager.mk_re_union(args.iter().copied()),
            "re.inter" => self.manager.mk_re_inter(args.iter().copied()),
            // `=>` is right-associative n-ary: `(=> a b c)` means
            // `(=> a (=> b c))`.
            "=>" => {
                let Some((&last, init)) = args.split_last() else {
                    return Err(self.min_arity_err("=>", 2, 0));
                };
                if init.is_empty() {
                    return Err(self.min_arity_err("=>", 2, args.len()));
                }
                self.charge_fold_depth(chain_depth(args.len(), 1))?;
                let mut result = last;
                for &lhs in init.iter().rev() {
                    result = self.manager.mk_implies(lhs, result);
                }
                result
            }
            // `xor` is left-associative n-ary: `(xor a b c)` means
            // `(xor (xor a b) c)`.
            "xor" => {
                let Some((&first, rest)) = args.split_first() else {
                    return Err(self.min_arity_err("xor", 2, 0));
                };
                if rest.is_empty() {
                    return Err(self.min_arity_err("xor", 2, args.len()));
                }
                // Each fold step wraps the accumulator in `Or(And(Not(..)))`,
                // so a step costs three levels, not one.
                self.charge_fold_depth(chain_depth(args.len(), 3))?;
                let mut result = first;
                for &next in rest {
                    result = self.mk_xor2(result, next);
                }
                result
            }
            // Chainable comparisons: `(op a b c)` means
            // `(and (op a b) (op b c))`.
            "=" | "<" | "<=" | ">" | ">=" | "str.<" | "str.<=" => {
                if args.len() < 2 {
                    return Err(self.min_arity_err(op, 2, args.len()));
                }
                let mut atoms = Vec::with_capacity(args.len() - 1);
                for pair in args.windows(2) {
                    let (a, b) = (pair[0], pair[1]);
                    atoms.push(match op {
                        "=" => self.manager.mk_eq(a, b),
                        "<" => self.manager.mk_lt(a, b),
                        "<=" => self.manager.mk_le(a, b),
                        ">" => self.manager.mk_gt(a, b),
                        ">=" => self.manager.mk_ge(a, b),
                        "str.<" => self.manager.mk_str_lt(a, b),
                        _ => self.manager.mk_str_le(a, b),
                    });
                }
                self.chain_conjunction(atoms)
            }
            // `-` is unary negation with one operand and left-associative
            // n-ary subtraction otherwise: `(- a b c)` means `(- (- a b) c)`.
            "-" => {
                let Some((&first, rest)) = args.split_first() else {
                    return Err(self.min_arity_err("-", 1, 0));
                };
                if rest.is_empty() {
                    self.manager.mk_neg(first)
                } else {
                    self.charge_fold_depth(chain_depth(args.len(), 1))?;
                    let mut result = first;
                    for &next in rest {
                        result = self.manager.mk_sub(result, next);
                    }
                    result
                }
            }
            // Integer (Euclidean) division and real division are both
            // left-associative n-ary. Real `/` is routed to the same general
            // division term kind (which the rewriter/evaluator interpret as
            // exact rational division) so QF_LRA constraints stay in the
            // arithmetic theory instead of degrading to a Bool apply.
            "div" | "/" => {
                let Some((&first, rest)) = args.split_first() else {
                    return Err(self.min_arity_err(op, 1, 0));
                };
                self.charge_fold_depth(chain_depth(args.len(), 1))?;
                let mut result = first;
                for &next in rest {
                    result = self.manager.mk_div(result, next);
                }
                result
            }
            "str.++" => {
                let Some((&first, rest)) = args.split_first() else {
                    return Err(self.min_arity_err("str.++", 1, 0));
                };
                self.charge_fold_depth(chain_depth(args.len(), 1))?;
                let mut result = first;
                for &next in rest {
                    result = self.manager.mk_str_concat(result, next);
                }
                result
            }
            _ => return Err(self.plan_mismatch(op)),
        };
        Ok(term)
    }
}

/// Binary operators that require both operands to share one `(_ BitVec w)`
/// sort — every SMT-LIB binary bit-vector operator except `concat`, which
/// joins different widths by design. The comparison rewrites (`bvugt` and
/// friends) are included because their operands face the same rule before
/// the swap.
fn is_same_width_bv_op(op: &str) -> bool {
    matches!(
        op,
        "bvand"
            | "bvor"
            | "bvadd"
            | "bvsub"
            | "bvmul"
            | "bvult"
            | "bvslt"
            | "bvule"
            | "bvsle"
            | "bvugt"
            | "bvsgt"
            | "bvuge"
            | "bvsge"
            | "bvxor"
            | "bvnand"
            | "bvnor"
            | "bvxnor"
            | "bvcomp"
            | "bvsmod"
            | "bvudiv"
            | "bvsdiv"
            | "bvurem"
            | "bvsrem"
            | "bvshl"
            | "bvlshr"
            | "bvashr"
    )
}

/// Depth of the binary chain that folding `operands` operands builds, when
/// each fold step adds `per_step` levels of nesting.
///
/// Saturating throughout: the operand count comes straight from the input and
/// is therefore attacker-controlled, and a wrapped product would understate
/// the very cost it exists to bound.
fn chain_depth(operands: usize, per_step: u32) -> u32 {
    let steps = u32::try_from(operands.saturating_sub(1)).unwrap_or(u32::MAX);
    steps.saturating_mul(per_step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    /// Every operator name [`operand_plan`] recognises.
    ///
    /// Kept in the same order as the plan table so the two can be diffed by
    /// eye when an operator is added.
    const PLANNED_OPS: &[&str] = &[
        // Fixed(1)
        "not",
        "abs",
        "to_real",
        "to_int",
        "is_int",
        "bvnot",
        "bvneg",
        "fp.isNormal",
        "fp.isSubnormal",
        "fp.isZero",
        "fp.isInfinite",
        "fp.isNaN",
        "fp.isNegative",
        "fp.isPositive",
        "fp.abs",
        "fp.neg",
        "fp.to_real",
        "str.len",
        "str.is_digit",
        "str.to_code",
        "str.from_code",
        "str.to_int",
        "str.to.int",
        "int.to_str",
        "int.to.str",
        "str.from_int",
        "str.to_re",
        "str.to.re",
        "re.*",
        "re.+",
        "re.opt",
        "re.comp",
        // Fixed(2)
        "mod",
        "select",
        "bvand",
        "bvor",
        "bvadd",
        "bvsub",
        "bvmul",
        "bvult",
        "bvslt",
        "bvule",
        "bvsle",
        "bvugt",
        "bvsgt",
        "bvuge",
        "bvsge",
        "bvxor",
        "bvnand",
        "bvnor",
        "bvxnor",
        "bvcomp",
        "bvsmod",
        "bvudiv",
        "bvsdiv",
        "bvurem",
        "bvsrem",
        "bvshl",
        "bvlshr",
        "bvashr",
        "concat",
        "fp.rem",
        "fp.eq",
        "fp.lt",
        "fp.gt",
        "fp.leq",
        "fp.geq",
        "fp.min",
        "fp.max",
        "str.at",
        "str.contains",
        "str.prefixof",
        "str.suffixof",
        "str.in_re",
        "str.in.re",
        "re.diff",
        "re.range",
        // Fixed(3)
        "ite",
        "store",
        "fp",
        "str.substr",
        "str.indexof",
        "str.replace",
        "str.replace_all",
        "str.replace_re",
        "str.replace_re_all",
        // Variadic
        "and",
        "or",
        "=>",
        "xor",
        "=",
        "distinct",
        "+",
        "-",
        "*",
        "div",
        "/",
        "<",
        "<=",
        ">",
        ">=",
        "str.++",
        "str.<",
        "str.<=",
        "re.++",
        "re.union",
        "re.inter",
    ];

    /// The floating-point operators whose head consumes a rounding mode, with
    /// the operand count each one takes after it.
    const ROUNDED_OPS: &[(&str, usize)] = &[
        ("fp.add", 2),
        ("fp.sub", 2),
        ("fp.mul", 2),
        ("fp.div", 2),
        ("fp.sqrt", 1),
        ("fp.roundToIntegral", 1),
        ("fp.fma", 3),
    ];

    /// The message `plan_mismatch` produces; reaching it means an operator has
    /// an operand plan but no builder arm.
    const MISMATCH: &str = "internal: no builder for planned operator";

    /// A placeholder operand for `op`.
    ///
    /// Bit-vector operators get a bit-vector because `try_mk_bv_concat`
    /// returns a sort error for a width-less operand; everything else gets an
    /// integer, which the builders that care about sorts reject through their
    /// own error path.
    fn placeholder(parser: &mut Parser<'_>, op: &str) -> TermId {
        if op.starts_with("bv") || *op == *"concat" {
            parser.manager.mk_bitvec(0, 8)
        } else {
            parser.manager.mk_int(0)
        }
    }

    /// An operator that [`operand_plan`] lists but whose `build_*` arm was
    /// never written would reach users as an "internal:" parse error. Drive
    /// every planned name into the builder that its plan selects and assert
    /// none of them falls through.
    ///
    /// The operands are deliberately loose placeholders: builders that care
    /// reject them with their *own* message, which is exactly what
    /// distinguishes a real operator from a missing arm.
    #[test]
    fn planned_operators_all_reach_a_builder() {
        for op in PLANNED_OPS {
            let mut manager = TermManager::new();
            let mut parser = Parser::new("", &mut manager);
            let x = placeholder(&mut parser, op);
            let plan = operand_plan(op).unwrap_or_else(|| panic!("{op} has no operand plan"));
            let result = match plan {
                Plan::Fixed(1) => parser.build_unary(op, x),
                Plan::Fixed(2) => parser.build_binary(op, x, x),
                Plan::Fixed(3) => parser.build_ternary(op, x, x, x),
                Plan::Fixed(n) => panic!("{op} has an unsupported fixed arity {n}"),
                Plan::Variadic => parser.build_variadic(op, &[x, x]),
            };
            if let Err(e) = result {
                assert!(
                    !e.to_string().contains(MISMATCH),
                    "{op} has an operand plan but no builder arm"
                );
            }
        }
    }

    /// The same guard for the rounding-mode family, which bypasses
    /// [`operand_plan`] because its head consumes a token of its own.
    ///
    /// Both rounding-mode spellings are driven: a literal mode must reach a
    /// builder arm, and so must a *symbolic* one — the symbolic path calls the
    /// very same arms once per mode through
    /// [`Parser::expand_symbolic_rm`](super::Parser::expand_symbolic_rm), so an
    /// operator missing an arm would surface as the same "internal:" error
    /// there.
    #[test]
    fn rounded_operators_all_reach_a_builder() {
        for (op, arity) in ROUNDED_OPS {
            let mut manager = TermManager::new();
            let mut parser = Parser::new("", &mut manager);
            let x = parser.manager.mk_int(0);
            let args = vec![x; *arity];
            let mode = parser.manager.mk_rounding_mode(RoundingMode::RNE);
            for rm in [
                RmOperand::Concrete(RoundingMode::RNE),
                RmOperand::Symbolic(mode),
            ] {
                if let Err(e) = parser.build_fp_rounded(op, rm, &args) {
                    assert!(
                        !e.to_string().contains(MISMATCH),
                        "{op} takes a rounding mode but has no builder arm ({rm:?})"
                    );
                }
            }
        }
    }

    /// A name that is *not* a built-in must fall through to the declaration
    /// tables rather than being claimed by the plan table.
    #[test]
    fn user_symbols_have_no_operand_plan() {
        for name in ["f", "my_fun", "bvmine", "str.mine", "cons", "head"] {
            assert!(
                operand_plan(name).is_none(),
                "{name} must not be treated as a built-in operator"
            );
        }
    }

    /// `(bvadd x8 y16)` and friends are sort errors in SMT-LIB, and Z3
    /// rejects them at parse time. `TermManager::mk_bv_*` would silently
    /// intern the term at the left operand's width, so the parser must be
    /// the layer that says no.
    #[test]
    fn mixed_width_bv_binary_ops_are_rejected_at_parse_time() {
        for op in [
            "bvadd", "bvand", "bvult", "bvugt", "bvshl", "bvcomp", "bvudiv",
        ] {
            let script = format!(
                "(declare-const a (_ BitVec 8)) (declare-const b (_ BitVec 16)) \
                 (assert (= (_ bv0 1) (_ bv0 1))) (assert ({op} a b))"
            );
            let mut manager = TermManager::new();
            let err = crate::smtlib::parser::parse_script(&script, &mut manager)
                .expect_err("mixed widths must not parse");
            assert!(
                err.to_string().contains("same bit-vector width"),
                "{op}: unexpected error {err}"
            );
        }
    }

    /// A non-bit-vector operand under a bit-vector operator is a sort error,
    /// not a term the theories should ever see.
    #[test]
    fn non_bv_operand_under_bv_op_is_rejected_at_parse_time() {
        let script = "(declare-const a (_ BitVec 8)) (assert (bvult a 3))";
        let mut manager = TermManager::new();
        let err = crate::smtlib::parser::parse_script(script, &mut manager)
            .expect_err("Int operand under bvult must not parse");
        assert!(
            err.to_string().contains("bit-vector sorts"),
            "unexpected error {err}"
        );
    }

    /// The rule must not over-fire: equal widths still parse, and `concat`
    /// joins different widths by design.
    #[test]
    fn same_width_and_concat_still_parse() {
        let script = "(declare-const a (_ BitVec 8)) (declare-const b (_ BitVec 8)) \
                      (declare-const w (_ BitVec 16)) \
                      (assert (bvult (bvadd a b) a)) \
                      (assert (= (concat a b) w))";
        let mut manager = TermManager::new();
        let commands = crate::smtlib::parser::parse_script(script, &mut manager)
            .expect("well-sorted script must parse");
        assert_eq!(commands.len(), 5);
    }
}
