//! Term builder methods for TermManager — all mk_* constructors

use super::super::term::{RoundingMode, TermId, TermKind};
use crate::error::{OxizError, Result};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use num_bigint::BigInt;
use num_rational::Rational64;
use smallvec::SmallVec;

use super::TermManager;
use super::bv_fold;
use super::str_fold;

/// Canonicalize operand order for commutative binary operators so that
/// `op(a, b)` and `op(b, a)` hash-cons to the same term.
fn canonical_pair(lhs: TermId, rhs: TermId) -> (TermId, TermId) {
    if lhs.0 <= rhs.0 {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    }
}

impl TermManager {
    /// Create the boolean true constant
    #[must_use]
    pub fn mk_true(&self) -> TermId {
        self.true_id
    }

    /// Create the boolean false constant
    #[must_use]
    pub fn mk_false(&self) -> TermId {
        self.false_id
    }

    /// Create a boolean constant
    #[must_use]
    pub fn mk_bool(&self, value: bool) -> TermId {
        if value { self.true_id } else { self.false_id }
    }

    /// Create an integer constant
    pub fn mk_int(&mut self, value: impl Into<BigInt>) -> TermId {
        let sort = self.sorts.int_sort;
        self.intern(TermKind::IntConst(value.into()), sort)
    }

    /// Create a rational constant
    pub fn mk_real(&mut self, value: Rational64) -> TermId {
        let sort = self.sorts.real_sort;
        self.intern(TermKind::RealConst(value), sort)
    }

    /// Create a bit vector constant
    pub fn mk_bitvec(&mut self, value: impl Into<BigInt>, width: u32) -> TermId {
        let sort = self.sorts.bitvec(width);
        self.intern(
            TermKind::BitVecConst {
                value: value.into(),
                width,
            },
            sort,
        )
    }

    /// Create a named variable
    pub fn mk_var(&mut self, name: &str, sort: SortId) -> TermId {
        let spur = self.intern_str(name);
        self.intern(TermKind::Var(spur), sort)
    }

    /// Create a logical NOT
    pub fn mk_not(&mut self, arg: TermId) -> TermId {
        // Simplify double negation
        if let Some(term) = self.get(arg) {
            if let TermKind::Not(inner) = term.kind {
                return inner;
            }
            if let TermKind::True = term.kind {
                return self.false_id;
            }
            if let TermKind::False = term.kind {
                return self.true_id;
            }
        }

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Not(arg), sort)
    }

    /// Create a logical AND
    ///
    /// # Performance
    ///
    /// This flattens a nested `And` child into the result and re-interns the
    /// whole operand list on every call. Accumulating in a loop —
    /// `acc = manager.mk_and([acc, lit])` — is therefore Theta(n^2) over the
    /// number of iterations, since each call re-flattens and re-interns the
    /// growing accumulator. Collect all operands first and call `mk_and`
    /// once on the full list instead.
    pub fn mk_and(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let mut flat_args: SmallVec<[TermId; 4]> = SmallVec::new();

        for arg in args {
            if let Some(term) = self.get(arg) {
                match &term.kind {
                    TermKind::False => return self.false_id,
                    TermKind::True => continue,
                    TermKind::And(inner) => flat_args.extend(inner.iter().copied()),
                    _ => flat_args.push(arg),
                }
            } else {
                flat_args.push(arg);
            }
        }

        match flat_args.len() {
            0 => self.true_id,
            1 => flat_args[0],
            _ => {
                let sort = self.sorts.bool_sort;
                self.intern(TermKind::And(flat_args), sort)
            }
        }
    }

    /// Create a logical OR
    ///
    /// # Performance
    ///
    /// This flattens a nested `Or` child into the result and re-interns the
    /// whole operand list on every call. Accumulating in a loop —
    /// `acc = manager.mk_or([acc, lit])` — is therefore Theta(n^2) over the
    /// number of iterations, since each call re-flattens and re-interns the
    /// growing accumulator. Collect all operands first and call `mk_or`
    /// once on the full list instead.
    pub fn mk_or(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let mut flat_args: SmallVec<[TermId; 4]> = SmallVec::new();

        for arg in args {
            if let Some(term) = self.get(arg) {
                match &term.kind {
                    TermKind::True => return self.true_id,
                    TermKind::False => continue,
                    TermKind::Or(inner) => flat_args.extend(inner.iter().copied()),
                    _ => flat_args.push(arg),
                }
            } else {
                flat_args.push(arg);
            }
        }

        match flat_args.len() {
            0 => self.false_id,
            1 => flat_args[0],
            _ => {
                let sort = self.sorts.bool_sort;
                self.intern(TermKind::Or(flat_args), sort)
            }
        }
    }

    /// Create a logical implication
    pub fn mk_implies(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Simplifications
        if let Some(term) = self.get(lhs) {
            if let TermKind::False = term.kind {
                return self.true_id;
            }
            if let TermKind::True = term.kind {
                return rhs;
            }
        }
        if let Some(term) = self.get(rhs)
            && let TermKind::True = term.kind
        {
            return self.true_id;
        }

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Implies(lhs, rhs), sort)
    }

    /// Create a logical XOR
    pub fn mk_xor(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Simplifications
        if lhs == rhs {
            return self.false_id;
        }
        if let Some(term) = self.get(lhs) {
            if let TermKind::False = term.kind {
                return rhs;
            }
            if let TermKind::True = term.kind {
                return self.mk_not(rhs);
            }
        }
        if let Some(term) = self.get(rhs) {
            if let TermKind::False = term.kind {
                return lhs;
            }
            if let TermKind::True = term.kind {
                return self.mk_not(lhs);
            }
        }

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Xor(lhs, rhs), sort)
    }

    /// Create an if-then-else
    pub fn mk_ite(&mut self, cond: TermId, then_branch: TermId, else_branch: TermId) -> TermId {
        // Simplifications
        if let Some(term) = self.get(cond) {
            if let TermKind::True = term.kind {
                return then_branch;
            }
            if let TermKind::False = term.kind {
                return else_branch;
            }
        }
        if then_branch == else_branch {
            return then_branch;
        }
        // ite(c, true, false) => c
        let then_is_true = self
            .get(then_branch)
            .is_some_and(|t| matches!(t.kind, TermKind::True));
        let else_is_false = self
            .get(else_branch)
            .is_some_and(|t| matches!(t.kind, TermKind::False));
        if then_is_true && else_is_false {
            return cond;
        }

        let sort = self
            .get(then_branch)
            .map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::Ite(cond, then_branch, else_branch), sort)
    }

    /// Create an equality
    pub fn mk_eq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if lhs == rhs {
            return self.true_id;
        }

        // Check for constant comparisons
        let lhs_kind = self.get(lhs).map(|t| t.kind.clone());
        let rhs_kind = self.get(rhs).map(|t| t.kind.clone());

        match (&lhs_kind, &rhs_kind) {
            // Integer constants
            (Some(TermKind::IntConst(a)), Some(TermKind::IntConst(b))) => {
                return self.mk_bool(a == b);
            }
            // Boolean constants
            (Some(TermKind::True), Some(TermKind::True)) => return self.true_id,
            (Some(TermKind::False), Some(TermKind::False)) => return self.true_id,
            (Some(TermKind::True), Some(TermKind::False)) => return self.false_id,
            (Some(TermKind::False), Some(TermKind::True)) => return self.false_id,
            // BitVec constants
            (
                Some(TermKind::BitVecConst {
                    value: v1,
                    width: w1,
                }),
                Some(TermKind::BitVecConst {
                    value: v2,
                    width: w2,
                }),
            ) => {
                return self.mk_bool(v1 == v2 && w1 == w2);
            }
            _ => {}
        }

        // Canonicalize order
        let (lhs, rhs) = if lhs.0 <= rhs.0 {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Eq(lhs, rhs), sort)
    }

    /// Create a distinct constraint
    pub fn mk_distinct(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();

        if args.len() <= 1 {
            return self.true_id;
        }

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Distinct(args), sort)
    }

    /// Create an addition
    pub fn mk_add(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();

        match args.len() {
            0 => self.mk_int(0),
            1 => args[0],
            _ => {
                let sort = self.get(args[0]).map_or(self.sorts.int_sort, |t| t.sort);
                self.intern(TermKind::Add(args), sort)
            }
        }
    }

    /// Create a subtraction
    pub fn mk_sub(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Sub(lhs, rhs), sort)
    }

    /// Create arithmetic negation
    pub fn mk_neg(&mut self, arg: TermId) -> TermId {
        let sort = self.get(arg).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Neg(arg), sort)
    }

    /// Create a multiplication
    pub fn mk_mul(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();

        match args.len() {
            0 => self.mk_int(1),
            1 => args[0],
            _ => {
                let sort = self.get(args[0]).map_or(self.sorts.int_sort, |t| t.sort);
                self.intern(TermKind::Mul(args), sort)
            }
        }
    }

    /// Create a division
    pub fn mk_div(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Div(lhs, rhs), sort)
    }

    /// Create a modulo operation
    pub fn mk_mod(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Mod(lhs, rhs), sort)
    }

    /// Create a less-than comparison
    pub fn mk_lt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Lt(lhs, rhs), sort)
    }

    /// Create a less-than-or-equal comparison
    pub fn mk_le(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Le(lhs, rhs), sort)
    }

    /// Create a greater-than comparison
    pub fn mk_gt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Gt(lhs, rhs), sort)
    }

    /// Create a greater-than-or-equal comparison
    pub fn mk_ge(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Ge(lhs, rhs), sort)
    }

    /// Create a greater-than-or-equal comparison (alias for mk_ge)
    pub fn mk_geq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_ge(lhs, rhs)
    }

    /// Create a less-than-or-equal comparison (alias for mk_le)
    pub fn mk_leq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_le(lhs, rhs)
    }

    /// Create an array select operation
    pub fn mk_select(&mut self, array: TermId, index: TermId) -> TermId {
        // Get the range sort from the array's sort
        let sort = if let Some(term) = self.get(array) {
            if let Some(array_sort) = self.sorts.get(term.sort) {
                if let crate::sort::SortKind::Array { range, .. } = array_sort.kind {
                    range
                } else {
                    self.sorts.int_sort
                }
            } else {
                self.sorts.int_sort
            }
        } else {
            self.sorts.int_sort
        };
        self.intern(TermKind::Select(array, index), sort)
    }

    /// Create an array store operation
    pub fn mk_store(&mut self, array: TermId, index: TermId, value: TermId) -> TermId {
        let sort = self.get(array).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Store(array, index, value), sort)
    }

    /// Create a string literal
    pub fn mk_string_lit(&mut self, value: &str) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StringLit(value.to_string()), string_sort)
    }

    /// Create a string concatenation
    pub fn mk_str_concat(&mut self, s1: TermId, s2: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrConcat(s1, s2), string_sort)
    }

    /// Create a string length operation
    pub fn mk_str_len(&mut self, s: TermId) -> TermId {
        let int_sort = self.sorts.int_sort;
        self.intern(TermKind::StrLen(s), int_sort)
    }

    /// Create a substring operation
    pub fn mk_str_substr(&mut self, s: TermId, start: TermId, len: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrSubstr(s, start, len), string_sort)
    }

    /// Create a character at index operation
    pub fn mk_str_at(&mut self, s: TermId, i: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrAt(s, i), string_sort)
    }

    /// Create a contains substring operation
    pub fn mk_str_contains(&mut self, s: TermId, sub: TermId) -> TermId {
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrContains(s, sub), bool_sort)
    }

    /// Create a prefix check operation
    pub fn mk_str_prefixof(&mut self, prefix: TermId, s: TermId) -> TermId {
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrPrefixOf(prefix, s), bool_sort)
    }

    /// Create a suffix check operation
    pub fn mk_str_suffixof(&mut self, suffix: TermId, s: TermId) -> TermId {
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrSuffixOf(suffix, s), bool_sort)
    }

    /// Create an index of operation
    pub fn mk_str_indexof(&mut self, s: TermId, sub: TermId, offset: TermId) -> TermId {
        let int_sort = self.sorts.int_sort;
        self.intern(TermKind::StrIndexOf(s, sub, offset), int_sort)
    }

    /// Create a string replace operation
    pub fn mk_str_replace(&mut self, s: TermId, pattern: TermId, replacement: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrReplace(s, pattern, replacement), string_sort)
    }

    /// Create a replace all operation
    pub fn mk_str_replace_all(
        &mut self,
        s: TermId,
        pattern: TermId,
        replacement: TermId,
    ) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(
            TermKind::StrReplaceAll(s, pattern, replacement),
            string_sort,
        )
    }

    /// `str.replace_re` — replace the leftmost shortest match of a regular
    /// language.
    ///
    /// The regex operand carries the reserved `RegLan` sort (see
    /// [`Self::reglan_sort`]); the theory compiles it with its Brzozowski
    /// derivative engine, so no folding happens here (`oxiz-core` deliberately
    /// hosts no regex matcher — `str.in_re` is left symbolic for the same
    /// reason).
    pub fn mk_str_replace_re(&mut self, s: TermId, re: TermId, replacement: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrReplaceRe(s, re, replacement), string_sort)
    }

    /// `str.replace_re_all` — replace every shortest non-empty match of a
    /// regular language, scanning left to right.
    pub fn mk_str_replace_re_all(&mut self, s: TermId, re: TermId, replacement: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrReplaceReAll(s, re, replacement), string_sort)
    }

    /// `str.<` — strict lexicographic order over code points.
    ///
    /// Folded on constant operands, and simplified on the three shapes whose
    /// truth is fixed by the order's structure alone. Reference: Z3's
    /// `seq_rewriter.cpp` `mk_str_lt`, which applies the same empty-operand
    /// rules.
    pub fn mk_str_lt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Irreflexivity. Terms are hash-consed, so equal ids are the same term.
        if lhs == rhs {
            return self.mk_false();
        }
        if let (Some(a), Some(b)) = (self.string_lit_of(lhs), self.string_lit_of(rhs)) {
            return if str_fold::str_lt(&a, &b) {
                self.mk_true()
            } else {
                self.mk_false()
            };
        }
        // Nothing is strictly below the empty string, which is the minimum.
        if self.is_empty_string_lit(rhs) {
            return self.mk_false();
        }
        // `"" < b` iff `b` is not itself empty.
        if self.is_empty_string_lit(lhs) {
            let eq = self.mk_eq(lhs, rhs);
            return self.mk_not(eq);
        }
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrLt(lhs, rhs), bool_sort)
    }

    /// `str.<=` — the reflexive closure of [`Self::mk_str_lt`].
    pub fn mk_str_le(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if lhs == rhs {
            return self.mk_true();
        }
        if let (Some(a), Some(b)) = (self.string_lit_of(lhs), self.string_lit_of(rhs)) {
            return if str_fold::str_le(&a, &b) {
                self.mk_true()
            } else {
                self.mk_false()
            };
        }
        // The empty string is below everything.
        if self.is_empty_string_lit(lhs) {
            return self.mk_true();
        }
        // `a <= ""` iff `a` is itself empty.
        if self.is_empty_string_lit(rhs) {
            return self.mk_eq(lhs, rhs);
        }
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrLe(lhs, rhs), bool_sort)
    }

    /// `str.to_code` — the code point of a singleton string, `-1` otherwise.
    pub fn mk_str_to_code(&mut self, s: TermId) -> TermId {
        if let Some(value) = self.string_lit_of(s) {
            return self.mk_int(str_fold::str_to_code(&value));
        }
        let int_sort = self.sorts.int_sort;
        self.intern(TermKind::StrToCode(s), int_sort)
    }

    /// `str.from_code` — the singleton string for a code point in the
    /// theory's alphabet, `""` outside it.
    ///
    /// A surrogate code point is deliberately left unfolded; see
    /// [`str_fold::FromCode::Unrepresentable`].
    pub fn mk_str_from_code(&mut self, n: TermId) -> TermId {
        if let Some(TermKind::IntConst(value)) = self.get(n).map(|t| t.kind.clone()) {
            match str_fold::str_from_code(&value) {
                str_fold::FromCode::Char(c) => {
                    let mut text = String::new();
                    text.push(c);
                    return self.mk_string_lit(&text);
                }
                str_fold::FromCode::Empty => return self.mk_string_lit(""),
                str_fold::FromCode::Unrepresentable => {}
            }
        }
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrFromCode(n), string_sort)
    }

    /// The value of `term` when it is a string literal, else `None`.
    fn string_lit_of(&self, term: TermId) -> Option<String> {
        match self.get(term).map(|t| &t.kind) {
            Some(TermKind::StringLit(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Whether `term` is the empty string literal.
    fn is_empty_string_lit(&self, term: TermId) -> bool {
        matches!(self.get(term).map(|t| &t.kind), Some(TermKind::StringLit(s)) if s.is_empty())
    }

    /// Create a string to integer conversion
    pub fn mk_str_to_int(&mut self, s: TermId) -> TermId {
        let int_sort = self.sorts.int_sort;
        self.intern(TermKind::StrToInt(s), int_sort)
    }

    /// Create an integer to string conversion
    pub fn mk_int_to_str(&mut self, i: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::IntToStr(i), string_sort)
    }

    /// Create a string in regex operation
    pub fn mk_str_in_re(&mut self, s: TermId, re: TermId) -> TermId {
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrInRe(s, re), bool_sort)
    }

    // ==================== Regular-expression (RegLan) terms ====================
    //
    // The SMT-LIB Strings theory `RegLan` sort has no dedicated `SortKind`
    // variant (that enum is matched exhaustively across sibling crates, so it
    // cannot be extended here). Instead `RegLan` is modelled as a reserved,
    // interned built-in sort (`Uninterpreted("RegLan")`) obtained through the
    // regular sort-creation API, and each regex operator is represented as an
    // `Apply` node whose function symbol is the canonical SMT-LIB operator name
    // (`re.++`, `re.union`, ...). The reserved name never collides with a
    // user-declared sort because the parser rejects `RegLan` as a declarable
    // sort name. The strings theory (`oxiz-theories`) recognises these nodes by
    // their function symbol and compiles them into a Brzozowski-derivative
    // regex for membership solving.

    /// Get (interning on first use) the reserved built-in `RegLan` sort used
    /// as the sort of every regular-expression term.
    pub fn reglan_sort(&mut self) -> SortId {
        let spur = self.intern_str("RegLan");
        self.sorts
            .intern(crate::sort::SortKind::Uninterpreted(spur))
    }

    /// Build a regular-expression operator node (`Apply` with the canonical
    /// SMT-LIB operator name and `RegLan` sort).
    fn mk_regex_op(&mut self, name: &str, args: impl IntoIterator<Item = TermId>) -> TermId {
        let sort = self.reglan_sort();
        self.mk_apply(name, args, sort)
    }

    /// `re.none` — the empty regular language.
    pub fn mk_re_none(&mut self) -> TermId {
        self.mk_regex_op("re.none", core::iter::empty())
    }

    /// `re.all` — the language of all strings.
    pub fn mk_re_all(&mut self) -> TermId {
        self.mk_regex_op("re.all", core::iter::empty())
    }

    /// `re.allchar` — the language of all single-character strings.
    pub fn mk_re_all_char(&mut self) -> TermId {
        self.mk_regex_op("re.allchar", core::iter::empty())
    }

    /// `str.to_re` — singleton language containing exactly one string.
    pub fn mk_str_to_re(&mut self, s: TermId) -> TermId {
        self.mk_regex_op("str.to_re", [s])
    }

    /// `re.++` — regular-language concatenation.
    pub fn mk_re_concat(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        self.mk_regex_op("re.++", args)
    }

    /// `re.union` — regular-language union.
    pub fn mk_re_union(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        self.mk_regex_op("re.union", args)
    }

    /// `re.inter` — regular-language intersection.
    pub fn mk_re_inter(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        self.mk_regex_op("re.inter", args)
    }

    /// `re.*` — Kleene star.
    pub fn mk_re_star(&mut self, re: TermId) -> TermId {
        self.mk_regex_op("re.*", [re])
    }

    /// `re.+` — Kleene plus (one or more).
    pub fn mk_re_plus(&mut self, re: TermId) -> TermId {
        self.mk_regex_op("re.+", [re])
    }

    /// `re.opt` — optional (zero or one).
    pub fn mk_re_opt(&mut self, re: TermId) -> TermId {
        self.mk_regex_op("re.opt", [re])
    }

    /// `re.comp` — complement.
    pub fn mk_re_comp(&mut self, re: TermId) -> TermId {
        self.mk_regex_op("re.comp", [re])
    }

    /// `re.diff` — difference of two regular languages.
    pub fn mk_re_diff(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_regex_op("re.diff", [lhs, rhs])
    }

    /// `re.range` — the language of single-character strings between two
    /// one-character string literals (`lo` and `hi`, passed through as the
    /// operator's operands).
    pub fn mk_re_range(&mut self, lo: TermId, hi: TermId) -> TermId {
        self.mk_regex_op("re.range", [lo, hi])
    }

    /// `(_ re.^ n) re` — the regex repeated exactly `n` times. The repetition
    /// count is encoded as a leading `Int` operand.
    pub fn mk_re_power(&mut self, n: u32, re: TermId) -> TermId {
        let count = self.mk_int(n);
        self.mk_regex_op("re.^", [count, re])
    }

    /// `(_ re.loop lo hi) re` — the regex repeated between `lo` and `hi` times.
    /// The bounds are encoded as two leading `Int` operands.
    pub fn mk_re_loop(&mut self, lo: u32, hi: u32, re: TermId) -> TermId {
        let lo_t = self.mk_int(lo);
        let hi_t = self.mk_int(hi);
        self.mk_regex_op("re.loop", [lo_t, hi_t, re])
    }

    // Floating-point operations

    /// Create a floating-point literal from components
    pub fn mk_fp_lit(
        &mut self,
        sign: bool,
        exp: impl Into<BigInt>,
        sig: impl Into<BigInt>,
        eb: u32,
        sb: u32,
    ) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(
            TermKind::FpLit {
                sign,
                exp: exp.into(),
                sig: sig.into(),
                eb,
                sb,
            },
            sort,
        )
    }

    /// Create floating-point positive infinity
    pub fn mk_fp_plus_infinity(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpPlusInfinity { eb, sb }, sort)
    }

    /// Create floating-point negative infinity
    pub fn mk_fp_minus_infinity(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpMinusInfinity { eb, sb }, sort)
    }

    /// Create floating-point positive zero
    pub fn mk_fp_plus_zero(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpPlusZero { eb, sb }, sort)
    }

    /// Create floating-point negative zero
    pub fn mk_fp_minus_zero(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpMinusZero { eb, sb }, sort)
    }

    /// Create floating-point NaN
    pub fn mk_fp_nan(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpNaN { eb, sb }, sort)
    }

    /// Create floating-point absolute value
    pub fn mk_fp_abs(&mut self, arg: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(arg).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpAbs(arg), sort)
    }

    /// Create floating-point negation
    pub fn mk_fp_neg(&mut self, arg: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(arg).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpNeg(arg), sort)
    }

    /// Create floating-point square root
    pub fn mk_fp_sqrt(&mut self, rm: RoundingMode, arg: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(arg).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpSqrt(rm, arg), sort)
    }

    /// Create floating-point round to integral
    pub fn mk_fp_round_to_integral(&mut self, rm: RoundingMode, arg: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(arg).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpRoundToIntegral(rm, arg), sort)
    }

    /// Create floating-point addition
    pub fn mk_fp_add(&mut self, rm: RoundingMode, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpAdd(rm, lhs, rhs), sort)
    }

    /// Create floating-point subtraction
    pub fn mk_fp_sub(&mut self, rm: RoundingMode, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpSub(rm, lhs, rhs), sort)
    }

    /// Create floating-point multiplication
    pub fn mk_fp_mul(&mut self, rm: RoundingMode, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpMul(rm, lhs, rhs), sort)
    }

    /// Create floating-point division
    pub fn mk_fp_div(&mut self, rm: RoundingMode, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpDiv(rm, lhs, rhs), sort)
    }

    /// Create floating-point remainder
    pub fn mk_fp_rem(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpRem(lhs, rhs), sort)
    }

    /// Create floating-point minimum
    pub fn mk_fp_min(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpMin(lhs, rhs), sort)
    }

    /// Create floating-point maximum
    pub fn mk_fp_max(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpMax(lhs, rhs), sort)
    }

    /// Create floating-point less than or equal comparison
    pub fn mk_fp_leq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpLeq(lhs, rhs), sort)
    }

    /// Create floating-point less than comparison
    pub fn mk_fp_lt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpLt(lhs, rhs), sort)
    }

    /// Create floating-point greater than or equal comparison
    pub fn mk_fp_geq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpGeq(lhs, rhs), sort)
    }

    /// Create floating-point greater than comparison
    pub fn mk_fp_gt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpGt(lhs, rhs), sort)
    }

    /// Create floating-point equality comparison
    pub fn mk_fp_eq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpEq(lhs, rhs), sort)
    }

    /// Create floating-point fused multiply-add: (x * y) + z
    pub fn mk_fp_fma(&mut self, rm: RoundingMode, x: TermId, y: TermId, z: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(x).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpFma(rm, x, y, z), sort)
    }

    /// Create floating-point is-normal predicate
    pub fn mk_fp_is_normal(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsNormal(arg), sort)
    }

    /// Create floating-point is-subnormal predicate
    pub fn mk_fp_is_subnormal(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsSubnormal(arg), sort)
    }

    /// Create floating-point is-zero predicate
    pub fn mk_fp_is_zero(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsZero(arg), sort)
    }

    /// Create floating-point is-infinite predicate
    pub fn mk_fp_is_infinite(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsInfinite(arg), sort)
    }

    /// Create floating-point is-NaN predicate
    pub fn mk_fp_is_nan(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsNaN(arg), sort)
    }

    /// Create floating-point is-negative predicate
    pub fn mk_fp_is_negative(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsNegative(arg), sort)
    }

    /// Create floating-point is-positive predicate
    pub fn mk_fp_is_positive(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsPositive(arg), sort)
    }

    /// Convert floating-point to another FP format
    pub fn mk_fp_to_fp(&mut self, rm: RoundingMode, arg: TermId, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpToFp { rm, arg, eb, sb }, sort)
    }

    /// Convert floating-point to signed bitvector
    pub fn mk_fp_to_sbv(&mut self, rm: RoundingMode, arg: TermId, width: u32) -> TermId {
        let sort = self.sorts.bitvec(width);
        self.intern(TermKind::FpToSBV { rm, arg, width }, sort)
    }

    /// Convert floating-point to unsigned bitvector
    pub fn mk_fp_to_ubv(&mut self, rm: RoundingMode, arg: TermId, width: u32) -> TermId {
        let sort = self.sorts.bitvec(width);
        self.intern(TermKind::FpToUBV { rm, arg, width }, sort)
    }

    /// Convert floating-point to real
    pub fn mk_fp_to_real(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.real_sort;
        self.intern(TermKind::FpToReal(arg), sort)
    }

    /// Convert real to floating-point
    pub fn mk_real_to_fp(&mut self, rm: RoundingMode, arg: TermId, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::RealToFp { rm, arg, eb, sb }, sort)
    }

    /// Convert signed bitvector to floating-point
    pub fn mk_sbv_to_fp(&mut self, rm: RoundingMode, arg: TermId, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::SBVToFp { rm, arg, eb, sb }, sort)
    }

    /// Convert unsigned bitvector to floating-point
    pub fn mk_ubv_to_fp(&mut self, rm: RoundingMode, arg: TermId, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::UBVToFp { rm, arg, eb, sb }, sort)
    }

    /// Create a function application
    pub fn mk_apply(
        &mut self,
        func: &str,
        args: impl IntoIterator<Item = TermId>,
        sort: SortId,
    ) -> TermId {
        let func_spur = self.intern_str(func);
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();
        self.intern(
            TermKind::Apply {
                func: func_spur,
                args,
            },
            sort,
        )
    }

    // Algebraic datatypes

    /// Create a datatype constructor application
    ///
    /// Constructs a datatype value using the specified constructor.
    /// For example, `cons(1, nil)` for a list.
    pub fn mk_dt_constructor(
        &mut self,
        constructor: &str,
        args: impl IntoIterator<Item = TermId>,
        sort: SortId,
    ) -> TermId {
        let constructor_spur = self.intern_str(constructor);
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();
        self.intern(
            TermKind::DtConstructor {
                constructor: constructor_spur,
                args,
            },
            sort,
        )
    }

    /// Create a datatype tester/discriminator
    ///
    /// Tests if a term was constructed with a specific constructor.
    /// For example, `is-cons(x)` tests if `x` is a cons cell.
    pub fn mk_dt_tester(&mut self, constructor: &str, arg: TermId) -> TermId {
        let constructor_spur = self.intern_str(constructor);
        let bool_sort = self.sorts.bool_sort;
        self.intern(
            TermKind::DtTester {
                constructor: constructor_spur,
                arg,
            },
            bool_sort,
        )
    }

    /// Create a datatype selector/accessor
    ///
    /// Extracts a field from a datatype value.
    /// For example, `head(x)` extracts the first element of a cons cell.
    pub fn mk_dt_selector(&mut self, selector: &str, arg: TermId, result_sort: SortId) -> TermId {
        let selector_spur = self.intern_str(selector);
        self.intern(
            TermKind::DtSelector {
                selector: selector_spur,
                arg,
            },
            result_sort,
        )
    }

    /// Create a universal quantifier without patterns
    pub fn mk_forall<'a>(
        &mut self,
        vars: impl IntoIterator<Item = (&'a str, SortId)>,
        body: TermId,
    ) -> TermId {
        self.mk_forall_with_patterns(vars, body, core::iter::empty::<Vec<TermId>>())
    }

    /// Create a universal quantifier with instantiation patterns
    ///
    /// Patterns are lists of terms that guide quantifier instantiation.
    /// Each pattern is a conjunction of terms that must match for instantiation.
    ///
    /// # Example
    /// ```ignore
    /// // (forall ((x Int)) (! (> (f x) 0) :pattern ((f x))))
    /// let x_var = manager.mk_var("x", int_sort);
    /// let fx = manager.mk_apply("f", [x_var], int_sort);
    /// let body = manager.mk_gt(fx, zero);
    /// let forall = manager.mk_forall_with_patterns(
    ///     [("x", int_sort)],
    ///     body,
    ///     [[fx]],  // pattern: (f x)
    /// );
    /// ```
    pub fn mk_forall_with_patterns<'a, P, Q>(
        &mut self,
        vars: impl IntoIterator<Item = (&'a str, SortId)>,
        body: TermId,
        patterns: P,
    ) -> TermId
    where
        P: IntoIterator<Item = Q>,
        Q: IntoIterator<Item = TermId>,
    {
        use crate::interner::Spur;
        let vars: SmallVec<[(Spur, SortId); 2]> = vars
            .into_iter()
            .map(|(name, sort)| (self.intern_str(name), sort))
            .collect();

        if vars.is_empty() {
            return body;
        }

        let patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
            .into_iter()
            .map(|p| p.into_iter().collect())
            .collect();

        let sort = self.sorts.bool_sort;
        self.intern(
            TermKind::Forall {
                vars,
                body,
                patterns,
            },
            sort,
        )
    }

    /// Create an existential quantifier without patterns
    pub fn mk_exists<'a>(
        &mut self,
        vars: impl IntoIterator<Item = (&'a str, SortId)>,
        body: TermId,
    ) -> TermId {
        self.mk_exists_with_patterns(vars, body, core::iter::empty::<Vec<TermId>>())
    }

    /// Create an existential quantifier with instantiation patterns
    pub fn mk_exists_with_patterns<'a, P, Q>(
        &mut self,
        vars: impl IntoIterator<Item = (&'a str, SortId)>,
        body: TermId,
        patterns: P,
    ) -> TermId
    where
        P: IntoIterator<Item = Q>,
        Q: IntoIterator<Item = TermId>,
    {
        use crate::interner::Spur;
        let vars: SmallVec<[(Spur, SortId); 2]> = vars
            .into_iter()
            .map(|(name, sort)| (self.intern_str(name), sort))
            .collect();

        if vars.is_empty() {
            return body;
        }

        let patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
            .into_iter()
            .map(|p| p.into_iter().collect())
            .collect();

        let sort = self.sorts.bool_sort;
        self.intern(
            TermKind::Exists {
                vars,
                body,
                patterns,
            },
            sort,
        )
    }

    /// Create a let expression
    pub fn mk_let<'a>(
        &mut self,
        bindings: impl IntoIterator<Item = (&'a str, TermId)>,
        body: TermId,
    ) -> TermId {
        use crate::interner::Spur;
        let bindings: SmallVec<[(Spur, TermId); 2]> = bindings
            .into_iter()
            .map(|(name, term)| (self.intern_str(name), term))
            .collect();

        if bindings.is_empty() {
            return body;
        }

        let sort = self.get(body).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::Let { bindings, body }, sort)
    }

    // BitVector operations

    /// Describe an operand's sort for a builder-level type error.
    ///
    /// Falls back to a marker rather than a plausible-looking sort name when
    /// the term or its sort cannot be resolved — the whole point of the
    /// callers that use this is to stop guessing about unresolvable sorts.
    fn describe_operand_sort(&self, term: TermId) -> String {
        match self.get(term) {
            None => format!("<unknown term #{}>", term.0),
            Some(t) => self
                .sorts
                .sort_name(t.sort)
                .unwrap_or_else(|| format!("<unknown sort #{}>", t.sort.0)),
        }
    }

    /// Create a bit vector concatenation.
    ///
    /// Both operands must have a bit-vector sort; the result width is
    /// exactly the sum of the operand widths, per SMT-LIB
    /// `FixedSizeBitVectors` semantics.
    ///
    /// This is the first `Result`-returning `mk_*` constructor in this file,
    /// and it sets the rule for the ones that follow: a builder is
    /// **fallible** when the caller may pass a term whose sort it did not
    /// itself establish (the SMT-LIB parser, the FFI/language bindings), and
    /// **infallible** when the caller owns well-sortedness by construction.
    /// `concat` is in the first group, so it reports a bad operand rather
    /// than guessing at one.
    ///
    /// It replaces an infallible `mk_bv_concat` that defaulted an
    /// unresolvable operand's width to a fabricated `32` in release builds
    /// (a `debug_assert!` caught the violation only under debug assertions).
    /// A fabricated width is not cosmetic: it interns a well-formed term at
    /// the wrong sort, which can flip a query's answer between `sat` and
    /// `unsat`.
    ///
    /// # Errors
    ///
    /// Returns [`OxizError::SortMismatchSimple`] naming **both** operand
    /// sorts when either operand is not bit-vector sorted. The
    /// location-carrying `SortMismatch`/`TypeError` variants are
    /// deliberately not used here: the term manager holds no source span to
    /// put in them, and inventing one would aim a diagnostic at the wrong
    /// place. `SortMismatchSimple` is the enum's existing span-free
    /// type-error variant, so no new variant is introduced.
    pub fn try_mk_bv_concat(&mut self, lhs: TermId, rhs: TermId) -> Result<TermId> {
        let lhs_width = self
            .get(lhs)
            .and_then(|t| self.sorts.get(t.sort))
            .and_then(|s| s.bitvec_width());
        let rhs_width = self
            .get(rhs)
            .and_then(|t| self.sorts.get(t.sort))
            .and_then(|s| s.bitvec_width());
        let (Some(lhs_width), Some(rhs_width)) = (lhs_width, rhs_width) else {
            return Err(OxizError::SortMismatchSimple {
                expected: "two bit-vector operands for concat".to_string(),
                found: format!(
                    "lhs: {}, rhs: {}",
                    self.describe_operand_sort(lhs),
                    self.describe_operand_sort(rhs)
                ),
            });
        };
        let width = lhs_width + rhs_width;

        // Both halves literal: splice them into a single literal.
        if let Some(lhs_value) = self.bv_const_unsigned(lhs, lhs_width)
            && let Some(rhs_value) = self.bv_const_unsigned(rhs, rhs_width)
        {
            return Ok(self.mk_bitvec(bv_fold::bv_concat(&lhs_value, &rhs_value, rhs_width), width));
        }

        let sort = self.sorts.bitvec(width);
        Ok(self.intern(TermKind::BvConcat(lhs, rhs), sort))
    }

    /// Create a bit vector NAND: `bvnand(a, b) = bvnot(bvand(a, b))`.
    pub fn mk_bv_nand(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let and = self.mk_bv_and(lhs, rhs);
        self.mk_bv_not(and)
    }

    /// Create a bit vector NOR: `bvnor(a, b) = bvnot(bvor(a, b))`.
    pub fn mk_bv_nor(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let or = self.mk_bv_or(lhs, rhs);
        self.mk_bv_not(or)
    }

    /// Create a bit vector XNOR: `bvxnor(a, b) = bvnot(bvxor(a, b))`.
    pub fn mk_bv_xnor(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let xor = self.mk_bv_xor(lhs, rhs);
        self.mk_bv_not(xor)
    }

    /// Create a bit vector comparison: a 1-bit result that is `#b1` when
    /// the two (equal-width) operands are equal and `#b0` otherwise, per
    /// SMT-LIB `FixedSizeBitVectors` `bvcomp`.
    pub fn mk_bv_comp(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let eq = self.mk_eq(lhs, rhs);
        let one = self.mk_bitvec(1i64, 1);
        let zero = self.mk_bitvec(0i64, 1);
        self.mk_ite(eq, one, zero)
    }

    /// Create a signed bit-vector modulo (`bvsmod`), whose result sign
    /// follows the *divisor* `rhs` — distinct from `bvsrem`, whose result
    /// sign follows the dividend. Implements the standard SMT-LIB
    /// `FixedSizeBitVectors` definition by reducing to the unsigned
    /// remainder over the operands' absolute values and then reintroducing
    /// the sign according to the operand-sign combination:
    ///
    /// ```text
    /// u = bvurem(abs(s), abs(t))
    /// bvsmod(s, t) = u                    if u = 0
    ///              = u                    if sign(s) = sign(t) = +
    ///              = -u + t               if sign(s) = -, sign(t) = +
    ///              = u + t                if sign(s) = +, sign(t) = -
    ///              = -u                   if sign(s) = sign(t) = -
    /// ```
    ///
    /// Two literal operands are folded directly instead of being expanded
    /// into that `ite` chain.  The chain would collapse to the same constant
    /// on its own (every condition becomes literal), but evaluating it here
    /// keeps the definition of the total zero-divisor case — `bvsmod s 0` is
    /// `s` — in one auditable place alongside the rest of the division
    /// family.
    pub fn mk_bv_smod(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let width = self
            .get(lhs)
            .and_then(|t| self.sorts.get(t.sort))
            .and_then(|s| s.bitvec_width())
            .unwrap_or(32);
        if width == 0 {
            return self.mk_bv_urem(lhs, rhs);
        }
        if let Some((lhs_value, rhs_value)) = self.bv_const_pair(lhs, rhs, width) {
            return self.mk_bitvec(bv_fold::bv_smod(&lhs_value, &rhs_value, width), width);
        }
        let msb = width - 1;
        let msb_s = self.mk_bv_extract(msb, msb, lhs);
        let msb_t = self.mk_bv_extract(msb, msb, rhs);
        let zero_bit = self.mk_bitvec(0i64, 1);
        let s_nonneg = self.mk_eq(msb_s, zero_bit);
        let t_nonneg = self.mk_eq(msb_t, zero_bit);
        let not_s_nonneg = self.mk_not(s_nonneg);
        let not_t_nonneg = self.mk_not(t_nonneg);

        let neg_s = self.mk_bv_neg(lhs);
        let neg_t = self.mk_bv_neg(rhs);
        let abs_s = self.mk_ite(s_nonneg, lhs, neg_s);
        let abs_t = self.mk_ite(t_nonneg, rhs, neg_t);
        let u = self.mk_bv_urem(abs_s, abs_t);

        let zero_w = self.mk_bitvec(0i64, width);
        let u_is_zero = self.mk_eq(u, zero_w);
        let neg_u = self.mk_bv_neg(u);
        let u_plus_t = self.mk_bv_add(u, rhs);
        let negu_plus_t = self.mk_bv_add(neg_u, rhs);

        let both_nonneg = self.mk_and([s_nonneg, t_nonneg]);
        let s_neg_t_nonneg = self.mk_and([not_s_nonneg, t_nonneg]);
        let s_nonneg_t_neg = self.mk_and([s_nonneg, not_t_nonneg]);

        // Innermost: both negative -> -u.
        let case_both_neg = neg_u;
        // s non-negative, t negative -> u + t.
        let case3 = self.mk_ite(s_nonneg_t_neg, u_plus_t, case_both_neg);
        // s negative, t non-negative -> -u + t.
        let case2 = self.mk_ite(s_neg_t_nonneg, negu_plus_t, case3);
        // Both non-negative -> u.
        let case1 = self.mk_ite(both_nonneg, u, case2);
        // u = 0 -> u.
        self.mk_ite(u_is_zero, u, case1)
    }

    /// Create a bit vector extraction.
    ///
    /// Callers (in particular the SMT-LIB parser lowering `(_ extract i j)`)
    /// must ensure `low <= high` and `high < width(arg)` *before* calling this
    /// so the resulting term is semantically meaningful. As defense in depth
    /// against malformed indices reaching this far (which would otherwise
    /// underflow `high - low + 1` — a panic in debug builds and a ~4-billion
    /// bit sort in release builds), the width computation uses checked
    /// arithmetic and falls back to a minimal 1-bit result instead of
    /// panicking or wrapping.
    pub fn mk_bv_extract(&mut self, high: u32, low: u32, arg: TermId) -> TermId {
        let width = high
            .checked_sub(low)
            .and_then(|span| span.checked_add(1))
            .unwrap_or(1);

        // A literal operand yields a literal slice, provided the indices are
        // in range — malformed indices are left for the parser's sort check
        // rather than silently folded to a fabricated value.
        if low <= high
            && let Some(arg_width) = self.bv_width_of(arg)
            && high < arg_width
            && let Some(value) = self.bv_const_unsigned(arg, arg_width)
        {
            return self.mk_bitvec(bv_fold::bv_extract(&value, high, low), width);
        }

        let sort = self.sorts.bitvec(width);
        self.intern(TermKind::BvExtract { high, low, arg }, sort)
    }

    /// Create a bit vector NOT.
    ///
    /// Folds a literal operand and collapses `bvnot (bvnot t)` to `t`.
    pub fn mk_bv_not(&mut self, arg: TermId) -> TermId {
        if let Some(width) = self.bv_width_of(arg).filter(|width| *width > 0) {
            if let Some(value) = self.bv_const_unsigned(arg, width) {
                return self.mk_bitvec(bv_fold::bv_not(&value, width), width);
            }
            // bvnot (bvnot t) -> t: complement is an involution.
            if let Some(term) = self.get(arg)
                && let TermKind::BvNot(inner) = term.kind
            {
                return inner;
            }
        }

        let sort = self.get(arg).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvNot(arg), sort)
    }

    /// Create a bit vector AND.
    ///
    /// Folds two literals and applies `t & t -> t`, `t & 0 -> 0` and
    /// `t & all-ones -> t`.
    pub fn mk_bv_and(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let (lhs, rhs) = canonical_pair(lhs, rhs);
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            let (lhs_value, rhs_value) = self.bv_operand_consts(lhs, rhs, width);
            if let (Some(lhs_value), Some(rhs_value)) = (&lhs_value, &rhs_value) {
                return self.mk_bitvec(bv_fold::bv_and(lhs_value, rhs_value, width), width);
            }
            if lhs == rhs {
                return lhs;
            }
            let all_ones = bv_fold::all_ones(width);
            if let Some(lhs_value) = &lhs_value {
                if *lhs_value == BigInt::ZERO {
                    return lhs;
                }
                if *lhs_value == all_ones {
                    return rhs;
                }
            }
            if let Some(rhs_value) = &rhs_value {
                if *rhs_value == BigInt::ZERO {
                    return rhs;
                }
                if *rhs_value == all_ones {
                    return lhs;
                }
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvAnd(lhs, rhs), sort)
    }

    /// Create a bit vector OR.
    ///
    /// Folds two literals and applies `t | t -> t`, `t | 0 -> t` and
    /// `t | all-ones -> all-ones`.
    pub fn mk_bv_or(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let (lhs, rhs) = canonical_pair(lhs, rhs);
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            let (lhs_value, rhs_value) = self.bv_operand_consts(lhs, rhs, width);
            if let (Some(lhs_value), Some(rhs_value)) = (&lhs_value, &rhs_value) {
                return self.mk_bitvec(bv_fold::bv_or(lhs_value, rhs_value, width), width);
            }
            if lhs == rhs {
                return lhs;
            }
            let all_ones = bv_fold::all_ones(width);
            if let Some(lhs_value) = &lhs_value {
                if *lhs_value == BigInt::ZERO {
                    return rhs;
                }
                if *lhs_value == all_ones {
                    return lhs;
                }
            }
            if let Some(rhs_value) = &rhs_value {
                if *rhs_value == BigInt::ZERO {
                    return lhs;
                }
                if *rhs_value == all_ones {
                    return rhs;
                }
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvOr(lhs, rhs), sort)
    }

    /// Create a bit vector addition.
    ///
    /// Folds two literals and applies `t + 0 -> t`.
    pub fn mk_bv_add(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let (lhs, rhs) = canonical_pair(lhs, rhs);
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            let (lhs_value, rhs_value) = self.bv_operand_consts(lhs, rhs, width);
            if let (Some(lhs_value), Some(rhs_value)) = (&lhs_value, &rhs_value) {
                return self.mk_bitvec(bv_fold::bv_add(lhs_value, rhs_value, width), width);
            }
            if lhs_value.is_some_and(|value| value == BigInt::ZERO) {
                return rhs;
            }
            if rhs_value.is_some_and(|value| value == BigInt::ZERO) {
                return lhs;
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvAdd(lhs, rhs), sort)
    }

    /// Create a bit vector subtraction.
    ///
    /// Folds two literals and applies `t - t -> 0` and `t - 0 -> t`.
    pub fn mk_bv_sub(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            if let Some((lhs_value, rhs_value)) = self.bv_const_pair(lhs, rhs, width) {
                return self.mk_bitvec(bv_fold::bv_sub(&lhs_value, &rhs_value, width), width);
            }
            if lhs == rhs {
                return self.mk_bitvec(0i64, width);
            }
            if self
                .bv_const_unsigned(rhs, width)
                .is_some_and(|value| value == BigInt::ZERO)
            {
                return lhs;
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvSub(lhs, rhs), sort)
    }

    /// Create a bit vector multiplication.
    ///
    /// Folds two literals and applies `t * 0 -> 0` and `t * 1 -> t`.
    pub fn mk_bv_mul(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let (lhs, rhs) = canonical_pair(lhs, rhs);
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            let (lhs_value, rhs_value) = self.bv_operand_consts(lhs, rhs, width);
            if let (Some(lhs_value), Some(rhs_value)) = (&lhs_value, &rhs_value) {
                return self.mk_bitvec(bv_fold::bv_mul(lhs_value, rhs_value, width), width);
            }
            let one = BigInt::from(1u8);
            if let Some(lhs_value) = &lhs_value {
                if *lhs_value == BigInt::ZERO {
                    return lhs;
                }
                if *lhs_value == one {
                    return rhs;
                }
            }
            if let Some(rhs_value) = &rhs_value {
                if *rhs_value == BigInt::ZERO {
                    return rhs;
                }
                if *rhs_value == one {
                    return lhs;
                }
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvMul(lhs, rhs), sort)
    }

    /// Width of `term` when it is bit-vector sorted.
    fn bv_width_of(&self, term: TermId) -> Option<u32> {
        let sort = self.get(term)?.sort;
        self.sorts.get(sort)?.bitvec_width()
    }

    /// Value of `term` when it is a bit-vector literal, normalised into the
    /// unsigned range `[0, 2^width)`.
    fn bv_const_unsigned(&self, term: TermId, width: u32) -> Option<BigInt> {
        let TermKind::BitVecConst { value, .. } = &self.get(term)?.kind else {
            return None;
        };
        Some(bv_fold::bv_wrap_unsigned(value, width))
    }

    /// Both operands' values, when both are bit-vector literals.
    fn bv_const_pair(&self, lhs: TermId, rhs: TermId, width: u32) -> Option<(BigInt, BigInt)> {
        Some((
            self.bv_const_unsigned(lhs, width)?,
            self.bv_const_unsigned(rhs, width)?,
        ))
    }

    /// Each operand's value, or `None` where it is not a literal.
    ///
    /// Normalising a literal allocates, so the identity rules below take both
    /// values once from here instead of re-reading each operand per rule.
    fn bv_operand_consts(
        &self,
        lhs: TermId,
        rhs: TermId,
        width: u32,
    ) -> (Option<BigInt>, Option<BigInt>) {
        (
            self.bv_const_unsigned(lhs, width),
            self.bv_const_unsigned(rhs, width),
        )
    }

    /// The common width of a binary bit-vector operator's operands, when it is
    /// a usable (non-degenerate) bit-vector width.
    ///
    /// Constant folding is skipped for width `0`, which is not a legal
    /// SMT-LIB bit-vector sort and therefore never carries a meaningful value.
    fn bv_binop_width(&self, lhs: TermId, rhs: TermId) -> Option<u32> {
        self.bv_width_of(lhs)
            .or_else(|| self.bv_width_of(rhs))
            .filter(|width| *width > 0)
    }

    /// Decide a bit-vector comparison that holds (or fails) for *every*
    /// assignment, returning the constant truth value it folds to.
    ///
    /// Reference: Z3's `bv_rewriter.cpp`, which folds exactly these atoms.
    /// Without them, an assertion like `(bvult x #b00000000)` — false for every
    /// `x`, since nothing is unsigned-less-than zero — survives as an
    /// unconstrained boolean atom and the solver answers a spurious `sat`.
    ///
    /// The rules, for width `w` with `MAX_U = 2^w - 1`, `MIN_S = -2^(w-1)` and
    /// `MAX_S = 2^(w-1) - 1`:
    ///
    /// * `t <u t`, `t <s t` → `false`; `t <=u t`, `t <=s t` → `true`.
    /// * `t <u 0`, `MAX_U <u t`, `t <s MIN_S`, `MAX_S <s t` → `false`.
    /// * `0 <=u t`, `t <=u MAX_U`, `MIN_S <=s t`, `t <=s MAX_S` → `true`.
    /// * both operands literal → evaluate directly.
    ///
    /// `signed` selects the two's-complement order, `strict` selects `<` over
    /// `<=`.  Returns `None` when the atom is not decidable syntactically.
    fn fold_bv_compare(
        &self,
        lhs: TermId,
        rhs: TermId,
        signed: bool,
        strict: bool,
    ) -> Option<bool> {
        let width = self.bv_width_of(lhs).or_else(|| self.bv_width_of(rhs))?;
        if width == 0 {
            return None;
        }

        // Both orders are total and reflexive, so `t < t` is false and
        // `t <= t` is true for any term — hash-consing makes the syntactic
        // identity check exact.
        if lhs == rhs {
            return Some(!strict);
        }

        // Reinterpret an unsigned literal under the selected order.
        let in_order = |v: BigInt| -> BigInt {
            if signed && v >= (BigInt::from(1u8) << (width - 1) as usize) {
                v - (BigInt::from(1u8) << width as usize)
            } else {
                v
            }
        };
        let lhs_const = self.bv_const_unsigned(lhs, width).map(&in_order);
        let rhs_const = self.bv_const_unsigned(rhs, width).map(&in_order);

        if let (Some(l), Some(r)) = (&lhs_const, &rhs_const) {
            return Some(if strict { l < r } else { l <= r });
        }

        let (min_value, max_value) = if signed {
            (
                -(BigInt::from(1u8) << (width - 1) as usize),
                (BigInt::from(1u8) << (width - 1) as usize) - 1,
            )
        } else {
            (BigInt::ZERO, (BigInt::from(1u8) << width as usize) - 1)
        };

        if let Some(r) = &rhs_const {
            // `t < MIN` is unsatisfiable; `t <= MAX` is a tautology.
            if strict && *r == min_value {
                return Some(false);
            }
            if !strict && *r == max_value {
                return Some(true);
            }
        }
        if let Some(l) = &lhs_const {
            // `MAX < t` is unsatisfiable; `MIN <= t` is a tautology.
            if strict && *l == max_value {
                return Some(false);
            }
            if !strict && *l == min_value {
                return Some(true);
            }
        }

        None
    }

    /// Create a bit vector unsigned less-than
    pub fn mk_bv_ult(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(value) = self.fold_bv_compare(lhs, rhs, false, true) {
            return self.mk_bool(value);
        }
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::BvUlt(lhs, rhs), sort)
    }

    /// Create a bit vector signed less-than
    pub fn mk_bv_slt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(value) = self.fold_bv_compare(lhs, rhs, true, true) {
            return self.mk_bool(value);
        }
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::BvSlt(lhs, rhs), sort)
    }

    /// Create a bit vector unsigned less-than-or-equal
    pub fn mk_bv_ule(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(value) = self.fold_bv_compare(lhs, rhs, false, false) {
            return self.mk_bool(value);
        }
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::BvUle(lhs, rhs), sort)
    }

    /// Create a bit vector signed less-than-or-equal
    pub fn mk_bv_sle(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(value) = self.fold_bv_compare(lhs, rhs, true, false) {
            return self.mk_bool(value);
        }
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::BvSle(lhs, rhs), sort)
    }

    /// Create a bit vector negation (two's complement).
    ///
    /// Lowered to `0 - arg` through [`Self::mk_bv_sub`], so a literal operand
    /// is folded by the same rule that folds subtraction.
    pub fn mk_bv_neg(&mut self, arg: TermId) -> TermId {
        // Get the width from the argument's sort
        let sort = self.get(arg).map_or(self.sorts.bool_sort, |t| t.sort);
        let width = self
            .sorts
            .get(sort)
            .and_then(|s| s.bitvec_width())
            .unwrap_or(32);
        let zero = self.mk_bitvec(0i64, width);
        self.mk_bv_sub(zero, arg)
    }

    /// Create an unsigned bit vector division.
    ///
    /// Folds two literals, including the **total** division-by-zero case
    /// `(bvudiv s (_ bv0 m))` = all ones.
    pub fn mk_bv_udiv(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(width) = self.bv_binop_width(lhs, rhs)
            && let Some((lhs_value, rhs_value)) = self.bv_const_pair(lhs, rhs, width)
        {
            return self.mk_bitvec(bv_fold::bv_udiv(&lhs_value, &rhs_value, width), width);
        }
        let sort = self.get(lhs).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::BvUdiv(lhs, rhs), sort)
    }

    /// Create a signed bit vector division.
    ///
    /// Folds two literals, including the **total** division-by-zero case
    /// `(bvsdiv s (_ bv0 m))` = `-1` for non-negative `s` and `1` otherwise.
    pub fn mk_bv_sdiv(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(width) = self.bv_binop_width(lhs, rhs)
            && let Some((lhs_value, rhs_value)) = self.bv_const_pair(lhs, rhs, width)
        {
            return self.mk_bitvec(bv_fold::bv_sdiv(&lhs_value, &rhs_value, width), width);
        }
        let sort = self.get(lhs).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::BvSdiv(lhs, rhs), sort)
    }

    /// Create an unsigned bit vector remainder.
    ///
    /// Folds two literals, including the **total** remainder-by-zero case
    /// `(bvurem s (_ bv0 m))` = `s`.
    pub fn mk_bv_urem(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(width) = self.bv_binop_width(lhs, rhs)
            && let Some((lhs_value, rhs_value)) = self.bv_const_pair(lhs, rhs, width)
        {
            return self.mk_bitvec(bv_fold::bv_urem(&lhs_value, &rhs_value, width), width);
        }
        let sort = self.get(lhs).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::BvUrem(lhs, rhs), sort)
    }

    /// Create a signed bit vector remainder.
    ///
    /// Folds two literals, including the **total** remainder-by-zero case
    /// `(bvsrem s (_ bv0 m))` = `s`.
    pub fn mk_bv_srem(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(width) = self.bv_binop_width(lhs, rhs)
            && let Some((lhs_value, rhs_value)) = self.bv_const_pair(lhs, rhs, width)
        {
            return self.mk_bitvec(bv_fold::bv_srem(&lhs_value, &rhs_value, width), width);
        }
        let sort = self.get(lhs).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::BvSrem(lhs, rhs), sort)
    }

    /// Create a bit vector XOR.
    ///
    /// Folds two literals and applies `t ^ t -> 0` and `t ^ 0 -> t`.
    pub fn mk_bv_xor(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let (lhs, rhs) = canonical_pair(lhs, rhs);
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            let (lhs_value, rhs_value) = self.bv_operand_consts(lhs, rhs, width);
            if let (Some(lhs_value), Some(rhs_value)) = (&lhs_value, &rhs_value) {
                return self.mk_bitvec(bv_fold::bv_xor(lhs_value, rhs_value, width), width);
            }
            if lhs == rhs {
                return self.mk_bitvec(0i64, width);
            }
            if lhs_value.is_some_and(|value| value == BigInt::ZERO) {
                return rhs;
            }
            if rhs_value.is_some_and(|value| value == BigInt::ZERO) {
                return lhs;
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvXor(lhs, rhs), sort)
    }

    /// Create a bit vector shift left.
    ///
    /// Folds two literals; a shift distance of at least the width discards
    /// every bit, so `t << k` is `0` for any `t` once `k >= width`.
    pub fn mk_bv_shl(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            match self.fold_bv_shift(lhs, rhs, width, bv_fold::bv_shl) {
                ShiftFold::Value(value) => return self.mk_bitvec(value, width),
                ShiftFold::Identity => return lhs,
                ShiftFold::None => {}
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvShl(lhs, rhs), sort)
    }

    /// Create a bit vector logical shift right.
    ///
    /// Folds two literals; a shift distance of at least the width discards
    /// every bit, so `t >>u k` is `0` for any `t` once `k >= width`.
    pub fn mk_bv_lshr(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            match self.fold_bv_shift(lhs, rhs, width, bv_fold::bv_lshr) {
                ShiftFold::Value(value) => return self.mk_bitvec(value, width),
                ShiftFold::Identity => return lhs,
                ShiftFold::None => {}
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvLshr(lhs, rhs), sort)
    }

    /// Create a bit vector arithmetic shift right.
    ///
    /// Folds two literals.  Unlike the other two shifts, an over-wide
    /// distance does *not* fold on its own: `t >>s k` for `k >= width` is
    /// all-ones or zero depending on `t`'s sign bit, so it is only decidable
    /// when `t` is also literal.
    pub fn mk_bv_ashr(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if let Some(width) = self.bv_binop_width(lhs, rhs) {
            if let Some((lhs_value, rhs_value)) = self.bv_const_pair(lhs, rhs, width) {
                let folded = bv_fold::bv_ashr(&lhs_value, &rhs_value, width);
                return self.mk_bitvec(folded, width);
            }
            // t >>s 0 -> t.
            if self
                .bv_const_unsigned(rhs, width)
                .is_some_and(|amount| amount == BigInt::ZERO)
            {
                return lhs;
            }
        }

        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvAshr(lhs, rhs), sort)
    }

    /// Shared folding for `bvshl` and `bvlshr`, whose results agree on the
    /// two operand-independent cases: shifting by `0` is the identity, and
    /// shifting by at least the width yields `0` regardless of the value.
    ///
    /// `fold_const` is the matching evaluator from
    /// [`super::bv_fold`] — an ordinary Rust function pointer chosen at the
    /// call site, not any form of dynamic evaluation.
    fn fold_bv_shift(
        &self,
        lhs: TermId,
        rhs: TermId,
        width: u32,
        fold_const: fn(&BigInt, &BigInt, u32) -> BigInt,
    ) -> ShiftFold {
        let Some(amount) = self.bv_const_unsigned(rhs, width) else {
            return ShiftFold::None;
        };
        if let Some(value) = self.bv_const_unsigned(lhs, width) {
            return ShiftFold::Value(fold_const(&value, &amount, width));
        }
        if amount == BigInt::ZERO {
            return ShiftFold::Identity;
        }
        if amount >= BigInt::from(width) {
            return ShiftFold::Value(BigInt::ZERO);
        }
        ShiftFold::None
    }
}

/// Outcome of folding a `bvshl` / `bvlshr` whose shift distance is literal.
enum ShiftFold {
    /// The whole shift evaluates to this literal value.
    Value(BigInt),
    /// The shift is the identity on its left operand (a zero distance).
    Identity,
    /// Nothing can be decided syntactically.
    None,
}
